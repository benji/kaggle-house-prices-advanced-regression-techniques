import pandas as pd
import math
import json
import sys
from tqdm import tqdm
import numpy as np
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from utils import *
from Training import *
from Autotrain import *
from kaggle import *

FAIL_SCORE = 9999

NUMERICAL_NONE = 'none'
NUMERICAL_EXTRACT_0 = 'extract_0'
NUMERICAL_LINEARIZE_ORDER_2 = 'linearize_order_2'
NUMERICAL_LINEARIZE_ORDER_3 = 'linearize_order_3'
NUMERICAL_EXTRACT_0_LINEARIZE_ORDER_2 = 'extract_0_linearize_order_2'
NUMERICAL_QUANTILE_BIN_5 = 'quantile_bin(5)'
NUMERICAL_QUANTILE_BIN_10 = 'quantile_bin(10)'
NUMERICAL_QUANTILE_BIN_20 = 'quantile_bin(20)'
NUMERICAL_TECHNIQUES = [NUMERICAL_NONE,
                        NUMERICAL_EXTRACT_0,
                        NUMERICAL_LINEARIZE_ORDER_2,
                        NUMERICAL_EXTRACT_0_LINEARIZE_ORDER_2,
                        NUMERICAL_LINEARIZE_ORDER_3,
                        NUMERICAL_QUANTILE_BIN_5,
                        NUMERICAL_QUANTILE_BIN_10,
                        NUMERICAL_QUANTILE_BIN_20]

CATEGORICAL_ONEHOT = 'one_hot'
CATEGORICAL_LABEL_ENCODING = 'label_encoding'
CATEGORICAL_TARGET_MEAN = 'target_mean'
CATEGORICAL_TARGET_MEDIAN = 'target_median'
CATEGORICAL_TECHNIQUES = [CATEGORICAL_ONEHOT,
                          CATEGORICAL_LABEL_ENCODING,
                          CATEGORICAL_TARGET_MEAN,
                          CATEGORICAL_TARGET_MEDIAN]


class Variants:

    def __init__(self, t, score_fn=None, remove_invalid=True, verbose=False):
        self.t_original = t
        self.t_committed = None
        self.score_fn = score_fn
        self.all_individual_variants = self.generate_all_variants_combinations()
        self.commited_variants = []
        self.verbose = verbose

    def remove_invalid_variants(self):
        print 'Testing variants validity...'
        valid_individual_variants = []
        excluded = 0

        for v in self.all_individual_variants:
            t2 = self.get_data_copy_for_columns([v[0]])
            t2.verbose = False

            if not self.apply_variants(t2, [v], fail_on_error=False):
                #print 'Excluding erroneous variant', v
                excluded += 1
            else:
                valid_individual_variants.append(v)

        self.all_individual_variants = valid_individual_variants
        print 'Removed', excluded, 'invalid variants.'

    def generate_all_variants_combinations(self):
        variants = []

        for c in self.t_original.numerical_columns():
            for op in NUMERICAL_TECHNIQUES:
                variants.append([c, op])

        for c in self.t_original.categorical_columns():
            for op in CATEGORICAL_TECHNIQUES:
                variants.append([c, op])

        return variants

    def generate_variants(self, existing_variants=[]):
        existing_cols = [v[0] for v in existing_variants]
        new_variants = [
            v for v in self.all_individual_variants if not v[0] in existing_cols]
        return new_variants

    def apply_variants(self, t2, variants, fail_on_error=True, copy_column=False):
        for variant in variants:
            c = variant[0]

            if copy_column:
                newcol = c+'_'+variant[1]
                t2.duplicate_column(c, newcol)
                c = newcol

            op = variant[1]

            if self.verbose:
                print 'Applying variant', op, 'on column', c

            success = True

            if op == CATEGORICAL_ONEHOT:
                success = t2.dummify_column(c)
            elif op == CATEGORICAL_LABEL_ENCODING:
                success = t2.label_encode_column(c, fail_on_unseen=False)
            elif op == CATEGORICAL_TARGET_MEAN:
                success = t2.replace_categorical_with_mean(c)
            elif op == CATEGORICAL_TARGET_MEDIAN:
                success = t2.replace_categorical_with_median(c)
            elif op == NUMERICAL_EXTRACT_0:
                success = t2.regularize_linear_numerical_singularity(c, 0)
            elif op == NUMERICAL_QUANTILE_BIN_5:
                success = t2.quantile_bin(c, 5)
            elif op == NUMERICAL_QUANTILE_BIN_10:
                success = t2.quantile_bin(c, 10)
            elif op == NUMERICAL_QUANTILE_BIN_20:
                success = t2.quantile_bin(c, 20)
            elif op == NUMERICAL_LINEARIZE_ORDER_2:
                success = t2.linearize_column_with_polynomial_x_transform(c, 2)
            elif op == NUMERICAL_LINEARIZE_ORDER_3:
                success = t2.linearize_column_with_polynomial_x_transform(c, 3)
            elif op == NUMERICAL_EXTRACT_0_LINEARIZE_ORDER_2:
                success = t2.linearize_column_with_polynomial_x_transform(
                    c, 2, 0)

            if not success:
                if fail_on_error:
                    raise Exception(
                        'Failed to applying variant', op, 'on column', c)
                else:
                    if self.verbose:
                        print 'Invalid variant.'
                    return False

        return True

    def score_variants(self, variants, scale=True):
        # print 'scoring variants', variants
        uncommitted_variants = self.get_uncommited_variants(variants)
        uncommitted_cols = [v[0] for v in uncommitted_variants]

        t2 = self.get_data_copy_for_columns(uncommitted_cols)
        t2.verbose = False

        if not self.apply_variants(t2, uncommitted_variants, fail_on_error=False):
            return FAIL_SCORE

        if scale:
            t2.scale()

        return self.score_fn(t2.df_train.values, t2.labels.values)

    def commit_variant(self, v):
        #print 'Commiting variant',v
        if self.t_committed is None:
            self.t_committed = self.t_original.copy()
            self.t_committed.verbose = False
            self.t_committed.df_train = self.t_committed.df_train[[]]
            self.t_committed.df_test = self.t_committed.df_test[[]]

        # add the new column
        c = v[0]
        self.t_committed.df_train[c] = self.t_original.df_train[c].copy()
        self.t_committed.df_test[c] = self.t_original.df_test[c].copy()

        # applies the variant
        if not self.apply_variants(self.t_committed, [v]):
            raise Exception('Could not apply validated variant', v)

        self.commited_variants.append(v)

    def get_uncommited_variants(self, variants):
        if self.t_committed is None:
            return variants
        else:
            return [v for v in variants if v not in self.commited_variants]

    def get_data_copy_for_columns(self, cols):
        if self.t_committed is None:
            t2 = self.t_original.copy()
            t2.df_train = t2.df_train[cols]
            t2.df_test = t2.df_test[cols]
            return t2
        else:
            t2 = self.t_committed.copy()
            for c in cols:
                if c not in t2.df_train.columns:
                    t2.df_train[c] = self.t_original.df_train[c]
                    t2.df_test[c] = self.t_original.df_test[c]
            return t2
