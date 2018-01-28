import pandas as pd
import numpy as np
import sys

from utils import *


class Training:
    logify_columns = []
    fill_na_value = None
    fill_na_mean = None
    remove_outliers = []
    dummify_at_init = False
    dummify_drop_first = True
    train_columns = []
    dummy_na = False
    quantile_bins = 1

    # will be filled by prepare()
    labels = None
    test_ids = None

    def __init__(self, df_train, df_test=None, schema=None):
        self.df_train = df_train
        self.df_test = df_test
        self.schema = schema

    def prepare(self):
        self.idcol = self.schema['id']
        self.targetcol = self.schema['target']

        # FILL MISSING VALUES
        if self.fill_na_value is not None:
            self.fillna(self.fill_na_value)

        if self.fill_na_mean is not None:
            self._fill_na_mean()

        # REMOVE ROWS
        for o in self.remove_outliers:
            self.remove_outlier(o)

        # TRANSFORM VALUES
        self.df_train[self.targetcol] = self.df_train[self.targetcol].astype(
            'float64')

        for c in self.logify_columns:
            self.logify(c)

        # SAVE IDS & LABELS
        self.test_ids = self.df_test[self.idcol]

        self.labels = self.df_train[self.targetcol]

        # REMOVE COLUMNS
        self.df_train.drop(self.targetcol, 1, inplace=True)

        if (self.idcol in self.df_train.columns):
            self.df_train.drop(self.idcol, 1, inplace=True)
            if self.df_test is not None:
                self.df_test.drop(self.idcol, 1, inplace=True)

        # TRANSFORM COLUMNS
        if len(self.train_columns) > 0:
            self.retain_columns(self.train_columns)

        if self.quantile_bins > 1:
            self.df_train, self.df_test = quantile_bin_all(
                self.schema, self.quantile_bins, self.df_train, self.df_test)
            if not self.dummify_at_init:
                raise Exception(
                    'You need to use dummies of you use quantile_bin')

        if self.dummify_at_init:
            self.df_train, self.df_test = dummify_with_schema(
                self.schema, self.df_train, self.df_test)

        #self.fillna(-99999)

    def retain_columns(self, columns):
        self.retain_columns_df(self.df_train, columns)
        if self.df_test is not None:
            self.retain_columns_df(self.df_test, columns)
        print 'Retained', len(self.df_train.columns), 'columns in dataframe'

    def retain_columns_df(self, df, columns):
        for col in df.columns:
            if (not (col in columns)):
                df.drop(col, 1, inplace=True)

    def _fill_na_mean(self):
        self.df_train.fillna(self.df_train.mean(), inplace=True)
        self.df_test.fillna(self.df_train.mean(), inplace=True)

    def remove_outlier(self, o):
        idx = self.df_train[self.df_train[self.idcol] == o].index
        self.df_train.drop(idx, inplace=True)

    def logify(self, col):
        self.df_train[col] = np.log(self.df_train[col])
        if self.df_test is not None and col in self.df_test.columns:
            self.df_test[col] = np.log(self.df_test[col])

    def fillna(self, val):
        self.df_train.fillna(val, inplace=True)
        if self.df_test is not None:
            self.df_test.fillna(val, inplace=True)

    def separate_out_value(self, col, value, newcol):
        '''
        will create a column with value 1 if this col is == value, 0 otherwise
        matching rows will be set to np.nan in former column
        '''

        def matchesZeroTransform(row):
            return 1 if row[col] == value else 0

        def otherwiseTransform(row):
            return row[col] if row[col] != value else np.nan

        for df in [self.df_train, self.df_test]:
            if df is not None:
                df[newcol] = df.apply(matchesZeroTransform, axis=1)
                #df[col] = df.apply(otherwiseTransform, axis=1)

        self.schema[newcol] = {"categories": [0, 1], "type": "CATEGORICAL"}
