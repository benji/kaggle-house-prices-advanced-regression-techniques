import pandas as pd
import math
import json
import sys
from tqdm import tqdm
import numpy as np
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from utils.utils import *
from utils.Training import *
from utils.Autotrain import *
from utils.kaggle import *

t = training()

t.dummify_at_init = False
t.dummify_drop_first = False
t.use_label_encoding = False
#t.train_columns = ['OverallQual', 'OverallCond', 'MSSubClass','MoSold']
#t.train_columns = ['MoSold']
t.explode_possible_types_columns = True
t.prepare()

use_runtime_dummies = True

autotrain = Autotrain(verbose=False)

NUMERICAL_NONE = 'none'
NUMERICAL_EXTRACT_0 = 'extract_0'
NUMERICAL_TECHNIQUES = [NUMERICAL_NONE,
                        NUMERICAL_EXTRACT_0]

CATEGORICAL_ONEHOT = 'one_hot'
CATEGORICAL_LABEL_ENCODING = 'label_encoding'
CATEGORICAL_TARGET_MEAN = 'target_mean'
CATEGORICAL_TECHNIQUES = [CATEGORICAL_ONEHOT,
                          CATEGORICAL_LABEL_ENCODING, CATEGORICAL_TARGET_MEAN]


def generate_variants(existing_variants=[]):
    existing_cols = [v[0] for v in existing_variants]

    variants = []

    for c in t.get_columns():
        if c not in existing_cols:
            variants.extend((generate_variants_for_col(c)))

    return variants


def generate_variants_for_col(c):
    variants = []

    if t.coltype(c) == 'CATEGORICAL':
        for op in CATEGORICAL_TECHNIQUES:
            variants.append([c, op])
    elif t.coltype(c) == 'NUMERIC':
        for op in NUMERICAL_TECHNIQUES:
            variants.append([c, op])

    return variants


def apply_variants(t2, variants):
    cols = [v[0] for v in variants]
    t2.df_train = t2.df_train[cols]
    t2.df_test = t2.df_test[cols]

    for variant in variants:
        c = variant[0]
        op = variant[1]

        if op == CATEGORICAL_ONEHOT:
            t2.do_dummify_column(c, False)
        elif op == CATEGORICAL_LABEL_ENCODING:
            t2.do_label_encode_column(c)
        elif op == CATEGORICAL_TARGET_MEAN:
            t2.replace_categorical_with_mean(c)
        elif op == NUMERICAL_EXTRACT_0:
            t2.numerical_singularities(c, 0)


def score_variant(variant):
    return score_variants([variant])


def score_variants(variants):
    #print 'scoring variants', variants

    t2 = t.copy()
    t2.verbose = False

    apply_variants(t2, variants)

    model = Lasso(alpha=0.0005, random_state=1)
    score_rmse = custom_score_using_kfolds(model.fit,
                                           model.predict,
                                           rmse,
                                           t2.df_train.values,
                                           t2.labels.values,
                                           n_splits=10,
                                           doShuffle=False,
                                           scale=True)

    return score_rmse


score_mean_function = rmse(t.labels.mean(), t.labels.values)

best_variants = [['OverallQual', 'target_mean'], ['TotalSF', 'none'], ['Neighborhood', 'one_hot'], ['OverallCond', 'label_encoding'], ['BsmtFinType1', 'one_hot'], ['GarageCars', 'none'], ['MSSubClass', 'one_hot'], ['LotArea', 'none'], ['YearBuilt', 'none'], ['MSZoning', 'one_hot'], ['BsmtFinSF1', 'none'], ['SaleCondition', 'target_mean'], ['2ndFlrSF', 'extract_0'], ['KitchenQual', 'one_hot'], ['BsmtExposure', 'one_hot'], ['Functional', 'target_mean'], ['GrLivArea', 'none'], ['Condition1', 'one_hot'], ['HeatingQC', 'target_mean'], ['Fireplaces', 'none'], ['ScreenPorch', 'extract_0'], ['BsmtFullBath', 'none'], ['BsmtQual', 'one_hot'], ['CentralAir', 'label_encoding'], ['YearRemodAdd', 'none'], ['PoolArea', 'none'], [
    'Heating', 'one_hot'], ['RoofMatl', 'target_mean'], ['Foundation', 'target_mean'], ['GarageCond', 'target_mean'], ['HalfBath', 'extract_0'], ['FullBath', 'extract_0'], ['KitchenAbvGr', 'none'], ['LotConfig', 'one_hot'], ['ExterQual', 'label_encoding'], ['Exterior1st', 'label_encoding'], ['WoodDeckSF', 'extract_0'], ['Exterior2nd', 'label_encoding'], ['GarageQual', 'one_hot'], ['BsmtFinSF2', 'extract_0'], ['LotFrontage', 'extract_0'], ['HouseStyle', 'target_mean'], ['1stFlrSF', 'none'], ['MiscVal', 'extract_0'], ['GarageArea', 'extract_0'], ['ExterCond', 'label_encoding'], ['MasVnrArea', 'none'], ['MasVnrType', 'label_encoding'], ['MoSold_categorical', 'target_mean'], ['TotalBsmtSF', 'none'], ['BsmtHalfBath', 'none']]


def get_best_variant_for_col(c):
    for v in best_variants:
        if c == v[0]:
            return v[1]


all_variants = []

for c in t.df_train.columns:
    ranks = pd.DataFrame(columns=['technique', 'score', 'diff', 'wtm','best'])
    print 'finding best variant for ', c
    for v in generate_variants_for_col(c):
        score = score_variant(v)
        ranks.loc[ranks.shape[0]] = [v[1], score, 0, '','']

    best_score = ranks['score'].min()
    ranks['diff'] = ranks['score'] - best_score
    ranks.sort_values('score', inplace=True)

    best_variant = get_best_variant_for_col(c)

    bvdiff = ranks['diff'][ranks['technique'] == best_variant].max()
    s=(bvdiff*1000)
    ranks['best'][ranks['technique'] == best_variant] = s

    ranks['wtm'][ranks['score'] > score_mean_function] = '!!!'

    print ranks
    print 'best variant was', best_variant
    print 'mean score:', score_mean_function
    print '-----------------------------'
    #print 'score:', score, 'variant:', best_variant
    # all_variants.append(best_variant)

print 'Done'
sys.exit(0)
print all_variants

best_variants, score = autotrain.find_multiple_best(
    variants=all_variants, score_variants_fn=score_variants, goal='min')

print 'Final variants', best_variants
print 'Final score', score

apply_variants(t, best_variants)

generate_predictions(t.labels, t.df_train, t.df_test, t.test_ids)
print "Predictions complete."
