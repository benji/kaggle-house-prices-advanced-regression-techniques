import pandas as pd
import math
import json
import sys
import time
from tqdm import tqdm
import numpy as np
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from utils.utils import *
from utils.Training import *
from utils.Autotrain import *
from utils.kaggle import *
from utils.Variants import *

np.random.seed(int(time.time()))

t = training()
t.explode_columns_possibilities()

train_variants = [['TotalSF', 'linearize_order_2'], ['OverallQual', 'target_mean'], ['Neighborhood', 'one_hot'], ['OverallCond', 'one_hot'], ['BsmtUnfSF', 'none'], ['Age', 'quantile_bin(10)'], ['LotArea', 'none'], ['GarageCars', 'none'], ['MSZoning', 'one_hot'], ['Fireplaces', 'none'], ['Functional', 'label_encoding'], ['SaleCondition', 'target_mean'], ['Condition1', 'one_hot'], ['GrLivArea', 'linearize_order_2'], ['YearBuilt', 'none'], ['KitchenQual', 'one_hot'], ['BsmtExposure', 'one_hot'], ['KitchenAbvGr', 'none'], ['YearsSinceRemodel', 'extract_0_linearize_order_2'], ['CentralAir', 'one_hot'], ['ScreenPorch', 'extract_0'], ['BsmtQual', 'one_hot'], ['BsmtFinType1', 'label_encoding'], ['PoolArea', 'none'], ['HeatingQC', 'target_mean'], ['LowQualFinSF', 'linearize_order_2'], ['OverallCond_numerical', 'none'], ['BsmtFinType2', 'target_mean'], ['WoodDeckSF', 'extract_0_linearize_order_2'], ['LotConfig', 'one_hot'], ['1stFlrSF', 'quantile_bin(10)'], [
    'GarageQual', 'target_mean'], ['GarageFinish', 'one_hot'], ['Exterior1st', 'label_encoding'], ['YearRemodAdd', 'none'], ['RoofMatl', 'target_mean'], ['FullBath_categorical', 'target_mean'], ['BsmtFullBath', 'none'], ['TotalBsmtSF', 'quantile_bin(5)'], ['BsmtFinSF1', 'linearize_order_2'], ['HalfBath', 'linearize_order_2'], ['2ndFlrSF', 'extract_0_linearize_order_2'], ['GarageArea', 'quantile_bin(5)'], ['OpenPorchSF', 'linearize_order_2'], ['Remodeled', 'extract_0_linearize_order_2'], ['Foundation', 'label_encoding'], ['ExterCond', 'target_mean'], ['BsmtFinSF2', 'linearize_order_2'], ['BsmtHalfBath', 'none'], ['Street', 'one_hot'], ['HouseStyle', 'label_encoding'], ['Exterior2nd', 'label_encoding'], ['SaleType', 'label_encoding'], ['FullBath', 'linearize_order_2'], ['GarageCars_categorical', 'target_mean'], ['RecentRemodel', 'extract_0'], ['Fireplaces_categorical', 'label_encoding'], ['RoofStyle', 'label_encoding'], ['GarageYrBlt', 'extract_0_linearize_order_2']]

train_cols = [v[0] for v in train_variants]
t.retain_columns(train_cols)
variants = Variants(t, verbose=True)
variants.apply_variants(t, train_variants)

t.health_check()

score_test_ratio = 0.1
n_scores_per_variant = 20


def score(X, y_true):
    model = Lasso(alpha=0.001)
    scores = []
    for i in range(n_scores_per_variant):
        scores.append(score_using_test_ratio(
            model.fit, model.predict, rmse, X, y_true, test_ratio=score_test_ratio))
    return np.array(scores[5: -5]).mean()
    # return custom_score_using_kfolds(model.fit, model.predict, rmse, X, y_true, n_splits=10, doShuffle=False, scale=True)


def scoreSingle(X, y_true):
    model = Lasso(alpha=0.001)
    return score_using_test_ratio(
        model.fit, model.predict, rmse, X, y_true, test_ratio=score_test_ratio)


scores = []

i=1
while True:
    s = scoreSingle(t.df_train.values, t.labels.values)
    scores.append(s)
    print np.array(scores).mean(), i
    i+=1
