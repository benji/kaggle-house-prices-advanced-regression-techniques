import pandas as pd
import math
import json
import sys
import time
from tqdm import tqdm
import numpy as np
from os import path
import matplotlib.pyplot as plt

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


def scoreSingle(X, y_true):
    model = Lasso(alpha=0.001)
    return score_using_test_ratio(
        model.fit, model.predict, rmse, X, y_true, test_ratio=score_test_ratio)

# moving avg 30  -> max +/- 0.0175 diff
# moving avg 60  -> max +/- 0.011 diff
# moving avg 100 -> max +/- 0.008 diff (1000:0.0097)
# moving avg 500 -> max +/- (10000:0.0027)

max_iter = 100
n_moving_average = 5

scores = []
means = []
moving_average = []

for i in range(max_iter):
    s = scoreSingle(t.df_train.values, t.labels.values)
    scores.append(s)
    mean = np.array(scores).mean()
    print mean, i

    means.append(mean)
    if i >= n_moving_average:
        moving_average.append(np.array(scores[-n_moving_average:]).mean())

final_mean = means[-1]

mean_diffs = [np.abs(final_mean-m) for m in means]

moving_average_diff = [np.abs(final_mean-m) for m in moving_average]
moving_average_diff = np.pad(
    moving_average_diff, (0, n_moving_average), 'constant')

xs = range(max_iter)

plt.plot(xs, mean_diffs, 'r-', xs, moving_average_diff, 'g-')
plt.show()
