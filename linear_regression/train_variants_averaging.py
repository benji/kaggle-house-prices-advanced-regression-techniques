import math
import json
import time
import sys
import os
import pandas as pd
from os import path
import yaml

from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, BayesianRidge, LassoLarsIC, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.base import clone
import xgboost as xgb
import lightgbm as lgb

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from utils.utils import *
from utils.kaggle import *
from utils.Training import *
from utils.Variants import *
from utils.AveragingModels import *


np.random.seed(int(time.time()))


def seed():
    return 1
#    return np.random.randint(2**32-1)


fit_intercept = True


t = training(exclude_outliers=True, remove_partials=False)
t.shuffle(seed=seed())

for idx in [632, 462, 523, 1298]:
    t.drop_row_by_index(idx)

t.explode_columns_possibilities()

if False:
    train_variants = [['TotalSF', 'linearize_order_2'], ['OverallQual', 'target_mean'], ['Neighborhood', 'one_hot'], ['OverallCond', 'one_hot'], ['BsmtUnfSF', 'quantile_bin(5)'], ['GarageYrBlt', 'quantile_bin(10)'], ['LotArea', 'linearize_order_3'], ['Age', 'extract_0'], ['MSZoning', 'one_hot'], ['FireplaceQu', 'target_mean'], [
        'GarageCars', 'none'], ['Condition1', 'target_mean'], ['SaleCondition', 'one_hot'], ['KitchenQual', 'one_hot'], ['Functional', 'target_median'], ['YearBuilt', 'linearize_order_2'], ['CentralAir', 'target_mean'], ['GrLivArea', 'extract_0_linearize_order_2'], ['BsmtExposure', 'one_hot']]
    train_cols = [v[0] for v in train_variants]
    t.retain_columns(train_cols)
    variants = Variants(t, verbose=True)
    variants.apply_variants(t, train_variants)
else:
    t.dummify_all_categoricals()

t.ready_for_takeoff()
t.scale()
if not fit_intercept:
    t.scale_target()

t.remove_columns_with_unique_value()
t.sanity()
t.summary()
# t.save('../tmp')


def new_model():
    lasso = Lasso(alpha=0.0005, fit_intercept=True,
                  random_state=seed(), warm_start=False, max_iter=10000)
    KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
    ENet = ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=seed())

    GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                       max_depth=4, max_features='sqrt',
                                       min_samples_leaf=15, min_samples_split=10,
                                       loss='huber', random_state=seed())

    model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                                 learning_rate=0.05, max_depth=3,
                                 min_child_weight=1.7817, n_estimators=2200,
                                 reg_alpha=0.4640, reg_lambda=0.8571,
                                 subsample=0.5213, silent=1,
                                 random_state=seed(), nthread=-1)

    model_lgb = lgb.LGBMRegressor(objective='regression', num_leaves=5,
                                  learning_rate=0.05, n_estimators=720,
                                  max_bin=55, bagging_fraction=0.8,
                                  bagging_freq=5, feature_fraction=0.2319,
                                  feature_fraction_seed=seed(), bagging_seed=seed(),
                                  min_data_in_leaf=6, min_sum_hessian_in_leaf=11)

    return AveragingModels(models=(ENet,  KRR, lasso, GBoost, model_xgb, model_lgb))
    # return lasso


m = new_model()

X = t.df_train.values
y = t.labels.values

#X, y = shuffle(X, y)

if True:
    left_out = 500
    X1 = X[:-left_out]
    y1 = y[:-left_out]
    X2 = X[left_out:]
    y2 = y[left_out:]

    m = new_model()
    m.fit(X1, y1)
    weights = [1,2,1,3,4,3]
    predicted = m.predict(X2,weights=weights)
    print 'RMSE unseen data (', left_out, '):', rmse(predicted, y2)
    sys.exit(0)
    np.savetxt('test_predicted.csv', predicted, fmt='%f')
    np.savetxt('test_actual.csv', y2, fmt='%f')

m = new_model()
m.fit(X, y)

# Predictions
df_predicted = pd.DataFrame(columns=['Id', 'SalePrice'])
df_predicted['Id'] = t.test_ids
df_predicted.set_index('Id')

if True:
    predicted = m.predict(t.df_test.values, [2,2,1,3,4,3])
    df_predicted['SalePrice'] = t.untransform_target(predicted)
    df_predicted.to_csv('predicted.csv', sep=',', index=False)
else:
    for i in range(6):
        weights = [1,2,1,3,4,3]
        weights[i] = weights[i] + 1
        predicted = m.predict(t.df_test.values, weights)
        df_predicted['SalePrice'] = t.untransform_target(predicted)
        df_predicted.to_csv('predicted{}.csv'.format(i), sep=',', index=False)

print 'predictions done.'
