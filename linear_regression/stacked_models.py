import math
import time
import json
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
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

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
    return np.random.randint(2**32-1)


fit_intercept = True

t = training()
t.explode_columns_possibilities()

if False:
    train_variants = [['TotalSF', 'linearize_order_2'], ['OverallQual', 'target_mean'], ['Neighborhood', 'one_hot'], ['OverallCond_numerical', 'none'], ['BsmtUnfSF', 'quantile_bin(10)'], ['Age', 'quantile_bin(20)'], ['LotArea', 'extract_0_linearize_order_2'], ['GarageCars_categorical', 'label_encoding'], ['MSZoning', 'one_hot'], ['Fireplaces', 'linearize_order_3'], ['Condition1', 'one_hot'], ['Functional', 'label_encoding'], ['SaleCondition', 'target_median'], [
        'CentralAir', 'target_mean'], ['KitchenQual', 'one_hot'], ['GrLivArea', 'extract_0_linearize_order_2'], ['BsmtExposure', 'one_hot'], ['YearBuilt', 'none'], ['KitchenAbvGr_categorical', 'label_encoding'], ['YearsSinceRemodel', 'extract_0_linearize_order_2'], ['HeatingQC', 'target_median'], ['ScreenPorch', 'linearize_order_2'], ['RoofMatl', 'label_encoding'], ['GarageArea', 'linearize_order_3'], ['GarageQual', 'target_median']]
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
t.shuffle()
t.remove_columns_with_unique_value()
t.sanity()
t.summary()
# t.save('../tmp')


lasso = Lasso(alpha=0.0005, random_state=seed())
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
                             random_state=7, nthread=-1)

model_lgb = lgb.LGBMRegressor(objective='regression', num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin=55, bagging_fraction=0.8,
                              bagging_freq=5, feature_fraction=0.2319,
                              feature_fraction_seed=seed(), bagging_seed=seed(),
                              min_data_in_leaf=6, min_sum_hessian_in_leaf=11)


models = [lasso, ENet, KRR, GBoost, model_xgb, model_lgb]
stacked_model = clone(model_lgb)

holdout = 0

X = np.array(t.df_train.values)
y = np.array(t.labels.values)

if holdout > 0:
    X_validation = X[-holdout:]
    y_validation = y[-holdout:]

    X = X[:-holdout]
    y = y[:-holdout]

X_pred = np.array(t.df_test.values)

n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=False)

stacked_train = pd.DataFrame()

model_i = 1
for m in models:
    print 'Model:', model_i
    model_predictions = np.zeros(shape=(len(X)))

    split_i = 1
    for train, test in kf.split(X):
        m_ = clone(m)
        m_.fit(X[train], y[train])
        model_predictions[test] = m_.predict(X[test])
        print 'Prediction on Kfold #', split_i, 'complete.'

        split_i += 1

    stacked_train['model_'+str(model_i)] = model_predictions
    model_i += 1

print 'Completed out of folds predictions.'

stacked_train = stacked_train.values

stacked_model_score = clone(stacked_model)
score = score_using_test_ratio(
    stacked_model_score.fit, stacked_model_score.predict, rmse, stacked_train, y, test_ratio=.4)
print 'Stacked model RMSE with test ratio:', score


print "Fitting on whole dataset for predictions..."
stacked_model_pred = clone(stacked_model)

print 'train stacked model for pred against:'
stacked_model_pred.fit(stacked_train, y)

print 'score test on itself', rmse(
    stacked_model_pred.predict(stacked_train), y)

print "Producing out of folds predictions for test data"
stacked_test = pd.DataFrame()
stacked_validation = pd.DataFrame()

model_i = 1
for m in models:
    m_ = clone(m)
    print 'Model:', model_i
    m_.fit(X, y)

    stacked_test['model_'+str(model_i)] = m_.predict(X_pred)
    if holdout > 0:
        stacked_validation['model_'+str(model_i)] = m_.predict(X_validation)

    model_i += 1


if holdout > 0:
    print stacked_validation.head()
    stacked_validation = stacked_validation.values

    y_validation_predicted = stacked_model_pred.predict(stacked_validation)
    print y_validation_predicted[:5]
    print y_validation[:5]
    print "Validation RMSE:", rmse(y_validation_predicted, y_validation)

stacked_test = stacked_test.values

df_predicted = pd.DataFrame(columns=['Id', 'SalePrice'])
df_predicted['Id'] = t.test_ids
df_predicted.set_index('Id')
predicted = stacked_model_pred.predict(stacked_test)
df_predicted['SalePrice'] = t.untransform_target(predicted)
df_predicted.to_csv('predicted.csv', sep=',', index=False)

print 'predictions done.'
