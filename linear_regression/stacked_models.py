import math
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


lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.001, random_state=1))
ENet = make_pipeline(RobustScaler(), ElasticNet(
    alpha=0.0005, l1_ratio=.9, random_state=3))
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state=5)

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
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf=6, min_sum_hessian_in_leaf=11)


models = [lasso, ENet, KRR, GBoost, model_xgb, model_lgb]
stacked_model = clone(model_lgb)

t = training()
t.explode_columns_possibilities()

train_variants = [['OverallQual_numerical', 'extract_0_linearize_order_2'], ['TotalSF', 'linearize_order_3'], ['MSZoning', 'target_median'], ['YearRemodAdd', 'linearize_order_3'], ['OverallCond', 'one_hot'], ['GarageCars_categorical', 'label_encoding'], ['BsmtUnfSF', 'quantile_bin(5)'], ['Fireplaces_categorical', 'target_median'], ['Neighborhood', 'target_median'], ['SaleType', 'target_median'], ['Heating', 'target_mean'], ['GarageArea', 'quantile_bin(5)'], ['BsmtFullBath', 'none'], ['FullBath_categorical', 'one_hot'], ['Electrical', 'label_encoding'], ['YearBuilt', 'extract_0_linearize_order_2'], ['GarageFinish', 'target_median'], ['LotArea', 'quantile_bin(5)'], ['BsmtFinType1', 'label_encoding'], ['FullBath', 'extract_0_linearize_order_2'], ['LowQualFinSF', 'extract_0_linearize_order_2'], ['YrSold', 'none'], ['ScreenPorch', 'linearize_order_2'], ['BsmtFinType2', 'target_median'], ['TotRmsAbvGrd_categorical', 'target_mean'], ['HalfBath', 'extract_0_linearize_order_2'], ['SaleCondition', 'target_mean'], ['BedroomAbvGr', 'linearize_order_2'], ['HeatingQC', 'target_mean'], ['GarageYrBlt', 'quantile_bin(5)'], [
    'CentralAir', 'target_median'], ['RoofStyle', 'target_median'], ['LandContour', 'target_median'], ['BsmtQual', 'one_hot'], ['BsmtCond', 'one_hot'], ['MiscFeature', 'label_encoding'], ['TotRmsAbvGrd', 'linearize_order_3'], ['BsmtFullBath_categorical', 'one_hot'], ['GarageCars', 'linearize_order_2'], ['3SsnPorch', 'extract_0_linearize_order_2'], ['Condition1', 'target_mean'], ['KitchenAbvGr_categorical', 'target_mean'], ['RecentRemodel', 'none'], ['WoodDeckSF', 'none'], ['PavedDrive', 'target_median'], ['2ndFlrSF', 'extract_0_linearize_order_2'], ['DateSold', 'quantile_bin(5)'], ['KitchenAbvGr', 'linearize_order_2'], ['HouseStyle', 'label_encoding'], ['Street', 'target_median'], ['EnclosedPorch', 'none'], ['Age', 'extract_0'], ['GarageCond', 'target_median'], ['BedroomAbvGr_categorical', 'target_median'], ['TotalBsmtSF', 'quantile_bin(5)'], ['TimeSinceSold', 'linearize_order_2'], ['GrLivArea', 'linearize_order_2'], ['MoSold_categorical', 'label_encoding'], ['BsmtExposure', 'target_mean'], ['Remodeled', 'extract_0'], ['MasVnrArea', 'linearize_order_2'], ['OpenPorchSF', 'linearize_order_3']]

variants = Variants(t, verbose=True)

t.retain_columns([v[0] for v in train_variants])
variants.apply_variants(t, train_variants)

t.verify_all_columns_are_numerical()

t.shuffle()
t.scale()
t.health_check()
t.summary()

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

stacked_train_shuffled, y_shuffled = shuffle(stacked_train.values, y)

scaler = StandardScaler()
scaler.fit(stacked_train_shuffled)
stacked_train_shuffled_scaled = scaler.transform(stacked_train_shuffled)

stacked_model_score = clone(stacked_model)
score = score_using_test_ratio(
    stacked_model_score.fit, stacked_model_score.predict, rmse, stacked_train_shuffled_scaled, y_shuffled, test_ratio=.15)
print 'Stacked model RMSE:', score


print "Fitting on whole dataset for predictions..."
stacked_model_pred = clone(stacked_model)

print 'train stacked model for pred against:'
print stacked_train[:5]
stacked_model_pred.fit(stacked_train_shuffled_scaled, y_shuffled)


# test
print 'score test', rmse(stacked_model_pred.predict(
    stacked_train_shuffled_scaled), y_shuffled)


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
    stacked_validation = scaler.transform(stacked_validation.values)

    y_validation_predicted = stacked_model_pred.predict(stacked_validation)
    print y_validation_predicted[:5]
    print y_validation[:5]
    print "Validation RMSE:", rmse(y_validation_predicted, y_validation)

stacked_test = scaler.transform(stacked_test.values)

df_predicted = pd.DataFrame(columns=['Id', 'SalePrice'])
df_predicted['Id'] = t.test_ids
df_predicted.set_index('Id')
df_predicted['SalePrice'] = np.expm1(stacked_model_pred.predict(stacked_test))
df_predicted.to_csv('predicted.csv', sep=',', index=False)

print 'predictions done.'
