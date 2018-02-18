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
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from utils.utils import *
from utils.kaggle import *
from utils.Training import *
from utils.Variants import *
from utils.AveragingModels import *

t = training()
t.explode_possible_types_columns = True

train_variants = [['OverallQual', 'target_mean'], ['TotalSF', 'none'], ['Neighborhood', 'one_hot'], ['OverallCond', 'label_encoding'], ['BsmtFinType1', 'one_hot'], ['GarageCars', 'none'], ['MSSubClass', 'one_hot'], ['LotArea', 'none'], ['YearBuilt', 'none'], ['MSZoning', 'one_hot'], ['BsmtFinSF1', 'none'], ['SaleCondition', 'target_mean'], ['2ndFlrSF', 'extract_0'], ['KitchenQual', 'one_hot'], ['BsmtExposure', 'one_hot'], ['Functional', 'target_mean'], ['GrLivArea', 'none'], ['Condition1', 'one_hot'], ['HeatingQC', 'target_mean'], ['Fireplaces', 'none'], ['ScreenPorch', 'extract_0'], ['BsmtFullBath', 'none'], ['BsmtQual', 'one_hot'], ['CentralAir', 'label_encoding'], ['YearRemodAdd', 'none'], ['PoolArea', 'none'], [
    'Heating', 'one_hot'], ['RoofMatl', 'target_mean'], ['Foundation', 'target_mean'], ['GarageCond', 'target_mean'], ['HalfBath', 'extract_0'], ['FullBath', 'extract_0'], ['KitchenAbvGr', 'none'], ['LotConfig', 'one_hot'], ['ExterQual', 'label_encoding'], ['Exterior1st', 'label_encoding'], ['WoodDeckSF', 'extract_0'], ['Exterior2nd', 'label_encoding'], ['GarageQual', 'one_hot'], ['BsmtFinSF2', 'extract_0'], ['LotFrontage', 'extract_0'], ['HouseStyle', 'target_mean'], ['1stFlrSF', 'none'], ['MiscVal', 'extract_0'], ['GarageArea', 'extract_0'], ['ExterCond', 'label_encoding'], ['MasVnrArea', 'none'], ['MasVnrType', 'label_encoding'], ['MoSold_categorical', 'target_mean'], ['TotalBsmtSF', 'none'], ['BsmtHalfBath', 'none']]

t.prepare()

variants = Variants(t)
variants.apply_variants(t, train_variants)

lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1))
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

for m in models:
    print 'testing', m

    rmse_score = custom_score_using_kfolds(
        m.fit,
        m.predict,
        rmse,
        np.array(t.df_train.values),
        np.array(t.labels.values),
        scale=True,
        doShuffle=True)

    print 'KFold RMSE:', rmse_score

    if False:
        cod_score = custom_score_using_kfolds(
            m.fit,
            m.predict,
            cod,
            np.array(t.df_train.values),
            np.array(t.labels.values),
            scale=True,
            doShuffle=True)

        print 'COD:', cod_score

sys.exit(0)
print 'RMSE', test_accuracy_rmsle(model, t.df_train, t.labels)

df_predicted = pd.DataFrame(columns=['Id', 'SalePrice'])
df_predicted['Id'] = t.test_ids
df_predicted.set_index('Id')
df_predicted['SalePrice'] = np.exp(model.predict(t.df_test))
df_predicted.to_csv('predicted.csv', sep=',', index=False)

print 'predictions done.'
