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

for m in models:
    print 'testing', m

    rmse_score = custom_score_using_kfolds(
        m.fit,
        m.predict,
        rmse,
        np.array(t.df_train.values),
        np.array(t.labels.values),
        doShuffle=True)

    print 'KFold RMSE:', rmse_score

sys.exit(0)
print 'RMSE', test_accuracy_rmsle(model, t.df_train, t.labels)

df_predicted = pd.DataFrame(columns=['Id', 'SalePrice'])
df_predicted['Id'] = t.test_ids
df_predicted.set_index('Id')
df_predicted['SalePrice'] = np.exp(model.predict(t.df_test))
df_predicted.to_csv('predicted.csv', sep=',', index=False)

print 'predictions done.'
