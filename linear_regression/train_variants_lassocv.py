import math
import json
import sys
import os
import time
import pandas as pd
from os import path
import yaml

from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, LassoCV, BayesianRidge, LassoLarsIC, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold, cross_val_score, train_test_split


sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from utils.utils import *
from utils.kaggle import *
from utils.Training import *
from utils.Variants import *

np.random.seed(int(time.time()))

t = training()
t.explode_columns_possibilities()

train_variants = [['TotalSF', 'linearize_order_2'], ['OverallQual', 'target_mean'], ['Neighborhood', 'one_hot'], ['OverallCond_numerical', 'none'], ['BsmtUnfSF', 'quantile_bin(10)'], ['Age', 'quantile_bin(20)'], ['LotArea', 'extract_0_linearize_order_2'], ['GarageCars_categorical', 'label_encoding'], ['MSZoning', 'one_hot'], ['Fireplaces', 'linearize_order_3'], ['Condition1', 'one_hot'], ['Functional', 'label_encoding'], ['SaleCondition', 'target_median'], [
    'CentralAir', 'target_mean'], ['KitchenQual', 'one_hot'], ['GrLivArea', 'extract_0_linearize_order_2'], ['BsmtExposure', 'one_hot'], ['YearBuilt', 'none'], ['KitchenAbvGr_categorical', 'label_encoding'], ['YearsSinceRemodel', 'extract_0_linearize_order_2'], ['HeatingQC', 'target_median'], ['ScreenPorch', 'linearize_order_2'], ['RoofMatl', 'label_encoding'], ['GarageArea', 'linearize_order_3'], ['GarageQual', 'target_median']]

train_cols = [v[0] for v in train_variants]

t.retain_columns(train_cols)
t.health_check()

variants = Variants(t, verbose=True)
variants.apply_variants(t, train_variants)

t.ready_for_takeoff()
t.scale()
t.shuffle()
t.remove_columns_with_unique_value()
t.sanity()
t.summary()
t.save('tmp')

assert t.diagnose_nas() == 0
X, X_test = t.df_train.values, t.df_test.values
#X,X_test = t.pca(n_components=100)
y = t.labels.values

model_lassocv = LassoCV()  # random_state=1)

lassocv_alphas = []


def train_lassocv(x, y):
    model_lassocv.fit(x, y)
    lassocv_alphas.append(model_lassocv.alpha_)
    print 'alpha=', model_lassocv.alpha_


def predict_lassocv(x):
    return model_lassocv.predict(x)

score_kfolds = 8

rmse_score_lassocv = custom_score_using_kfolds(
    train_lassocv,
    predict_lassocv,
    rmse,
    np.array(X),
    np.array(y),
    n_splits=score_kfolds,
    doShuffle=False)

print 'LassoCV KFold RMSE:', rmse_score_lassocv

alpha_mean = np.array(lassocv_alphas).mean()
print 'LassoCV mean alpha=', alpha_mean

model = Lasso(alpha=alpha_mean)  # , random_state=1)


def train(x, y):
    model.fit(x, y)


def predict(x):
    return model.predict(x)


rmse_score = custom_score_using_kfolds(
    train,
    predict,
    rmse,
    np.array(X),
    np.array(y),
    n_splits=score_kfolds,
    doShuffle=False)

print 'Lasso KFold RMSE:', rmse_score


#print 'weigths',model0.coef_
#print 'bias', model0.intercept_

df_predicted = pd.DataFrame(columns=['Id', 'SalePrice'])
df_predicted['Id'] = t.test_ids
df_predicted.set_index('Id')
df_predicted['SalePrice'] = np.expm1(model.predict(X_test))
df_predicted.to_csv('predicted.csv', sep=',', index=False)

print 'predictions done.'
