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
train_variants = [['OverallQual_numerical', 'extract_0_linearize_order_2'], ['TotalSF', 'linearize_order_3'], ['MSZoning', 'target_median'], ['YearRemodAdd', 'linearize_order_3'], ['OverallCond', 'one_hot'], ['GarageCars_categorical', 'label_encoding'], ['BsmtUnfSF', 'quantile_bin(5)'], ['Fireplaces_categorical', 'target_median'], ['Neighborhood', 'target_median'], ['SaleType', 'target_median'], ['Heating', 'target_mean'], ['GarageArea', 'quantile_bin(5)'], ['BsmtFullBath', 'none'], ['FullBath_categorical', 'one_hot'], ['Electrical', 'label_encoding'], ['YearBuilt', 'extract_0_linearize_order_2'], ['GarageFinish', 'target_median'], ['LotArea', 'quantile_bin(5)'], ['BsmtFinType1', 'label_encoding'], ['FullBath', 'extract_0_linearize_order_2'], ['LowQualFinSF', 'extract_0_linearize_order_2'], ['YrSold', 'none'], ['ScreenPorch', 'linearize_order_2'], ['BsmtFinType2', 'target_median'], ['TotRmsAbvGrd_categorical', 'target_mean'], ['HalfBath', 'extract_0_linearize_order_2'], ['SaleCondition', 'target_mean'], ['BedroomAbvGr', 'linearize_order_2'], ['HeatingQC', 'target_mean'], ['GarageYrBlt', 'quantile_bin(5)'], ['CentralAir', 'target_median'], ['RoofStyle', 'target_median'], ['LandContour', 'target_median'], ['BsmtQual', 'one_hot'], ['BsmtCond', 'one_hot'], ['MiscFeature', 'label_encoding'], ['TotRmsAbvGrd', 'linearize_order_3'], ['BsmtFullBath_categorical', 'one_hot'], ['GarageCars', 'linearize_order_2'], ['3SsnPorch', 'extract_0_linearize_order_2'], ['Condition1', 'target_mean'], ['KitchenAbvGr_categorical', 'target_mean'], ['RecentRemodel', 'none'], ['WoodDeckSF', 'none'], ['PavedDrive', 'target_median'], ['2ndFlrSF', 'extract_0_linearize_order_2'], ['DateSold', 'quantile_bin(5)'], ['KitchenAbvGr', 'linearize_order_2'], ['HouseStyle', 'label_encoding'], ['Street', 'target_median'], ['EnclosedPorch', 'none'], ['Age', 'extract_0'], ['GarageCond', 'target_median'], ['BedroomAbvGr_categorical', 'target_median'], ['TotalBsmtSF', 'quantile_bin(5)'], ['TimeSinceSold', 'linearize_order_2'], ['GrLivArea', 'linearize_order_2'], ['MoSold_categorical', 'label_encoding'], ['BsmtExposure', 'target_mean'], ['Remodeled', 'extract_0'], ['MasVnrArea', 'linearize_order_2'], ['OpenPorchSF', 'linearize_order_3']]
#train_variants = [['OverallQual_numerical', 'extract_0_linearize_order_2'], ['TotalSF', 'linearize_order_3'], ['MSZoning', 'target_median'], ['YearRemodAdd', 'linearize_order_3'], ['OverallCond', 'one_hot'], ['GarageCars_categorical', 'label_encoding'], ['BsmtUnfSF', 'quantile_bin(5)'], ['Fireplaces_categorical', 'target_median'], ['Neighborhood', 'target_median'], ['SaleType', 'target_median'], ['Heating', 'target_mean'], ['GarageArea', 'quantile_bin(5)'], ['BsmtFullBath', 'none'], ['FullBath_categorical', 'one_hot'], ['Electrical', 'label_encoding'], ['YearBuilt', 'extract_0_linearize_order_2'], ['GarageFinish', 'target_median'], ['LotArea', 'quantile_bin(5)'], ['BsmtFinType1', 'label_encoding'], ['FullBath', 'extract_0_linearize_order_2'], ['LowQualFinSF', 'extract_0_linearize_order_2'], ['YrSold', 'none'], ['ScreenPorch', 'linearize_order_2'], ['BsmtFinType2', 'target_median'], ['TotRmsAbvGrd_categorical', 'target_mean'], ['HalfBath', 'extract_0_linearize_order_2'], ['SaleCondition', 'target_mean'], ['BedroomAbvGr', 'linearize_order_2'], ['HeatingQC', 'target_mean'], ['GarageYrBlt', 'quantile_bin(5)'], ['CentralAir', 'target_median'], ['RoofStyle', 'target_median'], ['LandContour', 'target_median'], ['BsmtQual', 'one_hot'], ['BsmtCond', 'one_hot'], ['MiscFeature', 'label_encoding'], ['TotRmsAbvGrd', 'linearize_order_3'], ['BsmtFullBath_categorical', 'one_hot'], ['GarageCars', 'linearize_order_2'], ['3SsnPorch', 'extract_0_linearize_order_2'], ['Condition1', 'target_mean'], ['KitchenAbvGr_categorical', 'target_mean'], ['RecentRemodel', 'none'], ['WoodDeckSF', 'none'], ['PavedDrive', 'target_median'], ['2ndFlrSF', 'extract_0_linearize_order_2'], ['DateSold', 'quantile_bin(5)'], ['KitchenAbvGr', 'linearize_order_2'], ['HouseStyle', 'label_encoding']]
#train_variants = [['OverallQual_numerical', 'extract_0_linearize_order_2'], ['TotalSF', 'linearize_order_3'], ['MSZoning', 'target_median'], ['YearRemodAdd', 'linearize_order_3'], ['OverallCond', 'one_hot'], ['GarageCars_categorical', 'label_encoding'], ['BsmtUnfSF', 'quantile_bin(5)'], ['Fireplaces_categorical', 'target_median'], ['Neighborhood', 'target_median'], ['SaleType', 'target_median'], ['Heating', 'target_mean'], ['GarageArea', 'quantile_bin(5)'], ['BsmtFullBath', 'none'], ['FullBath_categorical', 'one_hot'], ['Electrical', 'label_encoding'], ['YearBuilt', 'extract_0_linearize_order_2'], ['GarageFinish', 'target_median'], ['LotArea', 'quantile_bin(5)'], ['BsmtFinType1', 'label_encoding'], ['FullBath', 'extract_0_linearize_order_2'], ['LowQualFinSF', 'extract_0_linearize_order_2'], ['YrSold', 'none'], ['ScreenPorch', 'linearize_order_2'], ['BsmtFinType2', 'target_median'], ['TotRmsAbvGrd_categorical', 'target_mean'], ['HalfBath', 'extract_0_linearize_order_2'], ['SaleCondition', 'target_mean'], ['BedroomAbvGr', 'linearize_order_2'], ['HeatingQC', 'target_mean'], ['GarageYrBlt', 'quantile_bin(5)'], ['CentralAir', 'target_median'], ['RoofStyle', 'target_median'], ['LandContour', 'target_median'], ['BsmtQual', 'one_hot'], ['BsmtCond', 'one_hot'], ['MiscFeature', 'label_encoding'], ['TotRmsAbvGrd', 'linearize_order_3'], ['BsmtFullBath_categorical', 'one_hot'], ['GarageCars', 'linearize_order_2'], ['3SsnPorch', 'extract_0_linearize_order_2'], ['Condition1', 'target_mean'], ['KitchenAbvGr_categorical', 'target_mean'], ['RecentRemodel', 'none'], ['WoodDeckSF', 'none']]
#train_variants = [['TotalSF', 'linearize_order_3'], ['OverallQual', 'target_mean'], ['Neighborhood', 'one_hot'], ['OverallCond', 'one_hot'], ['BsmtUnfSF', 'none'], ['Age', 'quantile_bin(10)'], ['LotArea', 'none'], ['MSZoning', 'one_hot'], ['GarageCars', 'none'], ['Fireplaces', 'linearize_order_2'], ['Condition1', 'one_hot'], ['Functional', 'label_encoding'], ['SaleCondition', 'one_hot'], ['KitchenQual', 'one_hot'], ['GrLivArea', 'linearize_order_3'], ['CentralAir', 'target_mean'], ['BsmtExposure', 'one_hot'], ['KitchenAbvGr_categorical', 'target_mean'], ['YearBuilt', 'none'], ['HeatingQC', 'target_median']]
#train_variants = [['TotalSF', 'linearize_order_2'], ['OverallQual', 'target_mean'], ['Neighborhood', 'one_hot'], ['OverallCond', 'one_hot'], ['BsmtUnfSF', 'none'], ['Age', 'quantile_bin(10)'], ['LotArea', 'none'], ['GarageCars', 'none'], ['MSZoning', 'one_hot'], ['Fireplaces', 'none'], ['Functional', 'label_encoding'], ['SaleCondition', 'target_mean'], ['Condition1', 'one_hot'], ['GrLivArea', 'linearize_order_2'], ['YearBuilt', 'none'], ['KitchenQual', 'one_hot'], ['BsmtExposure', 'one_hot'], ['KitchenAbvGr', 'none'], ['YearsSinceRemodel', 'extract_0_linearize_order_2'], ['CentralAir', 'one_hot'], ['ScreenPorch', 'extract_0'], ['BsmtQual', 'one_hot'], ['BsmtFinType1', 'label_encoding'], ['PoolArea', 'none'], ['HeatingQC', 'target_mean'], ['LowQualFinSF', 'linearize_order_2'], ['OverallCond_numerical', 'none'], ['BsmtFinType2', 'target_mean'], ['WoodDeckSF', 'extract_0_linearize_order_2'], ['LotConfig', 'one_hot'], ['1stFlrSF', 'quantile_bin(10)'], [
#    'GarageQual', 'target_mean'], ['GarageFinish', 'one_hot'], ['Exterior1st', 'label_encoding'], ['YearRemodAdd', 'none'], ['RoofMatl', 'target_mean'], ['FullBath_categorical', 'target_mean'], ['BsmtFullBath', 'none'], ['TotalBsmtSF', 'quantile_bin(5)'], ['BsmtFinSF1', 'linearize_order_2'], ['HalfBath', 'linearize_order_2'], ['2ndFlrSF', 'extract_0_linearize_order_2'], ['GarageArea', 'quantile_bin(5)'], ['OpenPorchSF', 'linearize_order_2'], ['Remodeled', 'extract_0_linearize_order_2'], ['Foundation', 'label_encoding'], ['ExterCond', 'target_mean'], ['BsmtFinSF2', 'linearize_order_2'], ['BsmtHalfBath', 'none'], ['Street', 'one_hot'], ['HouseStyle', 'label_encoding'], ['Exterior2nd', 'label_encoding'], ['SaleType', 'label_encoding'], ['FullBath', 'linearize_order_2'], ['GarageCars_categorical', 'target_mean'], ['RecentRemodel', 'extract_0'], ['Fireplaces_categorical', 'label_encoding'], ['RoofStyle', 'label_encoding'], ['GarageYrBlt', 'extract_0_linearize_order_2']]
#[['TotalSF', 'linearize_order_2'], ['OverallQual', 'target_mean'], ['Neighborhood', 'one_hot'], ['OverallCond', 'one_hot'], ['BsmtUnfSF', 'none'], ['Age', 'quantile_bin(10)'], ['LotArea', 'none'], ['GarageCars', 'none'], ['MSZoning', 'one_hot'], ['Fireplaces', 'none'], ['Functional', 'label_encoding'], ['SaleCondition', 'target_mean'], ['Condition1', 'one_hot'], ['GrLivArea', 'linearize_order_2'], ['YearBuilt', 'none'], ['KitchenQual', 'one_hot'], ['BsmtExposure', 'one_hot'], ['KitchenAbvGr', 'none'], ['YearsSinceRemodel', 'extract_0_linearize_order_2'], ['CentralAir', 'one_hot'], ['ScreenPorch', 'extract_0'], ['BsmtQual', 'one_hot'], ['BsmtFinType1', 'label_encoding'], ['PoolArea', 'none'], ['HeatingQC', 'target_mean'], ['LowQualFinSF', 'linearize_order_2'], ['OverallCond_numerical', 'none'], ['BsmtFinType2', 'target_mean'], ['WoodDeckSF', 'extract_0_linearize_order_2'], ['LotConfig', 'one_hot'], ['1stFlrSF', 'quantile_bin(10)'], [
#    'GarageQual', 'target_mean'], ['GarageFinish', 'one_hot'], ['Exterior1st', 'label_encoding'], ['YearRemodAdd', 'none'], ['RoofMatl', 'target_mean'], ['FullBath_categorical', 'target_mean'], ['BsmtFullBath', 'none'], ['TotalBsmtSF', 'quantile_bin(5)'], ['BsmtFinSF1', 'linearize_order_2'], ['HalfBath', 'linearize_order_2'], ['2ndFlrSF', 'extract_0_linearize_order_2'], ['GarageArea', 'quantile_bin(5)'], ['OpenPorchSF', 'linearize_order_2'], ['Remodeled', 'extract_0_linearize_order_2'], ['Foundation', 'label_encoding'], ['ExterCond', 'target_mean'], ['BsmtFinSF2', 'linearize_order_2'], ['BsmtHalfBath', 'none'], ['Street', 'one_hot'], ['HouseStyle', 'label_encoding'], ['Exterior2nd', 'label_encoding'], ['SaleType', 'label_encoding'], ['FullBath', 'linearize_order_2'], ['GarageCars_categorical', 'target_mean'], ['RecentRemodel', 'extract_0'], ['Fireplaces_categorical', 'label_encoding'], ['RoofStyle', 'label_encoding'], ['GarageYrBlt', 'extract_0_linearize_order_2']]
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


rmse_score_lassocv = custom_score_using_kfolds(
    train_lassocv,
    predict_lassocv,
    rmse,
    np.array(X),
    np.array(y),
    n_splits=10,
    scale=False,
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
    n_splits=10,
    scale=False,
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
