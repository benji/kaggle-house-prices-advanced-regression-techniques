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

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from utils.utils import *
from utils.kaggle import *
from utils.Training import *
from utils.Variants import *

t = training()
t.explode_possible_types_columns = True

#train_variants = [['OverallQual', 'target_mean'], ['TotalSF', 'none'], ['Neighborhood', 'one_hot'], ['OverallCond', 'label_encoding'], ['BsmtFinType1', 'one_hot'], ['GarageCars', 'none'], ['MSSubClass', 'one_hot'], ['LotArea', 'none'], ['YearBuilt', 'none'], ['MSZoning', 'one_hot'], ['BsmtFinSF1', 'none'], ['SaleCondition', 'target_mean'], ['2ndFlrSF', 'extract_0'], ['KitchenQual', 'one_hot'], ['BsmtExposure', 'one_hot'], ['Functional', 'target_mean'], ['GrLivArea', 'none'], ['Condition1', 'one_hot'], ['HeatingQC', 'target_mean'], ['Fireplaces', 'none'], ['ScreenPorch', 'extract_0'], ['BsmtFullBath', 'none'], ['BsmtQual', 'one_hot'], ['CentralAir', 'label_encoding'], ['YearRemodAdd', 'none'], ['PoolArea', 'none'], [ 'Heating', 'one_hot'], ['RoofMatl', 'target_mean'], ['Foundation', 'target_mean'], ['GarageCond', 'target_mean'], ['HalfBath', 'extract_0'], ['FullBath', 'extract_0'], ['KitchenAbvGr', 'none'], ['LotConfig', 'one_hot'], ['ExterQual', 'label_encoding'], ['Exterior1st', 'label_encoding'], ['WoodDeckSF', 'extract_0'], ['Exterior2nd', 'label_encoding'], ['GarageQual', 'one_hot'], ['BsmtFinSF2', 'extract_0'], ['LotFrontage', 'extract_0'], ['HouseStyle', 'target_mean'], ['1stFlrSF', 'none'], ['MiscVal', 'extract_0'], ['GarageArea', 'extract_0'], ['ExterCond', 'label_encoding'], ['MasVnrArea', 'none'], ['MasVnrType', 'label_encoding'], ['MoSold_categorical', 'target_mean'], ['TotalBsmtSF', 'none'], ['BsmtHalfBath', 'none']]

train_variants = [['TotalSF', 'linearize_order_2'], ['OverallQual', 'target_mean'], ['Neighborhood', 'one_hot'], ['OverallCond', 'one_hot'], ['BsmtUnfSF', 'none'], ['Age', 'quantile_bin(10)'], ['LotArea', 'none'], ['GarageCars', 'none'], ['MSZoning', 'one_hot'], ['Fireplaces', 'none'], ['Functional', 'label_encoding'], ['SaleCondition', 'target_mean'], ['Condition1', 'one_hot'], ['GrLivArea', 'linearize_order_2'], ['YearBuilt', 'none'], ['KitchenQual', 'one_hot'], ['BsmtExposure', 'one_hot'], ['KitchenAbvGr', 'none'], ['YearsSinceRemodel', 'extract_0_linearize_order_2'], ['CentralAir', 'one_hot'], ['ScreenPorch', 'extract_0'], ['BsmtQual', 'one_hot'], ['BsmtFinType1', 'label_encoding'], ['PoolArea', 'none'], ['HeatingQC', 'target_mean'], ['LowQualFinSF', 'linearize_order_2'], ['OverallCond_numerical', 'none'], ['BsmtFinType2', 'target_mean'], ['WoodDeckSF', 'extract_0_linearize_order_2'], ['LotConfig', 'one_hot'], ['1stFlrSF', 'quantile_bin(10)'], [
    'GarageQual', 'target_mean'], ['GarageFinish', 'one_hot'], ['Exterior1st', 'label_encoding'], ['YearRemodAdd', 'none'], ['RoofMatl', 'target_mean'], ['FullBath_categorical', 'target_mean'], ['BsmtFullBath', 'none'], ['TotalBsmtSF', 'quantile_bin(5)'], ['BsmtFinSF1', 'linearize_order_2'], ['HalfBath', 'linearize_order_2'], ['2ndFlrSF', 'extract_0_linearize_order_2'], ['GarageArea', 'quantile_bin(5)'], ['OpenPorchSF', 'linearize_order_2'], ['Remodeled', 'extract_0_linearize_order_2'], ['Foundation', 'label_encoding'], ['ExterCond', 'target_mean'], ['BsmtFinSF2', 'linearize_order_2'], ['BsmtHalfBath', 'none'], ['Street', 'one_hot'], ['HouseStyle', 'label_encoding'], ['Exterior2nd', 'label_encoding'], ['SaleType', 'label_encoding'], ['FullBath', 'linearize_order_2'], ['GarageCars_categorical', 'target_mean'], ['RecentRemodel', 'extract_0'], ['Fireplaces_categorical', 'label_encoding'], ['RoofStyle', 'label_encoding'], ['GarageYrBlt', 'extract_0_linearize_order_2']]

t.prepare()

variants = Variants(t)
variants.apply_variants(t, train_variants)

model0 = Lasso(alpha=0.0005, random_state=1)
model = make_pipeline(RobustScaler(), model0)


def train(x, y):
    model.fit(x, y)


def predict(x):
    return model.predict(x)


rmse_score = custom_score_using_kfolds(
    train,
    predict,
    rmse,
    np.array(t.df_train.values),
    np.array(t.labels.values),
    n_splits=10,
    scale=True,
    doShuffle=False)

print 'KFold RMSE:', rmse_score

cod_score = custom_score_using_kfolds(
    train,
    predict,
    cod,
    np.array(t.df_train.values),
    np.array(t.labels.values),
    scale=True,
    doShuffle=True)

print 'COD:', cod_score

#print 'weigths',model0.coef_
#print 'bias', model0.intercept_

sys.exit(0)
print 'RMSE', test_accuracy_rmsle(model, t.df_train, t.labels)

df_predicted = pd.DataFrame(columns=['Id', 'SalePrice'])
df_predicted['Id'] = t.test_ids
df_predicted.set_index('Id')
df_predicted['SalePrice'] = np.exp(model.predict(t.df_test))
df_predicted.to_csv('predicted.csv', sep=',', index=False)

print 'predictions done.'
