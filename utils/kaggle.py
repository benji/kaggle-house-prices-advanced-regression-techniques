import math, json, sys, os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from os import path

from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
#import xgboost as xgb
#import lightgbm as lgb

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from utils import *
from Training import *

pd.options.mode.chained_assignment = None  # default='warn'

df_train = pd.read_csv('../data/train.csv')
df_test = pd.read_csv('../data/test.csv')
schema = json.loads(open('../schema.json', 'r').read())

# special prep
df_train['DateSold'] = df_train['YrSold'] + df_train['MoSold'] / 12.0
df_test['DateSold'] = df_test['YrSold'] + df_test['MoSold'] / 12.0

# -4%!
#outliers_LotArea = df_train['LotArea'][df_train['LotArea'] > 100000]
#print outliers_LotArea
#df_train = df_train.drop(outliers_LotArea.index)

#t.separate_out_value('PoolArea', 0, 'NoPool')

t = Training(df_train, df_test, schema=schema)
t.logify_columns.append('SalePrice')  # +4%!
t.logify_columns.extend(('LotArea', 'GrLivArea', '1stFlrSF'))  # +0.4%!
t.fill_na_mean = True
t.remove_outliers.extend((524, 1299))  # +3%!
t.dummify_at_init = True
#t.train_columns = [
#    'OverallQual', 'GrLivArea', 'Neighborhood', 'BsmtFinSF1', 'CentralAir',
#    'LotArea', 'YearBuilt', 'OverallCond', 'TotalBsmtSF', '1stFlrSF',
#    'GarageCars', '2ndFlrSF', 'Condition1', 'SaleCondition', 'MiscFeature',
#    'Exterior1st', 'Fireplaces', 'KitchenQual'
#]

t.columns_na = {
    'PoolQC': 'None',
    'Alley': 'None',
    'Fence': 'None',
    'FireplaceQu': 'None',
    'GarageType': 'None',
    'GarageFinish': 'None',
    'GarageQual': 'None',
    'GarageCond': 'None',
    'GarageYrBlt': 0,
    'GarageArea': 0,
    'GarageCars': 0,
    'BsmtQual': 'None',
    'BsmtCond': 'None',
    'BsmtExposure': 'None',
    'BsmtFinType1': 'None',
    'BsmtFinType2': 'None',
    'MasVnrType': 'None',
    'MasVnrArea': 0,
    'MSZoning': 'RL',
    'Functional': 'Typ',
    'Electrical': 'SBrkr',
    'KitchenQual': 'TA',
    'Exterior1st': 'VinylSd',
    'Exterior2nd': 'VinylSd',
    'SaleType': 'WD',
    'MSSubClass': 'None',
    'MiscFeature': 'None'
}

t.replace_values.append(['Neighborhood', 'NAmes','NWAmes'])
t.replace_values.append(['BldgType', '2fmCon','2FmCon'])
t.replace_values.append(['BldgType', 'Duplex','Duplx'])
t.replace_values.append(['BldgType', 'Twnhs','TwnhsE'])

t.drop_columns = ['Utilities']

t.prepare()


def training():
    return t
