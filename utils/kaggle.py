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

schema = json.loads(open('../schema2.json', 'r').read())
df_train = pd.read_csv('../data/train.csv')  #, dtype=get_pandas_types(schema))
df_test = pd.read_csv('../data/test.csv')  #, dtype=get_pandas_types(schema))

# special prep
df_train['DateSold'] = df_train['YrSold'] + df_train['MoSold'] / 12.0
df_test['DateSold'] = df_test['YrSold'] + df_test['MoSold'] / 12.0
schema['columns']['DateSold'] = {'type': 'NUMERIC'}

df_train[
    'TotalSF'] = df_train['TotalBsmtSF'] + df_train['1stFlrSF'] + df_train['2ndFlrSF']
df_test[
    'TotalSF'] = df_test['TotalBsmtSF'] + df_test['1stFlrSF'] + df_test['2ndFlrSF']
schema['columns']['TotalSF'] = {'type': 'NUMERIC'}


# -4%!
#outliers_LotArea = df_train['LotArea'][df_train['LotArea'] > 100000]
#print outliers_LotArea
#df_train = df_train.drop(outliers_LotArea.index)

t = Training(df_train, df_test, schema=schema)

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

t.replace_values.append(['Neighborhood', 'NAmes', 'NWAmes'])
t.replace_values.append(['BldgType', '2fmCon', '2FmCon'])
t.replace_values.append(['BldgType', 'Duplex', 'Duplx'])
t.replace_values.append(['BldgType', 'Twnhs', 'TwnhsE'])
t.replace_values.append(['Exterior2nd', 'Brk Cmn', 'BrkComm'])
t.replace_values.append(['Exterior2nd', 'CmentBd', 'CemntBd'])
t.replace_values.append(['Exterior2nd', 'Wd Shng', 'WdShing'])

t.separate_out_value('PoolArea', 0, 'NoPool')
t.logify_columns.append('SalePrice')  # +4%!
t.logify_columns.extend(('LotArea', 'GrLivArea', '1stFlrSF'))  # +0.4%!
t.fill_na_mean = False
t.remove_outliers.extend((524, 1299))  # +3%!
t.dummify_at_init = True
t.dummify_drop_first = False
t.use_label_encoding = False
#t.use_dummies_for_specific_columns = ['Neighborhood']


use_runtime_dummies = False

mssc_cats = schema['columns']['MSSubClass']['categories']
for c in mssc_cats:
    t.replace_values.append(['MSSubClass', int(c), c])

def training():
    return t
