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


def training():

    df_train = pd.read_csv('../data/train.csv')
    df_test = pd.read_csv('../data/test.csv')
    schema = json.loads(open('../schema2.json', 'r').read())

    # NA round 1

    for df in [df_train, df_test]:
        df['BsmtUnfSF'][df['BsmtQual'].isnull()] = 0
        df['BsmtFinSF1'][df['BsmtQual'].isnull()] = 0
        df['BsmtFinSF2'][df['BsmtQual'].isnull()] = 0
        df['BsmtHalfBath'][df['BsmtQual'].isnull()] = 0
        df['BsmtFullBath'][df['BsmtQual'].isnull()] = 0
        df['TotalBsmtSF'][df['BsmtQual'].isnull()] = 0

        df["LotFrontage"].fillna(0, inplace=True)

    #.fillna(np.mean(df_train["LotFrontage"].values),inplace=True)

    # ADD CUSTOM FEATURES

    #df_train['DateSold'] = df_train['YrSold'] + df_train['MoSold'] / 12.0
    #df_test['DateSold'] = df_test['YrSold'] + df_test['MoSold'] / 12.0
    #schema['columns']['DateSold'] = {'type': 'NUMERIC'}

    df_train[
        'TotalSF'] = df_train['TotalBsmtSF'] + df_train['1stFlrSF'] + df_train['2ndFlrSF']
    df_test[
        'TotalSF'] = df_test['TotalBsmtSF'] + df_test['1stFlrSF'] + df_test['2ndFlrSF']
    schema['columns']['TotalSF'] = {'type': 'NUMERIC'}

    t = Training(df_train, df_test, schema=schema)
    #t.separate_out_value('PoolArea', 0, 'NoPool')

    # HANDLE TYPOS IN THE DATA

    t.replace_values.append(['Neighborhood', 'NAmes', 'NWAmes'])
    t.replace_values.append(['BldgType', '2fmCon', '2FmCon'])
    t.replace_values.append(['BldgType', 'Duplex', 'Duplx'])
    t.replace_values.append(['BldgType', 'Twnhs', 'TwnhsE'])
    t.replace_values.append(['Exterior2nd', 'Brk Cmn', 'BrkComm'])
    t.replace_values.append(['Exterior2nd', 'CmentBd', 'CemntBd'])
    t.replace_values.append(['Exterior2nd', 'Wd Shng', 'WdShing'])

    # NA round 2

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

    t.drop_columns = ['Utilities']

    # NORMALIZE SOME FEATURES

    t.logify_columns.append('SalePrice')
    t.logify_columns.extend(('LotArea', 'GrLivArea', '1stFlrSF','LotFrontage','TotalBsmtSF','TotalSF'))

    # REMOVE A FEW OULIERS

    t.remove_outliers = [524, 1299]  # +3%!
    #t.remove_outliers.extend((249, 313, 335, 706)) # >100000 LotArea
    #t.remove_outliers.extend((934, 1298,346))

    # hack to convert 20 to '020'
    mssc_cats = schema['columns']['MSSubClass']['categories']
    for c in mssc_cats:
        t.replace_values.append(['MSSubClass', int(c), c])

    if True:
        for c in t.numerical_columns():
            t.numerical_singularities_columns.append(c)
    
    t.enable_transform_preferred_to_numerical = True

    return t
