import math
import json
import sys
import os
import yaml
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
    schema = yaml.safe_load(open('../schema.yaml', 'r').read())

    # NA round 1

    for df in [df_train, df_test]:
        df['BsmtUnfSF'][df['BsmtQual'].isnull()] = 0
        df['BsmtFinSF1'][df['BsmtQual'].isnull()] = 0
        df['BsmtFinSF2'][df['BsmtQual'].isnull()] = 0
        df['BsmtHalfBath'][df['BsmtQual'].isnull()] = 0
        df['BsmtFullBath'][df['BsmtQual'].isnull()] = 0
        df['TotalBsmtSF'][df['BsmtQual'].isnull()] = 0

        df["LotFrontage"].fillna(0, inplace=True)

    # ADD CUSTOM FEATURES

    for df in [df_train, df_test]:
        df["Age"] = 2010 - df["YearBuilt"]
        schema['columns']['Age'] = {'type': 'NUMERIC'}

        df["YearsSinceRemodel"] = df["YrSold"] - df["YearRemodAdd"]
        schema['columns']['YearsSinceRemodel'] = {'type': 'NUMERIC'}

        df["TimeSinceSold"] = 2010 - df["YrSold"]
        schema['columns']['TimeSinceSold'] = {'type': 'NUMERIC'}

        df["Remodeled"] = (df["YearRemodAdd"] != df["YearBuilt"]) * 1
        schema['columns']['Remodeled'] = {'type': 'NUMERIC'}

        df["RecentRemodel"] = (df["YearRemodAdd"] == df["YrSold"]) * 1
        schema['columns']['RecentRemodel'] = {'type': 'NUMERIC'}

        df["VeryNewHouse"] = (df["YearBuilt"] == df["YrSold"]) * 1
        schema['columns']['VeryNewHouse'] = {'type': 'NUMERIC'}

        df['DateSold'] = df['YrSold'] + df['MoSold'] / 12.0
        schema['columns']['DateSold'] = {'type': 'NUMERIC'}

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
    t.replace_values.append(['BsmtExposure', 'No', 'None'])

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
    t.logify_columns.extend(
        ('LotArea', 'GrLivArea', '1stFlrSF', 'TotalBsmtSF', 'TotalSF'))  # ,'LotFrontage'

    # REMOVE A FEW OULIERS
    t.remove_outliers = [524, 1299]  #,935 +3%!
    # t.remove_outliers.extend((249, 313, 335, 706)) # >100000 LotArea
    #t.remove_outliers.extend((934, 1298,346))

    if True:
        for c in t.numerical_columns():
            t.numerical_singularities_columns.append([c, 0])

    #t.enable_transform_preferred_to_numerical = True

    return t


if __name__ == "__main__":
    t = training()

    t.dummify_at_init = False
    t.dummify_drop_first = False
    t.use_label_encoding = False

    t.prepare()

    t.save('../tmp')
