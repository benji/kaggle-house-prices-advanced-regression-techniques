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

    t = Training(df_train, df_test, schema=schema)

    # REMOVE A FEW OULIERS

    for i in [88,462,523,588,632,968,1298,1324]:
        t.drop_row_by_index(i)
    #t.drop_row_by_id(524)
    #t.drop_row_by_id(1299)

    # HANDLE TYPOS IN THE DATA

    t.replace_values('Neighborhood', 'NAmes', 'NWAmes')
    t.replace_values('BldgType', '2fmCon', '2FmCon')
    t.replace_values('BldgType', 'Duplex', 'Duplx')
    t.replace_values('BldgType', 'Twnhs', 'TwnhsE')
    t.replace_values('Exterior2nd', 'Brk Cmn', 'BrkComm')
    t.replace_values('Exterior2nd', 'CmentBd', 'CemntBd')
    t.replace_values('Exterior2nd', 'Wd Shng', 'WdShing')
    t.replace_values('BsmtExposure', 'No', 'None')

    # NA
    t.fill_na_column('PoolQC', 'None')
    t.fill_na_column('Alley', 'None')
    t.fill_na_column('Fence', 'None')
    t.fill_na_column('FireplaceQu', 'None')
    t.fill_na_column('GarageType', 'None')
    t.fill_na_column('GarageFinish', 'None')
    t.fill_na_column('GarageQual', 'None')
    t.fill_na_column('GarageCond', 'None')
    t.fill_na_column('GarageYrBlt', 0)
    t.fill_na_column('GarageArea', 0)
    t.fill_na_column('GarageCars', 0)
    t.fill_na_column('BsmtQual', 'None')
    t.fill_na_column('BsmtCond', 'None')
    t.fill_na_column('BsmtExposure', 'None')
    t.fill_na_column('BsmtFinType1', 'None')
    t.fill_na_column('BsmtFinType2', 'None')
    t.fill_na_column('BsmtFullBath', 0)
    t.fill_na_column('BsmtHalfBath', 0)
    t.fill_na_column('TotalBsmtSF', 0)
    t.fill_na_column('BsmtFinSF1', 0)
    t.fill_na_column('BsmtFinSF2', 0)
    t.fill_na_column('BsmtUnfSF', 0)
    t.fill_na_column('MasVnrType', 'None')
    t.fill_na_column('MasVnrArea', 0)
    t.fill_na_column('MSZoning', 'RL')
    t.fill_na_column('Functional', 'Typ')
    t.fill_na_column('Electrical', 'SBrkr')
    t.fill_na_column('KitchenQual', 'TA')
    t.fill_na_column('Exterior1st', 'VinylSd')
    t.fill_na_column('Exterior2nd', 'VinylSd')
    t.fill_na_column('SaleType', 'WD')
    t.fill_na_column('MSSubClass', 'None')
    t.fill_na_column('MiscFeature', 'None')

    def transform_df(df):
        # NA
        df['BsmtUnfSF'][df['BsmtQual'].isnull()] = 0
        df['BsmtFinSF1'][df['BsmtQual'].isnull()] = 0
        df['BsmtFinSF2'][df['BsmtQual'].isnull()] = 0
        df['BsmtHalfBath'][df['BsmtQual'].isnull()] = 0
        df['BsmtFullBath'][df['BsmtQual'].isnull()] = 0
        df['TotalBsmtSF'][df['BsmtQual'].isnull()] = 0

        df["LotFrontage"].fillna(0, inplace=True)

        # ADD CUSTOM FEATURES

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

        df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']

    t.df_transform(transform_df)

    schema['columns']['TotalSF'] = {'type': 'NUMERIC'}

    #t.separate_out_value('PoolArea', 0, 'NoPool')

    t.drop_column('Utilities')

    # NORMALIZE SOME FEATURES
    t.normalize_column_log1p('SalePrice')
    t.normalize_column_log1p('LotArea')
    t.normalize_column_log1p('GrLivArea')
    t.normalize_column_log1p('1stFlrSF')
    t.normalize_column_log1p('TotalBsmtSF')
    t.normalize_column_log1p('TotalSF')

    if False:
        for c in t.numerical_columns():
            t.regularize_linear_numerical_singularity(c, 0)

    return t


if __name__ == "__main__":
    t = training()

    t.dummify_at_init = False
    t.dummify_drop_first = False
    t.use_label_encoding = False

    t.prepare()

    t.save('../tmp')
