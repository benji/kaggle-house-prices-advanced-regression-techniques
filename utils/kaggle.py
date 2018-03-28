import math
import json
import sys
import os
import time
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

np.random.seed(int(time.time()))


def seed():
    return np.random.randint(2**32-1)


def training(exclude_outliers=True,remove_partials=False):

    df_train = pd.read_csv('../data/train.csv')
    df_test = pd.read_csv('../data/test.csv')
    schema = yaml.safe_load(open('../schema.yaml', 'r').read())

    if remove_partials:
        print '!!!!!!!!!!!!!!!!!!!!!!!!!!!!! EXCLUDE PARTIALS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
        df_train = df_train[df_train['SaleCondition'] != 'Partial']


    t = Training(df_train, df_test, schema=schema)
    #t.strict_check = False

    return kaggle_default_configure(t, exclude_outliers=exclude_outliers,remove_partials=remove_partials)


def kaggle_default_configure(t, exclude_outliers=True,remove_partials=False):

    # REMOVE A FEW OULIERS

    #if exclude_outliers:
    #    print '!!!!!!!!!!!!!!!!!!!!!!!!!!!!! EXCLUDE OUTLIERS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
        #t.drop_row_by_index(632)
        #for i in [88, 462, 523, 588, 968, 1298, 1324]:
        #    t.drop_row_by_index(i)
        #for i in [632,523,462,1324,968,970]:
        #    t.drop_row_by_index(i)

            

    # for i in [30, 964, 409, 493, 683, 1445, 1424, 869, 706]:
    #    t.drop_row_by_index(i)

    # t.drop_row_by_id(524)
    # t.drop_row_by_id(1299)

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

    t.convert_categorical_column_to_numerical('OverallQual')
    t.convert_categorical_column_to_numerical('OverallCond')
    t.label_encode_column('ExterQual')
    t.label_encode_column('ExterCond')
    t.label_encode_column('GarageQual')
    t.label_encode_column('GarageCond')
    t.label_encode_column('BsmtQual')
    t.label_encode_column('BsmtCond')
    # t.label_encode_column('KitchenQual')

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
        df['IsMSSubClassPUD'] = (df['MSSubClass'] == '120')*1 + \
            (df['MSSubClass'] == '150')*1 + (df['MSSubClass'] == '160')*1 + \
            (df['MSSubClass'] == '180')*1

        # 120	1-STORY PUD (Planned Unit Development) - 1946 & NEWER
        # 150	1-1/2 STORY PUD - ALL AGES
        # 160	2-STORY PUD - 1946 & NEWER
        # 180	PUD - MULTILEVEL - INCL SPLIT LEV/FOYER

        for cq in ['Overall', 'Bsmt', 'Exter', 'Garage']:
            df[cq+'CondXQual'] = df[cq+'Cond'] * df[cq+'Qual']
            df[cq+'Cond+Qual'] = df[cq+'Cond'] + df[cq+'Qual']

        df['DateSold'] = df['YrSold'] + df['MoSold'] / 12.0

        df['BsmtFinishedSF'] = df['TotalBsmtSF']-df['BsmtUnfSF']
        df['BsmtUnfSFPct'] = df['BsmtUnfSF']/(1+df['TotalBsmtSF'])

        df['IsBsmtType1Living'] = (df['BsmtFinType1'] == 'GLQ')*1 + \
            (df['BsmtFinType1'] == 'ALQ')*1 + (df['BsmtFinType1'] == 'BLQ')*1 + \
            (df['BsmtFinType1'] == 'Rec')*1
        df['IsBsmtType2Living'] = (df['BsmtFinType2'] == 'GLQ')*1 + \
            (df['BsmtFinType2'] == 'ALQ')*1 + (df['BsmtFinType2'] == 'BLQ')*1 + \
            (df['BsmtFinType2'] == 'Rec')*1

        df['TotalRooms'] = df['TotRmsAbvGrd'] + \
            df['IsBsmtType1Living']+df['IsBsmtType2Living']

        df['BsmtLivingArea'] = df['BsmtFinSF1']*df['IsBsmtType1Living'] + \
            df['BsmtFinSF2']*df['IsBsmtType2Living']
        df['TotalLivingArea'] = df['GrLivArea'] + df['BsmtLivingArea']

        for cond in t.schema['columns']['Condition1']['categories']:
            if cond != 'Norm':
                cname = 'HasCondition_'+cond
                df[cname] = (df['Condition1'] == cond)*1 + \
                    (df['Condition2'] == cond)*1
                t.schema['columns'][cname] = {'type': 'NUMERIC'}

                cname = 'HasCondition2_'+cond
                df[cname] = (df['HasCondition_'+cond] > 0)*1
                t.schema['columns'][cname] = {'type': 'NUMERIC'}

        # Timeline: Built / remodeled / sold

        # differences
        df["AgeWhenSold"] = df["YrSold"] - df["YearBuilt"]
        df["AgeWhenLastRemod"] = df["YearRemodAdd"] - df["YearBuilt"]
        df["YearsSinceLastRemodel"] = df["YrSold"] - df["YearRemodAdd"]
        df['YearsSinceLastRemodel'][df["YearsSinceLastRemodel"] < 0] = 0
        df["OneFloorOnly"] = (df["2ndFlrSF"] == 0)*1
        df["SummerSale"] = (df["MoSold"] == 5)*1+(df["MoSold"] == 6)*1+(df["MoSold"] == 7)*1

        # binaries
        df["BuiltSoldSameYear"] = (df["YearBuilt"] == df["YrSold"]) * 1
        df["Remodeled"] = (df["YearRemodAdd"] != df["YearBuilt"]) * 1
        df["RemodeledSoldSameYear"] = (df["YearRemodAdd"] == df["YrSold"]) * 1

        df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
        df['ExtraGarageAreaApprox'] = (df["GarageType"] == 'Basment') * \
            df["GarageArea"] + (df["GarageType"] ==
                                '2Types') * df["GarageArea"]
        df['TotalSF_with_garage'] = df['TotalSF'] + df['ExtraGarageAreaApprox']
        df['TotalPorch'] = df['WoodDeckSF'] + df['OpenPorchSF'] + \
            df['3SsnPorch'] + df['ScreenPorch'] + df['EnclosedPorch']
        df['TotalSF_with_garage_porch'] = df['TotalSF_with_garage']+df['TotalPorch']

        df['RoofMatl=CompShg'] = (df['RoofMatl'] =='CompShg') * 1

    t.df_transform(transform_df)
    t.drop_column('IsBsmtType1Living')
    t.drop_column('IsBsmtType2Living')
    t.drop_column('Condition1')
    t.drop_column('Condition2')
    t.drop_column('ExtraGarageAreaApprox')
    t.drop_column('RoofMatl')

    for c in ['AgeWhenLastRemod', 'YearsSinceLastRemodel', 'Remodeled', 'DateSold', 'AgeWhenSold', 'BuiltSoldSameYear', 'RemodeledSoldSameYear',
              'TotalSF', 'TotalSF_with_garage', 'TotalPorch', 'TotalSF_with_garage_porch', 'BsmtFinishedSF',
              'TotalRooms', 'BsmtLivingArea', 'TotalLivingArea', 'BsmtUnfSFPct',
              'IsMSSubClassPUD','OneFloorOnly','SummerSale','RoofMatl=CompShg']:
        t.schema['columns'][c] = {'type': 'NUMERIC'}

    for cq in ['Overall', 'Bsmt', 'Exter', 'Garage']:
        t.schema['columns'][cq+'Cond+Qual'] = {'type': 'NUMERIC'}
        t.schema['columns'][cq+'CondXQual'] = {'type': 'NUMERIC'}

    #t.separate_out_value('PoolArea', 0, 'NoPool')

    t.drop_column('Utilities')

    # NORMALIZE SOME FEATURES
    t.normalize_column_log1p('SalePrice')
    if False:
        t.normalize_column_log1p('LotArea')
        t.normalize_column_log1p('GrLivArea')
        t.normalize_column_log1p('1stFlrSF')
        t.normalize_column_log1p('TotalBsmtSF')
        t.normalize_column_log1p('TotalSF')
        t.normalize_column_log1p('TotalSF_with_garage')
        t.normalize_column_log1p('TotalSF_with_garage_porch')

        t.normalize_column_log1p('BsmtUnfSF')
        t.normalize_column_log1p('BsmtFinishedSF')
        t.normalize_column_log1p('BsmtLivingArea')
        t.normalize_column_log1p('TotalLivingArea')

    # for c in ['SalePrice', 'LotArea', 'GrLivArea', '1stFlrSF', 'TotalBsmtSF',
    #          'TotalSF', 'TotalSF_with_garage', 'TotalPorch', 'TotalSF_with_garage_porch']:
    #    t.normalize_column_log1p(c)

    #t.duplicate_column('TotalSF','TotalSF_l3')
    #t.linearize_column_with_polynomial_x_transform('TotalSF_l3',order=3)
    #t.duplicate_column('OverallQual','OverallQual_l2')
    #t.linearize_column_with_polynomial_x_transform('OverallQual_l2',order=2)

    if False:
        for c in t.numerical_columns():
            t.regularize_linear_numerical_singularity(c, 0)

    return t


def training_train_test_holdout(holdout=300, exclude_outliers=True):

    for i in range(50):
        print 'WWWWWWWWWAAAAAAAAAAAAWWWWWWWWWAAAAAAAAAAAAWWWWWWWWWAAAAAAAAAAAAWWWWWWWWWAAAAAAAAAAAAWWWWWWWWWAAAAAAAAAAAAWWWWWWWWWAAAAAAAAAAAAWWWWWWWWWAAAAAAAAAAAAWWWWWWWWWAAAAAAAAAAAA'

    df_train = shuffle(pd.read_csv('../data/train.csv'), random_state=seed())

    df_test = df_train[-holdout:].copy()
    print df_test.head()

    test_labels = df_test['SalePrice'].copy()
    df_test.drop(['SalePrice'], axis=1, inplace=True)

    df_train = df_train[:-holdout].copy()

    #df_test = pd.read_csv('../data/test.csv')
    schema = yaml.safe_load(open('../schema.yaml', 'r').read())

    t = Training(df_train, df_test, schema=schema)
    t.test_labels = test_labels
    t.y2_labels = t.labels.values[-holdout:].copy()

    return kaggle_default_configure(t)


if __name__ == "__main__":
    t = training()

    t.dummify_at_init = False
    t.dummify_drop_first = False
    t.use_label_encoding = False

    t.prepare()

    t.save('../tmp')
