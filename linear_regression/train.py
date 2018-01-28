import math, json, sys, os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from utils.utils import *
from utils.Training import *

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
t.train_columns = [
    'OverallQual', 'GrLivArea', 'Neighborhood', 'BsmtFinSF1', 'CentralAir',
    'LotArea', 'YearBuilt', 'OverallCond', 'TotalBsmtSF', '1stFlrSF',
    'GarageCars', '2ndFlrSF', 'Condition1', 'SaleCondition', 'MiscFeature',
    'Exterior1st', 'Fireplaces', 'KitchenQual'
]
t.prepare()

accuracy = test_accuracy(t.df_train, t.labels, passes=1000) * 100
print 'Accuracy', accuracy

generate_predictions(t.labels, t.df_train, t.df_test, t.test_ids)
print "Predictions complete."