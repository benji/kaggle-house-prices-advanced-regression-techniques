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
from sklearn.base import clone

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from utils.utils import *
from utils.kaggle import *
from utils.Training import *
from utils.Variants import *

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')
schema = yaml.safe_load(open('../schema.yaml', 'r').read())

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

if False:
    tmp = train[train['Neighborhood'] == 'Sawyer']

    tmp['TotalSF'] = tmp['TotalBsmtSF'] + tmp['1stFlrSF'] + tmp['2ndFlrSF']
    plt.scatter(tmp['TotalSF'], tmp['SalePrice'])
    plt.show()
    sys.exit(0)

for df in [train, test]:
#    print df[['Id','GrLivArea','LotArea','Condition1','Condition2','MSSubClass','MSZoning','SaleType','SaleCondition']][df['GrLivArea'] > 4000]
#    print df[['Id','GrLivArea','LotArea']][df['Condition1'] == 'PosN'][df['LotArea'] > 20000]
#    print df[['Id','GrLivArea','LotArea']][df['Condition2'] == 'PosN'][df['LotArea'] > 20000]

    #tmp = df[df['TotalBsmtSF'] > 0]
    #tmp = tmp[tmp['BsmtUnfSF'] == tmp['TotalBsmtSF']]
    #print tmp[['BsmtUnfSF','TotalBsmtSF','SaleType','SaleCondition']]

    for v in df['RoofMatl'].unique():
        print v,len(df[df['RoofMatl']==v])

    print '--'

    # print has basement + BsmtUnfSF == TotalBsmtSF
