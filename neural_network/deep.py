import pandas as pd
import math, json, sys, os
from tqdm import tqdm
import numpy as np
import keras
from os import path

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from utils.utils import *
from utils.Training import *

pd.options.mode.chained_assignment = None  # default='warn'

use_log_label = False
hm_manual_validation = 130

df_train = pd.read_csv('../data/train.csv')
df_test = pd.read_csv('../data/test.csv')

schema = json.loads(open('../schema2.json', 'r').read())

# special prep
#df_train['DateSold'] = df_train['YrSold'] + df_train['MoSold'] / 12.0
#df_test['DateSold'] = df_test['YrSold'] + df_test['MoSold'] / 12.0

#outliers_LotArea = df_train['LotArea'][df_train['LotArea'] > 100000]
#print outliers_LotArea
#df_train = df_train.drop(outliers_LotArea.index)

#t.separate_out_value('PoolArea', 0, 'NoPool')

outliers_LotArea = df_train['LotArea'][df_train['LotArea'] > 100000]
df_train = df_train.drop(outliers_LotArea.index)

#df_train['TotalSF'] = df_train['TotalBsmtSF'] + df_train['1stFlrSF'] + df_train['2ndFlrSF']
#df_test['TotalSF'] = df_test['TotalBsmtSF'] + df_test['1stFlrSF'] + df_test['2ndFlrSF']


t = Training(df_train, df_test, schema=schema)

t.columns_na = {
    'PoolQC':'None',
    'Alley':'None',
    'Fence':'None',
    'FireplaceQu':'None',
    'GarageType':'None',
    'GarageFinish':'None',
    'GarageQual':'None',
    'GarageCond':'None',
    'GarageYrBlt':0,
    'GarageArea':0,
    'GarageCars':0,
    'BsmtQual':'None',
    'BsmtCond':'None',
    'BsmtExposure':'None',
    'BsmtFinType1':'None',
    'BsmtFinType2':'None',
    'MasVnrType':'None',
    'MasVnrArea':0,
    'MSZoning':'RL',
    'Functional':'Typ',
    'Electrical':'SBrkr',
    'KitchenQual':'TA',
    'Exterior1st':'VinylSd',
    'Exterior2nd':'VinylSd',
    'SaleType':'WD',
    'MSSubClass':'None',
    'MiscFeature':'None'
}

t.to_numeric_columns = ['OverallQual']

#t.logify_columns.append('SalePrice')
t.logify_columns = ['LotArea']
#, 'GrLivArea', '1stFlrSF'))
#t.fill_na_mean = True
#t.fill_na_value = -99999
t.quantile_bins = 20
t.remove_outliers.extend((524, 1299))
t.dummify_at_init = True

t.separate_out_value('BsmtFinSF1', 0, 'NoBsmt')
t.separate_out_value('YearRemodAdd', np.nan, 'NoRemodAdd')
t.separate_out_value('PoolArea', 0, 'NoPool')

t.prepare()


#t.df_train.to_csv('temp.csv')

#t.df_train = pd.read_csv('../neural_network.bak/deep_train.csv', index_col=0)
#t.labels = pd.read_csv('../neural_network.bak/deep_labels.csv', header=None)

ncols = len(t.df_train.columns)
print 'columns: ', ncols

seed = 7
np.random.seed(seed)

X_train, y_train = X, y = shuffle(t.df_train.values, t.labels.values)

if hm_manual_validation > 0:
    X_train = X_train[:-hm_manual_validation]
    y_train = y_train[:-hm_manual_validation]

model = Sequential()
model.add(Dense(ncols, input_dim=ncols, activation='linear'))
model.add(Dense(ncols, activation='linear'))
model.add(Dense(ncols, activation='linear'))
model.add(Dense(ncols, activation='linear'))
model.add(Dense(1))

model.summary()

model.compile(loss='mean_squared_error', optimizer='rmsprop')

history = model.fit(X_train, y_train, epochs=10, batch_size=128)
print 'loss $$', np.sqrt(history.history['loss'][-1])

if hm_manual_validation > 0:
    # custom validation
    valX = X[-hm_manual_validation:]
    valY = y[-hm_manual_validation:]

    prediction_val = model.predict(valX)
    if use_log_label:
        prediction_val = np.exp(prediction_val)
        valY = np.exp(valY)

    diff_pred = prediction_val - valY

    print 'diff prediction $', np.mean(np.absolute(diff_pred))

prediction = model.predict(t.df_test.values)

df_predicted = pd.DataFrame(columns=['Id', 'SalePrice'])
df_predicted['Id'] = t.test_ids
df_predicted.set_index('Id')
df_predicted['SalePrice'] = prediction
df_predicted.to_csv('predicted.csv', sep=',', index=False)

print 'Predictions done.'