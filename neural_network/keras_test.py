import pandas as pd
import math
import json
import sys
import os
import collections
from tqdm import tqdm
from os import path

import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import backend as K

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from utils.utils import *
from utils.Training import *
from utils.kaggle import *
from deep_models import *

pd.options.mode.chained_assignment = None  # default='warn'

t = training()

t.dummify_at_init = True
t.dummify_drop_first = True
t.use_label_encoding = False
t.explode_possible_types_columns = True

if True:
    t.train_columns = [
        'OverallQual', 'TotalSF', 'Neighborhood', 'OverallCond', 'BsmtQual',
        'MSSubClass', 'GarageArea', 'BsmtUnfSF', 'YearBuilt', 'LotArea',
        'MSZoning', 'Fireplaces', 'Functional', 'HeatingQC', 'SaleCondition',
        'Condition1', 'BsmtExposure', 'GrLivArea', 'BsmtFinType1',
        'KitchenQual', 'BsmtFinSF1', 'Exterior1st', '2ndFlrSF', 'GarageCars',
        'ScreenPorch', 'WoodDeckSF', 'BsmtFullBath', 'CentralAir', '1stFlrSF',
        'HalfBath', 'PoolArea', 'GarageYrBlt', 'MasVnrType', 'ExterQual',
        'KitchenAbvGr', 'FullBath', 'LotConfig', 'Foundation', 'LowQualFinSF',
        'BedroomAbvGr', 'BsmtFinSF2', 'Condition2', 'PoolQC'
    ]

t.prepare()

ncols = len(t.df_train.columns)
print 'columns: ', ncols


X_train, y_train = t.df_train.values, t.labels.values


def root_mean_squared_error(y_true, y_pred):
    squared_diff = K.square(y_pred - y_true)
    return K.sqrt(K.mean(squared_diff, axis=-1))


inf = 999999
max_epochs = 1000
min_delta_score = 0.0001
n_scores = 8
alpha = 0.0002


model = Sequential()
model.add(Dense(ncols, input_dim=ncols, activation='linear',
                kernel_regularizer=l2(alpha)))
model.add(Dense(1))

model.compile(optimizer="rmsprop", loss=root_mean_squared_error,
              metrics=["accuracy"])


def train(x, y):
    history = model.fit(x, y, batch_size=32, shuffle=False, epochs=1)


def predict(x):
    return model.predict(x)


epoch = 0
making_progress = True
last_n_scores = collections.deque(n_scores * [inf], n_scores)
best_score = inf

while making_progress:

    print X_train.shape
    print y_train.shape
    rmse_score = custom_score_using_kfolds(
        train, predict, rmse, X_train, y_train, scale=True, doShuffle=True, n_splits=10)
    # rmse_score = custom_rmse_using_kfolds(
    #    train, predict, X_train, y_train, scale=True, doShuffle=True, n_splits=10)

    last_n_scores.appendleft(rmse_score)
    avg = np.array(last_n_scores).mean()
    diff = best_score - avg

    print 'RMSE:', rmse_score, 'avg:', avg, 'diff:', diff
    print '------------------------------------------------------------------------------------------------------------------------>', rmse_score

    making_progress = diff > min_delta_score
    best_score = avg

print 'KFold cross validation RMSE:', last_n_scores
print 'Final RMSE:', np.array(last_n_scores).mean()

print 'weigths', model.layers[0].get_weights()[0][:3]
sys.exit(0)
# acc = keras_deep_test_accuracy_for_model_using_kfolds(
#    build_model, t.df_train, t.labels, n_splits=4,epochs=epochs)
#print 'Cross validation R2:', acc

# ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True, verbose=0),

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0, patience=3, verbose=2, mode='auto')
]
# callbacks=callbacks

# replace this and fit manually
estimator = KerasRegressor(
    build_fn=build_model, nb_epoch=100, batch_size=128, verbose=True)


def test_accuracy_rmsle2(model, train, y, n_folds=5):
    kf = KFold(
        n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    score = np.sqrt(-cross_val_score(model, train.values, y.values, cv=kf))
    return score.mean()


rmse_score = test_accuracy_rmsle2(estimator, t.df_train, t.labels)

print 'RMSE', rmse_score

if False:
    model = None
    prediction = model.predict(t.df_test.values)

    df_predicted = pd.DataFrame(columns=['Id', 'SalePrice'])
    df_predicted['Id'] = t.test_ids
    df_predicted.set_index('Id')
    df_predicted['SalePrice'] = np.exp(prediction)
    df_predicted.to_csv('predicted.csv', sep=',', index=False)

    print 'Predictions done.'
