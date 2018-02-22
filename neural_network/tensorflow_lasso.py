import pandas as pd
import math
import json
import sys
import os
import time
import collections
from tqdm import tqdm
from os import path

import numpy as np

import tensorflow as tf

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from utils.utils import *
from utils.Training import *
from utils.kaggle import *
from deep_models import *

pd.options.mode.chained_assignment = None  # default='warn'

np.random.seed(int(time.time()))

# kaggle rmse score: 0.12179

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

use_cross_validation = False

if use_cross_validation:
    X_train, X_test, y_train, y_test = t.train_test_split(test_size=.25)
else:
    X_train, y_train = t.df_train.values, t.labels.values

lasso_lambda = 50 / n_samples
learning_rate = 0.01
training_epochs = 10000

n_samples = X_train.shape[0]
n_dims = X_train.shape[1]

X = tf.placeholder("float", shape=(None, n_dims))
Y = tf.placeholder("float", shape=(None))

W = tf.Variable(tf.random_normal([n_dims, 1]), name="weight")
layer1 = tf.matmul(X, W)
b = tf.Variable(tf.random_normal([1, 1]), name="bias")
pred = tf.reduce_sum(layer1, 1) + b

diff = pred - Y
squared_diff = tf.pow(diff, 2)
mse = tf.reduce_sum(squared_diff / n_samples)
rmsecost = tf.sqrt(mse)

l1_norm = tf.reduce_sum(tf.abs(W))

lassocost = rmsecost + lasso_lambda * l1_norm


optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(
    lassocost)

init = tf.global_variables_initializer()


def predict(sess, Xs, ids, filename):
    y_pred = sess.run(pred, feed_dict={X: Xs})

    df_predicted = pd.DataFrame(columns=['Id', 'SalePrice'])
    df_predicted['Id'] = ids
    df_predicted.set_index('Id')
    df_predicted['SalePrice'] = np.exp(y_pred[0])
    df_predicted.to_csv(filename, sep=',', index=False)

    print 'Predictions done.'


with tf.Session() as sess:
    sess.run(init)

    for epoch in range(training_epochs):

        X_train, y_train = shuffle(X_train, y_train)

        sess.run(optimizer, feed_dict={X: X_train, Y: y_train})

        if epoch % 500 == 0:
            y_pred = sess.run(pred, feed_dict={X: X_train, Y: y_train})
            cod_score = cod(y_pred, y_train)
            print 'Epoch', epoch, 'COD', cod_score
            #print 'X', sess.run(X,feed_dict={ X: X_train })[:5]
            #print 'layer1', sess.run(layer1,feed_dict={ X: X_train })[:5]
            #print 'pred', sess.run(pred,feed_dict={ X: X_train })[:5]
            #print 'y_true', sess.run(Y,feed_dict={ Y: y_train })[:5]
            #print 'diff', sess.run(diff, feed_dict={X: X_train,Y: y_train})[:5]
            print 'W', sess.run(W)[:5]
            print 'b', sess.run(b)
            print 'W shape', sess.run(W).shape
            print 'b shape', sess.run(b).shape
            print 'rmsecost train', sess.run(
                rmsecost, feed_dict={X: X_train, Y: y_train})
            if use_cross_validation:
                print 'rmsecost test', sess.run(
                    rmsecost, feed_dict={X: X_test, Y: y_test})

    print("Exiting training loop")

    print 'Making predictions'

    #X_test = scaler.transform(t.df_test.values)
    X_test = t.df_test.values

    print 'Xtrain:', X_train[:5]
    print 'ytrain:', np.exp(y_train[:5])

    print 'Xtest:', X_test[:5]

    predict(sess, X_test, t.test_ids, 'predicted_test.csv')
    predict(sess, X_train, t.train_ids, 'predicted_train.csv')

    np.savetxt('saleprice.csv', np.exp(y_train), fmt='%f')
