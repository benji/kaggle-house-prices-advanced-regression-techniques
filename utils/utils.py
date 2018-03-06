import pandas as pd
import math
import json
import sys
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.model_selection import KFold, cross_val_score, train_test_split, StratifiedKFold
from sklearn.utils import shuffle
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from keras import backend as K
from scipy.optimize import leastsq

pd.options.mode.chained_assignment = None  # default='warn'
import warnings
warnings.filterwarnings('ignore')


def dummify_col_with_schema(col, schema, df_train, df_test=None):
    if not col in df_train.columns:
        raise Exception('Could\'t find column', col, ' in training dataset')

    if df_test is not None:
        if len(df_train.columns) != len(df_test.columns):
            raise Exception(
                'Coming with different number of columns for train and test!')

    scol = schema['columns'][col]
    if scol['type'] == 'CATEGORICAL':
        # now conversion to categorical strings should work
        df_train[col] = df_train[col].astype(
            'category', categories=scol['categories'])

        if df_test is not None:
            df_test[col] = df_test[col].astype(
                'category', categories=scol['categories'])

    elif (scol['type'] == 'NUMERIC'):
        df_train[col] = df_train[col].astype('float64')
        if df_test is not None:
            if (col in df_test.columns):
                df_test[col] = df_test[col].astype('float64')

    else:
        print 'ERROR coltype not supported', col, scol
        sys.exit(1)

    df_train2 = pd.get_dummies(
        df_train, drop_first=True, columns=[col])

    newcolumns = set(df_train2.columns) - set(df_train.columns)
    # print 'New dummy columns:', newcolumns

    meaningful_columns = []
    meaningless_columns = []

    for nc in newcolumns:
        if df_train2[nc].nunique() == 1:
            # print 'Dummy column', nc, 'has only 1 value', df_train2[nc].iloc[
            #    0], ', removing it.'
            df_train2.drop(nc, 1, inplace=True)
            meaningless_columns.append(nc)
        else:
            meaningful_columns.append(nc)
        schema['columns'][nc] = {'type': 'NUMERIC'}

    if len(meaningful_columns) == 0:
        print df_train[col].head()
        raise Exception('Dumification of column', col,
                        'resulted in no meaningful columns')

    df_train = df_train2

    if df_test is not None:
        df_test = pd.get_dummies(
            df_test, drop_first=True, columns=[col])

        for mnlc in meaningless_columns:
            df_test.drop(mnlc, 1, inplace=True)

        if len(df_train.columns) != len(df_test.columns):
            print df_train.columns
            print df_test.columns
            raise Exception(
                'Dummifying produced a different number of columns for train and test!'
            )

    return df_train, df_test


# expects np arrays
# deprecated
def __custom_rmse_using_kfolds(trainFn,
                             predictFn,
                             X,
                             y,
                             n_splits=10,
                             doShuffle=True,
                             scale=True):
    return custom_score_using_kfolds(trainFn,
                                     predictFn,
                                     rmse,
                                     X,
                                     y,
                                     n_splits,
                                     doShuffle,
                                     scale)


# expects np arrays
def custom_score_using_kfolds(trainFn,
                              predictFn,
                              scoreFn,
                              X,
                              y,
                              n_splits=10,
                              doShuffle=True,
                              seed=-1):

    scores = []
    kf = KFold(n_splits=n_splits, shuffle=doShuffle)
    split_i = 1

    for train, test in kf.split(X):
        score = doScore(trainFn, predictFn, scoreFn,
                        X[train], X[test], y[train], y[test])
        #print 'split', split_i, 'score', score
        scores.append(score)
        split_i += 1

    return np.mean(scores)


def score_using_test_ratio(trainFn, predictFn, scoreFn,
                           X, y, test_ratio=.25, shuffle=True):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, shuffle=shuffle)
    return doScore(trainFn, predictFn, scoreFn, X_train, X_test, y_train, y_test)


def doScore(trainFn, predictFn, scoreFn, X_train, X_test, y_train, y_test):
    trainFn(X_train, y_train)
    y_pred = predictFn(X_test)
    return scoreFn(y_pred, y_test)


def doShuffle(X, y, seed=None, doShuffle=True):
    if not doShuffle:
        return X, y

    if seed is not None:
        return shuffle(X, y, random_state=seed)
    else:
        return shuffle(X, y)


def rmse(y_predicted, y_actual):
    if y_predicted.shape != y_actual.shape:
        print "ERROR: incompatible shapes", y_predicted.shape, y_actual.shape
        #y_predicted = [v[0] for v in y_predicted]
        sys.exit(1)
    tmp = np.power(y_actual - y_predicted, 2)
    tmp = tmp.mean()
    return np.sqrt(tmp)


def cod(y_pred, y_true):
    y_mean = np.mean(y_true)
    SSres = np.sum(np.power(y_true - y_pred, 2))
    SStot = np.sum(np.power(y_true - y_mean, 2))
    return 1 - SSres / SStot


def generate_least_square_best_fit(x, y, order):
    def func(coefs, x):
        sum = 0
        for i in range(order+1):
            sum += coefs[i]*x**i
        return sum

    def error_fn(coefs, x, y):
        return func(coefs, x) - y

    coefs, success = leastsq(error_fn, np.zeros(order+1), args=(x, y))

    return lambda x: func(coefs, x), coefs
