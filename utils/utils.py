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

dummify_drop_first = False

scaler = StandardScaler()
clf = LinearRegression()


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
        df_train, drop_first=dummify_drop_first, columns=[col])

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
            df_test, drop_first=dummify_drop_first, columns=[col])

        for mnlc in meaningless_columns:
            df_test.drop(mnlc, 1, inplace=True)

        if len(df_train.columns) != len(df_test.columns):
            print df_train.columns
            print df_test.columns
            raise Exception(
                'Dummifying produced a different number of columns for train and test!'
            )

    return df_train, df_test


def dummify_with_schema_deprecated(schema, df_train, df_test=None):
    for col in schema['columns']:
        if col in df_train.columns:
            # print col
            scol = schema['columns'][col]
            if scol['type'] == 'CATEGORICAL':
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
                print 'ERROR coltype not supported', scol
                sys.exit(1)

    df_train2 = pd.get_dummies(df_train, drop_first=dummify_drop_first)

    # print some stats
    # print 'Dummified columns:', set(df_train.columns) - set(df_train2.columns)

    df_train = df_train2

    if df_test is not None:
        df_test = pd.get_dummies(df_test, drop_first=dummify_drop_first)

        if len(df_train.columns) != len(df_test.columns):
            print df_train.columns
            print df_test.columns
            raise Exception(
                'Dummifying produced a different number of columns for train and test!'
            )

    return df_train, df_test


def dummify_deprecated(schema, df_train, df_test=None):
    for col in schema:
        if col in df_train.columns:
            # print col
            scol = schema[col]
            if scol['type'] == 'CATEGORICAL':
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
                print 'ERROR coltype not supported', scol
                sys.exit(1)

    df_train = pd.get_dummies(df_train, drop_first=dummify_drop_first)
    if df_test is not None:
        df_test = pd.get_dummies(df_test, drop_first=dummify_drop_first)

    return df_train, df_test


def test_accuracy(df_train, y, passes=1):
    X = np.array(df_train)
    scaler.fit(df_train)

    X = scaler.transform(X)

    accuracies = []

    for _ in range(passes):
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(
            X, y, test_size=0.3)

        clf.fit(X_train, y_train)
        accuracy = clf.score(X_test, y_test)
        # print accuracy

        if False:  # (accuracy > 10 or accuracy < -10):
            print 'acc off', accuracy
            # pass
            # df_train.to_csv('train.csv')
            # np.savetxt('expected.txt', y_test, fmt='%f')
            # np.savetxt('predicted.txt', clf.predict(X_test), fmt='%f')
            # raise Exception('Accuracy is way off', accuracy)
        else:
            accuracies.append(accuracy)

    return np.mean(accuracies)


def test_accuracy_kfolds(df_train, y, n_splits=4):
    return test_accuracy_for_model_using_kfolds(clf, df_train, y, n_splits)


def test_accuracy_for_model_using_kfolds(model,
                                         df_train,
                                         y,
                                         n_splits=4,
                                         scale=True):
    X = np.array(df_train)

    if scale:
        scaler.fit(df_train)
        X = scaler.transform(X)

    X, y = shuffle(X, y)

    y = np.array(y)

    allcoefs = []

    kf = KFold(n_splits=n_splits, shuffle=True)
    for train, test in kf.split(X):
        model.fit(X[train], y[train])
        coef_of_determination = model.score(X[test], y[test])

        allcoefs.append(coef_of_determination)

    return np.mean(allcoefs)


# expects np arrays
def custom_rmse_using_kfolds(trainFn,
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
                              scale=True,
                              seed=-1):
    if scale:
        scaler.fit(X)
        X = scaler.transform(X)

    scores = []
    kf = KFold(n_splits=n_splits, shuffle=doShuffle)
    split_i = 1

    for train, test in kf.split(X):
        score = doScore(trainFn, predictFn, scoreFn,
                        X[train], X[test], y[train], y[test])
        #print 'split', split_i, 'score', score
        scores.append(score)
        split_i += 1

    print 1
    print scores
    print np.array(scores).shape
    print np.array(scores).mean()

    return np.mean(scores)


def score_using_test_ratio(trainFn, predictFn, scoreFn,
                           X, y, test_ratio=.25):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio)
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
    tmp = np.power(y_actual - y_predicted, 2) / y_actual.shape[0]
    return np.sqrt(np.sum(tmp))


def cod(y_pred, y_true):
    y_mean = np.mean(y_true)
    SSres = np.sum(np.power(y_true - y_pred, 2))
    SStot = np.sum(np.power(y_true - y_mean, 2))
    return 1 - SSres / SStot


def keras_deep_test_accuracy_for_model_using_kfolds(model_fn,
                                                    df_train,
                                                    y,
                                                    n_splits=4,
                                                    epochs=10):
    model = model_fn()

    X = np.array(df_train)
    scaler.fit(df_train)
    X = scaler.transform(X)

    y = np.array(y)

    scores = []

    kf = KFold(n_splits=n_splits, shuffle=True)
    for train, test in kf.split(X):
        history = model.fit(X[train], y[train], epochs=epochs, batch_size=128)

        y_pred = model.predict(X[test])

        y_pred = np.array([x[0] for x in y_pred])

        if True:
            bad_idx = np.argwhere((y_pred < 1000) | (y_pred > 1000000))
            y_pred[bad_idx] = 200000
        else:
            bad_idx = np.argwhere((y_pred < 0) | (y_pred > 100))
            y_pred[bad_idx] = 10

        score = r2_score(y[test], y_pred)

        print y_pred[:5]
        print y[test][:5]
        print 'score:', score
        if score < 0:
            pred = np.array(y_pred)
            true = np.array(y[test])
            diff = true - pred

            np.savetxt('ypred.txt', pred, fmt='%f')
            np.savetxt('ytest.txt', true, fmt='%f')
            np.savetxt('diff.txt', diff, fmt='%f')
            print 'error'
            sys.exit(1)

        scores.append(score)

    return np.mean(scores)


#    return r2_score(y_true, y_pred)

# def test_accuracy_rmsle(model, train, y, n_folds=5):
#    kf = KFold(
#        n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
#    score = np.sqrt(-cross_val_score(model, train.values,
#                                     y.values, scoring="neg_mean_squared_error", cv=kf))
#    return score.mean()


def generate_predictions(y,
                         df_train,
                         df_test,
                         test_ids,
                         use_log_saleprice=True):

    if df_train.shape[1] != df_test.shape[1]:
        raise Exception(
            'train and test datasets don\'t have the same number of columns')

    X = np.array(df_train)
    X_predict = np.array(df_test)

    scaler.fit(df_train)
    X = scaler.transform(X)
    X_predict = scaler.transform(X_predict)

    X, y = shuffle(X, y)
    clf.fit(X, y)

    y_predicted = clf.predict(X_predict)

    if use_log_saleprice:
        y_predicted = np.exp(y_predicted)

    y_predicted[y_predicted <= 0] = 200000
    y_predicted[y_predicted > 3E6] = 200000

    df_predicted = pd.DataFrame(columns=['Id', 'SalePrice'])
    df_predicted['Id'] = test_ids

    df_predicted['SalePrice'] = y_predicted

    df_predicted.to_csv('predicted.csv', sep=',', index=False)

    print 'Done.'


def quantile_bin_all(schema, nbins, df_train, df_test):
    for col in schema['columns']:
        if schema['columns'][col]['type'] == 'NUMERIC':
            print 'Quantile_bin on numeric column', col
            df_train, df_test = quantile_bin(df_train, df_test, col, nbins)
    return df_train, df_test


def quantile_bin(df_train, df_test, col, nbins):
    # numeric to categorical
    df_train[col], bins = pd.qcut(
        df_train[col], nbins, duplicates='drop', retbins=True)

    df_test[col] = pd.cut(df_test[col], bins=bins)

    # categorical to binaries
    return pd.get_dummies(
        df_train, columns=[col], drop_first=True), pd.get_dummies(
            df_test, columns=[col], drop_first=True)


def separate_out_value(df, col, value, newcol):
    '''
    will create a column with value 1 if this col is == value, 0 otherwise
    matching rows will be set to np.nan in former column
    '''

    def matchesZeroTransform(row):
        return 1 if row[col] == value else 0

    def otherwiseTransform(row):
        return row[col] if row[col] != value else np.nan

    df[newcol] = df.apply(matchesZeroTransform, axis=1)
    df[col] = df.apply(otherwiseTransform, axis=1)


def get_pandas_types(schema):
    types = {}

    for c in schema['columns']:
        if schema['columns'][c]['type'] == 'NUMERIC':
            types[c] = float
        else:
            types[c] = str

    print types
    return types


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
