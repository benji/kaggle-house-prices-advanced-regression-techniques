import pandas as pd
import math, json, sys
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib import style

pd.options.mode.chained_assignment = None  # default='warn'
style.use('ggplot')
import warnings
warnings.filterwarnings('ignore')

dummify_at_init = True
dummify_at_runtime = False
dummify_drop_first = False
only_numeric = False
use_log_saleprice = True
scaler = StandardScaler()
clf = LinearRegression()

schema = json.loads(open('../schema.json', 'r').read())


def dummify(df_train, df_test=None):
    for col in schema:
        if col in df_train.columns:
            #print col
            scol = schema[col]
            if (scol['type'] == 'BINARY' or scol['type'] == 'CATEGORICAL'):
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


def remove_columns_with_missing_data(df_train, df_test):
    total = df_train.isnull().sum().sort_values(ascending=False)
    percent = (df_train.isnull().sum() /
               df_train.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat(
        [total, percent], axis=1, keys=['Total', 'Percent'])
    print missing_data.head()

    rejected_cols = missing_data[missing_data['Total'] > 1].index

    if (rejected_cols.any()):
        print 'Columns with missing data will be removed:', rejected_cols
        df_train = df_train.drop(rejected_cols, 1)
        if df_test is not None:
            df_test = df_test.drop(rejected_cols, 1)

    return df_train, df_test


def prepare_data(df_train, df_test):
    # Remove Outliers
    df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)
    df_train = df_train.drop(df_train[df_train['Id'] == 524].index)

    # remove outliers > $700k
    #df_train = df_train.drop(df_train.index[691])
    #df_train = df_train.drop(df_train.index[1182])

    # fillnas with mean
    df_train = df_train.fillna(df_train.mean())
    df_test = df_test.fillna(df_train.mean())

    # remove columns with missing data
    # df_train, df_test = remove_columns_with_missing_data(df_train, df_test)

    # safety check
    nMissing = df_train.isnull().sum().max()
    if nMissing > 0:
        print 'Found Missing values:', nMissing
        sys.exit(1)

    if dummify_at_init:
        df_train, df_test = dummify(df_train, df_test)

    # drop Id, SalePrice
    y = df_train['SalePrice']
    df_train = df_train.drop('SalePrice', 1)

    test_ids = df_test['Id']

    if ('Id' in df_train.columns):
        df_train = df_train.drop('Id', 1)
        df_test = df_test.drop('Id', 1)

    return y, df_train, df_test, test_ids


def retain_columns(columns, df_train, df_test=None):
    df_train = retain_columns_df(df_train, columns)
    if df_test is not None:
        df_test = retain_columns_df(df_test, columns)

    return df_train, df_test


def retain_columns_df(df, columns):
    for col in df.columns:
        if (not (col in columns) and col != 'SalePrice'):
            df = df.drop(col, 1)
    return df


def test_accuracy(df_train, y, passes=1):
    if dummify_at_runtime:
        df_train, _ = dummify(df_train)

    X = np.array(df_train)
    scaler.fit(df_train)
    X = scaler.transform(X)

    accuracies = []

    for _ in range(passes):
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(
            X, y, test_size=0.2)

        clf.fit(X_train, y_train)
        accuracy = clf.score(X_test, y_test)
        #print accuracy

        if (accuracy >= 0 and accuracy <= 1):
            accuracies.append(accuracy)

    return np.mean(accuracies)


def generate_predictions(y, df_train, df_test, test_ids):
    if dummify_at_runtime:
        df_train, df_test = dummify(df_train, df_test)

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
