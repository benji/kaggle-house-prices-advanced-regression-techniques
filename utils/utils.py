import pandas as pd
import math, json, sys
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

pd.options.mode.chained_assignment = None  # default='warn'
import warnings
warnings.filterwarnings('ignore')

dummify_drop_first = False

scaler = StandardScaler()
clf = LinearRegression()


def dummify_with_schema(schema, df_train, df_test=None):
    for col in schema['columns']:
        if col in df_train.columns:
            #print col
            scol = schema['columns'][col]
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

    df_train2 = pd.get_dummies(df_train, drop_first=dummify_drop_first)

    # print some stats
    #print 'Dummified columns:', set(df_train.columns) - set(df_train2.columns)

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


def dummify(schema, df_train, df_test=None):
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
        #print accuracy

        if (accuracy > 10 or accuracy < -10):
            pass
            #np.savetxt('expected.txt', y_test, fmt='%f')
            #np.savetxt('predicted.txt', clf.predict(X_test), fmt='%f')
            #raise Exception('Accuracy is way off', accuracy)
        else:
            accuracies.append(accuracy)

    return np.mean(accuracies)


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
