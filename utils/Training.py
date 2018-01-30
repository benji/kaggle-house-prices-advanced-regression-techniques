import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import LabelEncoder

from utils import *


class Training:
    logify_columns = []
    fill_na_value = None
    fill_na_mean = None
    remove_outliers = []
    dummify_at_init = False
    dummify_drop_first = True
    train_columns = []
    dummy_na = False
    quantile_bins = 1
    columns_na = {}
    to_numeric_columns = []
    drop_columns = []
    prefer_numerical = False
    use_label_encoding = False
    replace_values = []
    use_dummies_for_specific_columns = []

    # will be filled by prepare()
    labels = None
    test_ids = None

    def __init__(self, df_train, df_test=None, schema=None):
        self.df_train = df_train
        self.df_test = df_test
        self.schema = schema

    def prepare(self):
        self.idcol = self.schema['id']
        self.targetcol = self.schema['target']

        # MISSING VALUES
        self.fillna_columns_user_strategy()

        if self.fill_na_value is not None:
            self.fillna(self.fill_na_value)

        if self.fill_na_mean is not None:
            self._fill_na_mean()

        # REMOVE COLS WITH UNIQUE DATA
        self.remove_columns_with_unique_value()

        # REMOVE ROWS
        for o in self.remove_outliers:
            self.remove_outlier(o)

        # TRANSFORM VALUES
        self.df_train[self.targetcol] = self.df_train[self.targetcol].astype(
            'float64')

        self.do_replace_values()

        for c in self.logify_columns:
            self.logify(c)

        self.transform_to_numeric_columns()

        # SAVE IDS & LABELS
        self.test_ids = self.df_test[self.idcol]

        self.labels = self.df_train[self.targetcol]

        # REMOVE COLUMNS
        self.df_train.drop(self.targetcol, 1, inplace=True)

        if (self.idcol in self.df_train.columns):
            self.df_train.drop(self.idcol, 1, inplace=True)
            if self.df_test is not None:
                self.df_test.drop(self.idcol, 1, inplace=True)

        for col in self.drop_columns:
            print 'Dropping', col
            self.df_train = self.df_train.drop([col], axis=1)
            self.df_test = self.df_test.drop([col], axis=1)

        # SAFETY CHECK
        if self.diagnose_nas() > 0:
            raise Exception('Found NAs in data')

        # TRANSFORM COLUMNS
        if len(self.train_columns) > 0:
            self.retain_columns(self.train_columns)

        self.transform_preferred_to_numerical()

        if self.quantile_bins > 1:
            self.df_train, self.df_test = quantile_bin_all(
                self.schema, self.quantile_bins, self.df_train, self.df_test)
            if not self.dummify_at_init:
                raise Exception(
                    'You need to use dummies of you use quantile_bin')

        if self.dummify_at_init:
            self.df_train, self.df_test = self.do_dummify(True)

        self.do_label_encoding()

        self.df_train.to_csv('tmp.csv')
        print 'Prepared produced', self.df_train.shape[1], 'columns'
        print self.df_train.columns
        #self.fillna(-99999)

    def should_dummify_col(self, c):
        if len(self.use_dummies_for_specific_columns) == 0:
            return True

        if c in self.use_dummies_for_specific_columns:
            return True

        return False

    def do_dummify(self, verbose=False):
        train, test = self.df_train.copy(), self.df_test.copy()
        for c in train.columns:
            if self.should_dummify_col(c):
                if verbose:
                    print 'Dummifying columns', c
                train, test = dummify_col_with_schema(c, self.schema, train,
                                                      test)
        return train, test

    def should_label_encode_col(self, c):
        if self.use_label_encoding == False:
            return False

        if c not in self.schema['columns']:
            return False

        coldata = self.schema['columns'][c]

        if c in self.use_dummies_for_specific_columns:
            return False

        if coldata['type'] == 'NUMERIC':
            return False

        if 'tonum' in coldata and coldata['tonum'] == True:
            return False

        return True

    def retain_columns(self, columns):
        self.retain_columns_df(self.df_train, columns)
        if self.df_test is not None:
            self.retain_columns_df(self.df_test, columns)
        print 'Retained', len(self.df_train.columns), 'columns in dataframe'

    def retain_columns_df(self, df, columns):
        for col in df.columns:
            if (not (col in columns)):
                df.drop(col, 1, inplace=True)

    def _fill_na_mean(self):
        self.df_train.fillna(self.df_train.mean(), inplace=True)
        self.df_test.fillna(self.df_train.mean(), inplace=True)

    def remove_outlier(self, o):
        idx = self.df_train[self.df_train[self.idcol] == o].index
        self.df_train.drop(idx, inplace=True)

    def logify(self, col):
        self.df_train[col] = np.log(self.df_train[col])
        if self.df_test is not None and col in self.df_test.columns:
            self.df_test[col] = np.log(self.df_test[col])

    def fillna(self, val):
        self.df_train.fillna(val, inplace=True)
        if self.df_test is not None:
            self.df_test.fillna(val, inplace=True)

    def separate_out_value(self, col, value, newcol):
        '''
        will create a column with value 1 if this col is == value, 0 otherwise
        matching rows will be set to np.nan in former column
        '''

        def matchesZeroTransform(row):
            return 1 if row[col] == value else 0

        def otherwiseTransform(row):
            return row[col] if row[col] != value else np.nan

        for df in [self.df_train, self.df_test]:
            if df is not None:
                df[newcol] = df.apply(matchesZeroTransform, axis=1)
                #df[col] = df.apply(otherwiseTransform, axis=1)

        self.schema[newcol] = {"categories": [0, 1], "type": "CATEGORICAL"}

    def fillna_columns_user_strategy(self):
        for c in self.columns_na:
            v = self.columns_na[c]
            print 'filling NAs in column', c, 'to', v
            self.df_train[c] = self.df_train[c].fillna(v)
            self.df_test[c] = self.df_test[c].fillna(v)

            # add to schema of not in there already
            coldata = self.schema['columns'][c]
            if 'categories' in coldata and v not in coldata['categories']:
                coldata['categories'].append(v)

    def transform_to_numeric_columns(self):
        for c in self.to_numeric_columns:
            print 'Transform to numeric', c
            self.df_train[c] = self.df_train[c].astype(str)
            self.df_test[c] = self.df_test[c].astype(str)

    def diagnose_nas(self):
        maxtrain = self.df_train.isnull().sum().max()
        if maxtrain == 0:
            print 'No NA found in train dataset'
            maxtest = self.df_test.isnull().sum().max()
            if maxtest == 0:
                print 'No NA found in test dataset'
            else:
                print 'NA values found in test dataset'
                print self.df_test.isnull().sum().sort_values(
                    ascending=False)[:6]
                return maxtest
        else:
            print 'NA values found in train dataset'
            print self.df_train.isnull().sum().sort_values(ascending=False)[:6]
            return maxtrain

    def transform_preferred_to_numerical(self):

        for col in self.schema['columns']:
            coldata = self.schema['columns'][col]
            if 'tonum' in coldata and coldata['tonum'] == True:
                self.df_train[col] = self.df_train[col].astype(str)
                self.df_test[col] = self.df_test[col].astype(str)

                cats = coldata['categories']
                print 'Converting', col, 'to numeric value'
                idx = 0
                for val in list(cats):
                    self.df_train[col][self.df_train[col] == val] = idx
                    self.df_test[col][self.df_test[col] == val] = idx
                    idx += 1

                self.df_train[col] = self.df_train[col].astype('float64')
                self.df_test[col] = self.df_test[col].astype('float64')

    def do_replace_values(self):
        for rv in self.replace_values:
            col = rv[0]
            self.df_train[col][self.df_train[col] == rv[1]] = rv[2]
            self.df_test[col][self.df_test[col] == rv[1]] = rv[2]

    def remove_columns_with_unique_value(self):
        cols_before = self.df_train.columns
        self.df_train = self.df_train.loc[:, (
            self.df_train != self.df_train.iloc[0]).any()]
        cols_after = self.df_train.columns

        cols_removed = set(cols_before) - set(cols_after)

        if len(cols_removed) > 0:
            print 'Removed constant columns', cols_removed
            self.df_test = self.df_test[cols_after]

    def do_label_encoding(self):
        for c in self.df_train.columns:
            if self.should_label_encode_col(c):
                print 'Label encoding:', c

                self.df_train[c] = self.df_train[c].astype(str)
                self.df_test[c] = self.df_test[c].astype(str)

                cats = np.asarray(self.schema['columns'][c]['categories'])
                #print 'cats', cats
                lbl = LabelEncoder()
                lbl.fit(cats)
                self.df_train[c] = lbl.transform(self.df_train[c].values)
                self.df_test[c] = lbl.transform(self.df_test[c].values)
