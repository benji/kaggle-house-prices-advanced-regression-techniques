import pandas as pd
import numpy as np
import sys, os
from sklearn.preprocessing import LabelEncoder

from utils import *


class Training:
    def __init__(self, df_train, df_test=None, schema=None):
        self.df_train = df_train
        self.df_test = df_test
        self.schema = schema
        self.logify_columns = []
        self.fill_na_value = None
        self.fill_na_mean = None
        self.remove_outliers = []
        self.dummify_at_init = False
        self.dummify_drop_first = True
        self.train_columns = []
        self.dummy_na = False
        self.quantile_bins = 1
        self.columns_na = {}
        self.to_numeric_columns = []
        self.drop_columns = []
        self.prefer_numerical = False
        self.use_label_encoding = False
        self.replace_values = []
        self.use_dummies_for_specific_columns = []
        self.numerical_singularities_columns = []
        self.enable_transform_preferred_to_numerical = True

        # will be filled by prepare()
        self.labels = None
        self.test_ids = None

    def prepare(self):
        self.idcol = self.schema['id']
        self.targetcol = self.schema['target']

        for c in self.df_train.columns:
            if c != self.idcol and c != self.targetcol and not c in self.schema['columns']:
                raise Exception('Column', c, 'not defined in schema.')

        # EXCLUDE ROWS

        for o in self.remove_outliers:
            self.remove_outlier(o)

        # IDS

        self.test_ids = self.df_test[self.idcol]
        self.train_ids = self.df_train[self.idcol]

        if (self.idcol in self.df_train.columns):
            self.df_train.drop(self.idcol, 1, inplace=True)
            if self.df_test is not None:
                self.df_test.drop(self.idcol, 1, inplace=True)

        # MODIFY VALUES

        self.do_replace_values()

        self.fillna_per_column()

        if self.fill_na_value is not None:
            self.fillna(self.fill_na_value)

        if self.fill_na_mean is not None:
            self._fill_na_mean()

        self.categoricals_as_string()

        for c in self.logify_columns:
            self.logify(c)

        # LABELS

        self.labels = self.df_train[self.targetcol]

        self.df_train.drop(self.targetcol, 1, inplace=True)

        # CLEANUP UNIQUE VALUE COLS

        self.remove_columns_with_unique_value()

        # TRANSFORM COLUMNS

        self.transform_to_numeric_columns()

        if self.enable_transform_preferred_to_numerical:
            self.transform_preferred_to_numerical()

        for c in self.numerical_singularities_columns:
            self.numerical_singularities(c, 0)

        # REMOVE COLS

        if len(self.train_columns) > 0:
            self.retain_columns(self.train_columns)

        for col in self.drop_columns:
            self.do_drop_column(col)

        # CATEGORICALS

        if self.quantile_bins > 1:
            self.df_train, self.df_test = quantile_bin_all(
                self.schema, self.quantile_bins, self.df_train, self.df_test)
            if not self.dummify_at_init:
                raise Exception(
                    'You need to use dummies of you use quantile_bin')

        if self.dummify_at_init:
            self.df_train, self.df_test = self.do_dummify(
                self.df_train, self.df_test, True)

        self.do_label_encoding()

        #self.df_train.to_csv('tmp.csv')
        print 'Prepared produced', self.df_train.shape[1], 'columns'
        print self.df_train.columns
        #self.fillna(-99999)

        # SAFETY CHECK
        if self.diagnose_nas() > 0:
            raise Exception('Found NAs in data')

    def get_columns_with_types(self, types):
        cols = []
        for col in self.df_train.columns:
            if col != self.schema['target'] and col != self.schema['id'] and self.schema['columns'][col]['type'] in types:
                cols.append(col)

        return cols

    def categorical_columns(self):
        return self.get_columns_with_types(['CATEGORICAL', 'BINARY'])

    def numerical_columns(self):
        return self.get_columns_with_types(['NUMERIC'])

    def categoricals_as_string(self):
        for col in self.categorical_columns():
            print 'as string', col
            self.df_train[col] = self.df_train[col].astype(str)
            self.df_test[col] = self.df_test[col].astype(str)

    def should_dummify_col(self, c):
        if self.schema['columns'][c]['type'] == 'NUMERIC':
            return False

        if len(self.use_dummies_for_specific_columns) == 0:
            return True

        if c in self.use_dummies_for_specific_columns:
            return True

        return False

    def do_dummify(self, train, test=None, verbose=False):
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

        if self.enable_transform_preferred_to_numerical and 'tonum' in coldata and coldata['tonum'] == True:
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
        print 'Logifying column', col
        self.df_train[col] = np.log1p(self.df_train[col])
        if self.df_test is not None and col in self.df_test.columns:
            self.df_test[col] = np.log1p(self.df_test[col])

    def fillna(self, val):
        self.df_train.fillna(val, inplace=True)
        if self.df_test is not None:
            self.df_test.fillna(val, inplace=True)

    def separate_out_value(self, col, value, newcol):
        '''
        will create a column with value 1 if this col is == value, 0 otherwise
        matching rows will be set to np.nan in former column
        '''

        if not col in self.df_train.columns:
            return

        def matchesZeroTransform(row):
            return 1 if row[col] == value else 0

        def otherwiseTransform(row):
            return row[col] if row[col] != value else np.nan

        for df in [self.df_train, self.df_test]:
            if df is not None:
                df[newcol] = df.apply(matchesZeroTransform, axis=1)
                self.schema['columns'][newcol] = {'type': 'NUMERIC'}
                #df[col] = df.apply(otherwiseTransform, axis=1)

        self.schema[newcol] = {"categories": [0, 1], "type": "CATEGORICAL"}

    def fillna_per_column(self):
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

                self.schema['columns'][col] = {'type': 'NUMERIC'}

    def do_replace_values(self):
        for rv in self.replace_values:
            col = rv[0]
            self.df_train[col][self.df_train[col] == rv[1]] = rv[2]
            self.df_test[col][self.df_test[col] == rv[1]] = rv[2]

            if self.df_train[self.df_train[col] == rv[1]].shape[0] > 0:
                raise Exception('Failed to replace values')

    def remove_columns_with_unique_value(self):
        cols_before = self.df_train.columns
        self.df_train = self.df_train.loc[:, (
            self.df_train != self.df_train.iloc[0]).any()]
        cols_after = self.df_train.columns

        cols_removed = set(cols_before) - set(cols_after)

        if len(cols_removed) > 0:
            print 'Removed constant columns', cols_removed
            cols_tokeep = set(cols_after) - set([self.targetcol])
            self.df_test = self.df_test[list(cols_tokeep)]

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

                if self.df_train[c].nunique() == 1:
                    raise Exception('Label encoding of column', c,
                                    'has only 1 value',
                                    self.df_train[c].iloc[0])

                print 'do_label_encoding - changing type to NUMERIC', c
                self.schema['columns'][c] = {'type': 'NUMERIC'}

    def sanity(self, throw=True):
        for df in [self.df_train, self.df_test]:
            print 'check'
            for c in df.columns:
                tmp = df[c][df[c] > 100000]
                if tmp.shape[0] > 0:
                    print tmp.head()
                    if throw:
                        raise Exception('Found bad value')
                    else:
                        return False
                tmp = df[c][df[c] < -100000]
                if tmp.shape[0] > 0:
                    print tmp.head()
                    if throw:
                        raise Exception('Found bad value')
                    else:
                        return False
                tmp = df[c][df[c].isnull()]
                if tmp.shape[0] > 0:
                    print tmp.head()
                    if throw:
                        raise Exception('Found bad value')
                    else:
                        return False

        print 'check'
        tmp = self.labels[self.labels.isnull()]
        if tmp.shape[0] > 0:
            print tmp.head()
            if throw:
                raise Exception('Found bad value in y')
            else:
                return False

        return True

    def do_drop_column(self, col):
        if col in self.df_train.columns:
            print 'Dropping', col
            self.df_train = self.df_train.drop([col], axis=1)
            self.df_test = self.df_test.drop([col], axis=1)

    def numerical_singularities(self, col, val):
        print 'Extracting singular value', val, 'from', col

        nonval = self.df_train[self.df_train[col] != val][col]

        # get best fit line for non zero elements
        z = np.polyfit(nonval, self.labels[nonval.index], 1)

        # where should the zero elements be to keep that line?
        vals = self.df_train[self.df_train[col] == val]
        avg0 = self.labels[vals.index].mean()
        x = (avg0 - z[1]) / z[0]

        self.df_train[col][self.df_train[col] == val] = x
        self.df_test[col][self.df_test[col] == val] = x

    def save(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)

        train = self.df_train.copy()
        train[self.idcol] = self.train_ids
        train[self.targetcol] = self.labels
        train.to_csv(folder + '/train.csv', index=False)

        test = self.df_test.copy()
        test[self.idcol] = self.test_ids
        test.to_csv(folder + '/test.csv', index=False)

        with open(folder + '/schema.json', 'w') as outfile:
            json.dump(self.schema, outfile, indent=4)