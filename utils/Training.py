import pandas as pd
import numpy as np
import sys
import os
import yaml
import copy
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

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
        self.use_label_encoding = False
        self.replace_values = []
        self.use_dummies_for_specific_columns = []
        self.numerical_singularities_columns = []
        # self.enable_transform_preferred_to_numerical = True
        self.categoricals_to_mean = []
        self.replace_all_categoricals_with_mean = False
        self.shuffle = True
        self.verbose = True
        self.explode_possible_types_columns = True
        self.scale = True
        self.linearize_all_numerical = False

        # will be filled by prepare()
        self.labels = None
        self.test_ids = None
        self.train_ids = None

    def prepare(self):
        self.idcol = self.schema['id']
        self.targetcol = self.schema['target']

        # VALIDATION CHECK

        # check schema has all columns
        for c in self.df_train.columns:
            if c != self.idcol and c != self.targetcol and not c in self.schema['columns']:
                raise Exception('Column', c, 'not defined in schema.')

        # check test and train have same columns
        if self.df_test is not None:
            train_cols = self.df_train.columns.copy().tolist()
            train_cols.remove(self.targetcol)
            assert train_cols == self.df_test.columns.tolist()

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

        # REMOVE COLS

        if len(self.train_columns) > 0:
            self.retain_columns(self.train_columns)

        for col in self.drop_columns:
            self.do_drop_column(col)

        # TRANSFORM COLUMNS

        for c in self.to_numeric_columns:
            self.convert_categorical_column_to_numerical(c)

        if self.explode_possible_types_columns:
            self.do_explode_possible_types_columns()

        # if self.enable_transform_preferred_to_numerical:
        #    self.transform_preferred_to_numerical()

        if self.linearize_all_numerical:
            for c in self.numerical_columns():
                self.linearize_polynomial_fit(c, 2, 0)

        for c, v in self.numerical_singularities_columns:
            if c in self.df_train.columns:
                self.numerical_singularities(c, v)

        # SHUFFLE

        if self.shuffle:
            self.do_shuffle()

        # CATEGORICALS

        if self.replace_all_categoricals_with_mean:
            self.do_replace_all_categoricals_with_mean()
        else:
            for c in self.categoricals_to_mean:
                self.replace_categorical_with_mean(c)

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

        # Final : scaling

        if self.scale:
            self.do_scale()

        print 'Produced', self.df_train.shape[1], 'columns'
        print self.df_train.columns

        # POST PREP SAFETY CHECK

        assert self.diagnose_nas() == 0

        if self.df_test is not None:
            assert self.df_train.columns.tolist() == self.df_test.columns.tolist()

    def do_scale(self):
        print 'scaling ...'
        columns = self.df_train.columns
        scaler = StandardScaler()
        scaler.fit(self.df_train[columns])
        self.df_train[columns] = scaler.transform(self.df_train[columns])
        self.df_test[columns] = scaler.transform(self.df_test[columns])

    def get_columns(self, types=None):
        cols = []
        for col in self.df_train.columns:
            if col != self.schema['target'] and col != self.schema['id'] and (types is None or self.schema['columns'][col]['type'] in types):
                cols.append(col)

        return cols

    def categorical_columns(self):
        return self.get_columns(['CATEGORICAL'])

    def numerical_columns(self):
        return self.get_columns(['NUMERIC'])

    def coltype(self, c):
        return self.schema['columns'][c]['type']

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
                if self.verbose or verbose:
                    print 'Dummifying columns', c
                train, test = dummify_col_with_schema(
                    c, self.schema, train, test)
        return train, test

    def do_dummify_column(self, c, verbose=False):
        if self.verbose or verbose:
            print 'Dummifying columns', c
        self.df_train, self.df_test = dummify_col_with_schema(
            c, self.schema, self.df_train, self.df_test)
        return True

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

        # if self.enable_transform_preferred_to_numerical and 'tonum' in coldata and coldata['tonum'] == True:
        #    return False

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
                # df[col] = df.apply(otherwiseTransform, axis=1)

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
                if 'NA' in coldata['categories']:
                    coldata['categories'] = [
                        v if x == 'NA' else x for x in coldata['categories']
                    ]
                else:
                    coldata['categories'].append(v)

    def diagnose_nas(self):
        maxtrain = self.df_train.isnull().sum().max()
        if maxtrain == 0:
            print 'No NA found in train dataset'
            maxtest = self.df_test.isnull().sum().max()
            if maxtest == 0:
                print 'No NA found in test dataset'
                return 0
            else:
                print 'NA values found in test dataset'
                print self.df_test.isnull().sum().sort_values(
                    ascending=False)[:6]
                return maxtest
        else:
            print 'NA values found in train dataset'
            print self.df_train.isnull().sum().sort_values(ascending=False)[:6]
            return maxtrain

    # DEPRECATED
    def __transform_preferred_to_numerical(self):

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
                self.do_label_encode_column(c)

    def do_label_encode_column(self, c):
        '''Manual implementation that supports unseen categories'''
        if self.verbose:
            print 'Label encoding:', c

        self.df_train[c] = self.df_train[c].astype(str)
        self.df_test[c] = self.df_test[c].astype(str)

        cats = np.asarray(self.schema['columns'][c]['categories'])

        train_tmp = self.df_train[c].values
        test_tmp = self.df_test[c].values

        for idx, cat in enumerate(cats):
            train_tmp[train_tmp == cat] = idx
            test_tmp[test_tmp == cat] = idx

        train_tmp[np.isin(train_tmp, cats)] = 999999
        test_tmp[np.isin(test_tmp, cats)] = 999999

        self.df_train[c] = train_tmp
        self.df_test[c] = test_tmp

        if self.df_train[c].nunique() == 1:
            print 'Label encoding of column', c, 'has only 1 value', self.df_train[c].iloc[0]
            return False

        self.schema['columns'][c] = {'type': 'NUMERIC'}

        return True

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
        if self.verbose:
            print 'Extracting singular value', val, 'from', col

        nonval = self.df_train[self.df_train[col] != val][col]

        if nonval.shape[0] == self.df_train.shape[0]:
            if self.verbose:
                print 'No rows are matching singular value.'
            return False

        # get best fit line for non zero elements
        z = np.polyfit(nonval, self.labels[nonval.index], 1)

        # where should the zero elements be to keep that line?
        vals = self.df_train[self.df_train[col] == val]
        avg0 = self.labels[vals.index].mean()
        x = (avg0 - z[1]) / z[0]

        self.df_train[col][self.df_train[col] == val] = x
        self.df_test[col][self.df_test[col] == val] = x

        return True

    def save(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)

        train = self.df_train.copy()
        train[self.idcol] = self.train_ids
        train[self.targetcol] = self.labels

        print 'Saving train data to ', folder + '/train.csv'
        train.to_csv(folder + '/train.csv', index=False)

        test = self.df_test.copy()
        test[self.idcol] = self.test_ids

        print 'Saving test data to ', folder + '/test.csv'
        test.to_csv(folder + '/test.csv', index=False)

        print 'Saving schema to ', folder + '/schema.yaml'
        with open(folder + '/schema.yaml', 'w') as outfile:
            yaml.dump(self.schema, outfile, default_flow_style=False)

    def do_replace_all_categoricals_with_mean(self):
        for c in self.categorical_columns():
            self.replace_categorical_with_mean(c)

    def replace_categorical_with_mean(self, c):
        if self.verbose:
            print 'Replacing', c, 'with mean...'

        for v in self.df_train[c].unique():
            matching = (self.df_train[c] == v)
            mean = self.labels[matching].mean()
            self.df_train[c][matching] = mean
            self.df_test[c][self.df_test[c] == v] = mean

        self.df_train[c] = self.df_train[c].astype(float)
        self.df_test[c] = self.df_test[c].astype(float)

        self.schema['columns'][c]['type'] = 'NUMERIC'

        return True

    def do_shuffle(self):
        print "Everyday I'm shuffling..."
        perm = np.random.permutation(self.df_train.index)
        self.df_train.reindex(perm)
        # if self.labels is not None:
        self.labels.reindex(perm)

    def train_test_split(self, test_size=.2):
        # X_train, X_test, y_train, y_test
        return train_test_split(self.df_train.values, self.labels.values, test_size=test_size)

    def copy(self):
        t2 = Training(self.df_train.copy(), self.df_test.copy(),
                      copy.deepcopy(self.schema))
        t2.labels = self.labels.copy()
        return t2

    def schemaget(self, data, var):
        if not var in data:
            return None
        return data[var]

    def duplicate_column(self, c, newcol):
        print 'Creating new column', newcol
        self.df_train[newcol] = self.df_train[c]
        if self.df_test is not None:
            self.df_test[newcol] = self.df_test[c]
        self.schema['columns'][newcol] = copy.deepcopy(
            self.schema['columns'][c])

    def do_explode_possible_types_columns(self):
        for c in self.numerical_columns():
            if self.schemaget(self.schema['columns'][c], 'possibly_categorical'):
                self.duplicate_column(c, c+'_categorical')
                self.convert_numerical_column_to_categorical(c+'_categorical')

        for c in self.categorical_columns():
            if self.schemaget(self.schema['columns'][c], 'possibly_numerical'):
                self.duplicate_column(c, c+'_numerical')
                self.convert_categorical_column_to_numerical(c+'_numerical')

    def convert_numerical_column_to_categorical(self, c):
        ''' Infers categories automatically form data '''
        train_categories = self.df_train[c].unique().tolist()
        test_categories = self.df_train[c].unique().tolist()
        categories = [str(x) for x in set(train_categories + test_categories)]
        self.convert_numerical_column_to_categorical_with_categories(
            c, categories)

    def convert_numerical_column_to_categorical_with_categories(self, c, categories):
        print 'Transform numerical column', c, 'to categorical'
        self.df_train[c] = self.df_train[c].astype(str)

        if self.df_test is not None:
            self.df_test[c] = self.df_test[c].astype(str)

        self.schema['columns'][c] = {
            'type': 'CATEGORICAL', 'categories': categories}

    def convert_categorical_column_to_numerical(self, c):
        print 'Transform categorical column', c, 'to numerical'
        self.df_train[c] = self.df_train[c].astype(float)

        if self.df_test is not None:
            self.df_test[c] = self.df_test[c].astype(float)

        self.schema['columns'][c] = {'type': 'NUMERIC'}

    def quantile_bin(self, col, nbins):

        self.df_train[col], bins = pd.qcut(
            self.df_train[col], nbins, duplicates='drop', retbins=True)

        if nbins != len(bins)-1:
            if self.verbose:
                print 'Asked for', nbins, 'bins but got', (len(bins)-1)
            return False

        # we want to adapt to the test frame
        bins[0] = self.df_test[col].min() - 1
        bins[-1] = self.df_test[col].max() + 1

        self.df_test[col] = pd.cut(self.df_test[col], bins=bins)

        # categorical to binaries
        self.df_train = pd.get_dummies(self.df_train, columns=[
                                       col], drop_first=True, prefix=col)
        self.df_test = pd.get_dummies(
            self.df_test, columns=[col], drop_first=True, prefix=col)

        return True

    def linearize_polynomial_fit(self, c, order=2, singurality=None):
        if self.verbose:
            print 'poly linearize', c
        nrows = self.df_train.shape[0]

        regulars = [True]*nrows
        test_regulars = [True]*(self.df_test.shape[0])

        if singurality is not None:
            regulars = self.df_train[c] != singurality
            test_regulars = self.df_test[c] != singurality

        has_train_singulars = (nrows - len(regulars) > 0)

        # curve fit
        x = self.df_train[c][regulars].values
        y = self.labels[regulars]
        orderedfit, coefs = generate_least_square_best_fit(x, y, order)

        self.df_train[c][regulars] = orderedfit(x)
        x_test = self.df_test[c][test_regulars].values
        self.df_test[c][test_regulars] = orderedfit(x_test)

        if has_train_singulars:
            singulars = [not s for s in regulars]
            test_singulars = [not s for s in test_regulars]

            smean = self.df_train[c][singulars].mean()

            coefs2 = np.flip(np.copy(coefs), 0)
            coefs2[-1] = coefs2[-1] - smean
            mean_target = np.roots(coefs2)[0]

            if not np.isreal(mean_target):
                if self.verbose:
                    print 'Could not find real root for polynomial fit'
                return False

            self.df_train[c][singulars] = mean_target
            self.df_test[c][test_singulars] = mean_target

        return True
