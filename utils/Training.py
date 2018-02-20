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

    def __init__(self, df_train=None, df_test=None, schema=None):
        if df_train is None:
            return

        self.df_train = df_train
        self.df_test = df_test
        self.schema = schema

        self.idcol = self.schema['id']
        self.targetcol = self.schema['target']

        self.train_ids = self.extract_column(self.df_train, self.idcol)
        self.test_ids = self.extract_column(self.df_test, self.idcol)

        self.labels = self.extract_column(self.df_train, self.targetcol)

        self.categoricals_as_string()

        self.health_check()

        self.verbose = True

    ######### TOOLS ##########

    def extract_column(self, df, col):
        if df is not None and col in df.columns:
            col_series = df[col].copy()
            df.drop(col, 1, inplace=True)
            return col_series
        if df is None:
            print 'None'
        elif col not in df.columns:
            print 'cols:', df.columns

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

    def schemaget(self, data, var):
        if not var in data:
            return None
        return data[var]

    def copy(self):
        t2 = Training()
        t2.df_train = self.df_train.copy()
        t2.df_test = self.df_test.copy()
        t2.schema = copy.deepcopy(self.schema)
        t2.labels = self.labels.copy()
        t2.train_ids = self.train_ids.copy()
        t2.test_ids = self.test_ids.copy()
        t2.idcol = self.idcol
        t2.targetcol = self.targetcol
        return t2

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

    ######### HEALTH ##########

    def verify_all_columns_are_numerical(self):
        cat_cols = self.categorical_columns()

        if len(cat_cols) == 0:
            return

        for c in cat_cols:
            print 'ERROR: Found categorical columns', c

        raise Exception('Found non numerical columns.')

    def health_check(self):
        print 'Check schema has all columns...'
        for c in self.df_train.columns:
            if c != self.idcol and c != self.targetcol:
                assert c in self.schema['columns']

        print 'Check test and train have same columns...'
        if self.df_test is not None:
            train_cols = self.df_train.columns.copy().tolist()
            # if self.targetcol in train_cols:
            #    train_cols.remove(self.targetcol)
            test_cols = self.df_test.columns.tolist()
            diff1_cols = [c for c in train_cols if c not in test_cols]
            diff2_cols = [c for c in test_cols if c not in train_cols]
            if len(diff1_cols) > 0:
                print 'ERROR: columns missing in test set:', diff1_cols
                sys.exit(1)
            if len(diff2_cols) > 0:
                print 'ERROR: columns missing in train set:', diff2_cols
                sys.exit(1)

        print 'Check ids sizes vs data sizes...'
        assert self.df_train.shape[0] == len(self.train_ids)
        if self.df_test is not None:
            assert self.df_test.shape[0] == len(self.test_ids)

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
            print self.df_train.isnull().sum().sort_values(
                ascending=False)[:6]
            return maxtrain

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

    def summary(self):
        s = self.df_train.shape
        print 'Training data has ', s[1], 'columns and', s[0], 'observations.'
        print 'Columns:', self.df_train.columns

    ##### TRANSFORMATIONS #####

    def train_test_split(self, test_size=.2):
        # X_train, X_test, y_train, y_test
        return train_test_split(self.df_train.values, self.labels.values, test_size=test_size)

    def explode_columns_possibilities(self):
        for c in self.numerical_columns():
            if self.schemaget(self.schema['columns'][c], 'possibly_categorical'):
                self.duplicate_column(c, c+'_categorical')
                self.convert_numerical_column_to_categorical(c+'_categorical')

        for c in self.categorical_columns():
            if self.schemaget(self.schema['columns'][c], 'possibly_numerical'):
                self.duplicate_column(c, c+'_numerical')
                self.convert_categorical_column_to_numerical(c+'_numerical')

    def df_transform(self, transform_fn):
        transform_fn(self.df_train)
        if self.df_test is not None:
            transform_fn(self.df_test)

    def shuffle(self):
        print "Everyday I'm shuffling..."
        perm = np.random.permutation(self.df_train.index)
        self.df_train.reindex(perm)
        # if self.labels is not None:
        self.labels.reindex(perm)

    ###### ADD / REMOVE COLUMNS / ROWS #####

    def duplicate_column(self, c, newcol):
        print 'Creating new column', newcol
        self.df_train[newcol] = self.df_train[c]
        if self.df_test is not None:
            self.df_test[newcol] = self.df_test[c]
        self.schema['columns'][newcol] = copy.deepcopy(
            self.schema['columns'][c])

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

    def retain_columns(self, columns):
        self.retain_columns_df(self.df_train, columns)
        if self.df_test is not None:
            self.retain_columns_df(self.df_test, columns)
        print 'Retained', len(self.df_train.columns), 'columns in dataframe'

    def drop_column(self, col):
        if col in self.df_train.columns:
            print 'Dropping', col
            self.df_train = self.df_train.drop([col], axis=1)
            self.df_test = self.df_test.drop([col], axis=1)

    def retain_columns_df(self, df, columns):
        for col in df.columns:
            if (not (col in columns)):
                df.drop(col, 1, inplace=True)

    def drop_row_by_id(self, rowid):
        idx = self.train_ids[self.train_ids == rowid].index[0]
        print 'dropping training row', rowid, 'at index', idx
        self.df_train.drop(idx, inplace=True)
        self.train_ids.drop(idx, inplace=True)
        self.labels.drop(idx, inplace=True)

    ####### MISSING VALUES #######

    def fillna_mean(self):
        self.df_train.fillna(self.df_train.mean(), inplace=True)
        self.df_test.fillna(self.df_train.mean(), inplace=True)

    def fillna_value(self, val):
        self.df_train.fillna(val, inplace=True)
        if self.df_test is not None:
            self.df_test.fillna(val, inplace=True)

    def fill_na_column(self, c, v):
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

    def replace_values(self, col, fromval, toval):
        self.df_train[col][self.df_train[col] == fromval] = toval
        self.df_test[col][self.df_test[col] == fromval] = toval

        if self.df_train[self.df_train[col] == fromval].shape[0] > 0:
            raise Exception('Failed to replace values')

    ##### SINGULARITIES #####

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

    def regularize_linear_numerical_singularity(self, col, val):
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

    ##### NORMALIZATION #####

    def scale(self):
        print 'Scaling features ...'
        columns = self.df_train.columns
        scaler = StandardScaler()
        scaler.fit(self.df_train[columns])
        self.df_train[columns] = scaler.transform(self.df_train[columns])
        self.df_test[columns] = scaler.transform(self.df_test[columns])

    def normalize_column_log1p(self, col):
        if col == self.targetcol:
            print 'Normalizing labels column', col
            self.labels = np.log1p(self.labels)
        else:
            print 'Normalizing training column', col
            self.df_train[col] = np.log1p(self.df_train[col])
            if self.df_test is not None:
                self.df_test[col] = np.log1p(self.df_test[col])

    def linearize_column_with_polynomial_x_transform(self, c, order=2, singurality=None):
        '''
        Tries to fit the function f(column)=target with a polynomial curve,
        then replaces the values with the curve's values.
        Can "fix" a meaningless/erroneous values by infering the columns value from their mean target value.
        '''
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

    ##### NUMERICAL TRANSFORMATIONS #####

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

    ##### COLUMN TYPES OPERATIONS #####

    def df_categoricals_as_string(self, df):
        for col in self.categorical_columns():
            nan_idx = df[col].isnull()
            df[col] = df[col].astype(str)
            df[col][nan_idx] = np.nan

    def categoricals_as_string(self):
        '''Preserves the nan values'''
        self.df_transform(self.df_categoricals_as_string)

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

    ########## CATEGORICAL -> NUMERICAL STRATEGIES ###########

    def dummify_all_categoricals(self):
        for c in self.categorical_columns():
            self.dummify_column(c)

    def dummify_column(self, c):
        if self.verbose:
            print 'Dummifying columns', c
        self.df_train, self.df_test = dummify_col_with_schema(
            c, self.schema, self.df_train, self.df_test)
        return True

    def label_encode_all_categoricals(self):
        for c in self.categorical_columns():
            self.label_encode_column(c)

    def label_encode_column(self, c, fail_on_unseen=True):
        '''
        Manual implementation that supports unseen categories.
        It's not that great of a technique.
        It's unlikely to find a linear fit here.
        The only advantage is that is keeps the categorical into a single feature.
        '''
        if self.verbose:
            print 'Label encoding:', c

        cats = np.asarray(self.schema['columns'][c]['categories'])

        for df in [self.df_train, self.df_test]:
            df[c] = df[c].astype(str)

            # this can be really bad
            unseen = df[c][~df[c].isin(cats)]
            if len(unseen) > 0 and fail_on_unseen:
                print 'ERROR: unseen values detected during label encoding for column', c
                print unseen[:5]
                sys.exit(1)
            unseen = 9999

            for idx, cat in enumerate(cats):
                df[c][df[c] == cat] = idx

        if self.df_train[c].nunique() == 1:
            print 'Label encoding of column', c, 'has only 1 value', self.df_train[c].iloc[0]
            return False

        self.schema['columns'][c] = {'type': 'NUMERIC'}

        return True

    def replace_all_categoricals_with_mean(self):
        for c in self.categorical_columns():
            self.replace_categorical_with_mean(c)

    def replace_categorical_with_mean(self, c):
        if self.verbose:
            print 'Replacing', c, 'with mean...'

        global_mean = self.labels.mean()

        _unique_train_values = self.df_train[c].unique()

        for v in _unique_train_values:
            matching = (self.df_train[c] == v)
            mean = self.labels[matching].mean()
            self.df_train[c][matching] = mean
            self.df_test[c][self.df_test[c] == v] = mean

        # we can't find the mean target for test values NOT in train set
        # so let's take global mean
        self.df_test[c][~ self.df_test[c].isin(
            _unique_train_values)] = global_mean

        self.df_train[c] = self.df_train[c].astype(float)
        self.df_test[c] = self.df_test[c].astype(float)

        self.schema['columns'][c]['type'] = 'NUMERIC'

        return True
