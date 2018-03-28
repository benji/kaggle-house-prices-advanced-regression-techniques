import pandas as pd
import numpy as np
import sys
import os
import operator
import yaml
import copy
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, RobustScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.neighbors import NearestNeighbors
from random import random
import statsmodels
import statsmodels.api as smapi
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns


from utils import *


class Training:

    def __init__(self, df_train=None, df_test=None, schema=None):
        self.verbose = True
        self.df_scores = None
        self.strict_check = True

        # private stuff
        self.y_restore_transforms = []
        self.target_scaler = None

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

        self.numericals_as_float()
        self.categoricals_as_string()

        self.health_check()

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

    def save(self, folder, index=False, with_scores=False):
        if not os.path.exists(folder):
            os.makedirs(folder)

        train = self.df_train.copy()
        train[self.idcol] = self.train_ids
        train[self.targetcol] = self.labels

        if with_scores:
            train['score'] = self.df_scores

        print 'Saving train data to ', folder + '/train.csv'
        train.to_csv(folder + '/train.csv', index=index)

        test = self.df_test.copy()
        test[self.idcol] = self.test_ids

        print 'Saving test data to ', folder + '/test.csv'
        test.to_csv(folder + '/test.csv', index=index)

        if self.df_scores is not None:
            scores = self.df_scores
            scores[self.idcol] = self.train_ids
            print 'Saving test data to ', folder + '/scores.csv'
            scores.to_csv(folder + '/scores.csv', index=index)

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

    def ready_for_takeoff(self):
        self.health_check()

        print 'Check for NAs'
        assert self.diagnose_nas() == 0

        self.assert_numerical(self.df_train)
        self.assert_numerical(self.df_test)

        if False:
            self.fail_on_value(np.inf)
            self.fail_on_value(np.nan)

            for c in self.get_columns():
                self.fail_if_found(self.df_train, self.df_train[c] > 1000)
                self.fail_if_found(self.df_train, self.df_train[c] < -1000)
                self.fail_if_found(self.df_test, self.df_train[c] > 1000)
                self.fail_if_found(self.df_test, self.df_train[c] < -1000)

            if np.isnan(np.sum(self.df_train.values)):
                raise Exception('Found NANs!')
            if np.isnan(np.sum(self.df_test.values)):
                raise Exception('Found NANs!')

        for df in [self.df_train, self.df_test]:
            if self.targetcol in df.columns:
                raise Exception('Target column found in dataframe')
            if self.idcol in df.columns:
                raise Exception('Target column found in dataframe')

    def hasnan(self, df):
        return np.isnan(np.sum(df.values))

    def fail_if_found(self, df, selected):
        found = self.df_train[selected]
        if len(found) > 0:
            print found.head()
            raise Exception('Found erroneous values')

    def assert_numerical(self, df):
        for c in df.columns:
            t = self.coltype(c)
            if t != 'NUMERIC':
                raise Exception('Column '+c+' is of schema type '+t)
            dt = df[c].dtype
            if not np.issubdtype(dt, np.number):
                raise Exception('Column '+c+' is of panda type '+str(dt))

    def fail_on_value(self, v):
        for c in self.get_columns():
            nvals = len(self.df_train[self.df_train[c] == v])
            if nvals > 0:
                raise Exception(
                    'Found', nvals, v, 'values in train data for column', c)
            nvals = len(self.df_test[self.df_test[c] == v])
            if nvals > 0:
                raise Exception(
                    'Found', nvals, v, 'values in test data for column', c)
        print 'No inf found.'

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

    def sanitize_column_name(self, c):
        '''
        Mostly for XGBoost:
        ValueError: feature_names may not contain [, ] or <
        '''
        c = c.replace('[', '_')
        c = c.replace(']', '_')
        c = c.replace('<', '_')
        return c

    def sanitize_column_names(self):
        newcols = [self.sanitize_column_name(c) for c in self.df_train.columns]
        self.df_train.columns = newcols
        self.df_test.columns = newcols

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
            elif self.schemaget(self.schema['columns'][c], 'meaningful_order'):
                self.duplicate_column(c, c+'_numerical')
                self.label_encode_column(c+'_numerical')

    def df_transform(self, transform_fn):
        transform_fn(self.df_train)
        if self.df_test is not None:
            transform_fn(self.df_test)

    def shuffle(self,seed=None):
        if self.verbose:
            print "Everyday I'm shuffling..."
        perm = np.random.permutation(self.df_train.index)
        np.random.RandomState(seed=seed).permutation(self.df_train.index)
        self.df_train = self.df_train.reindex(perm)
        self.train_ids = self.train_ids.reindex(perm)
        self.labels = self.labels.reindex(perm)

    ###### ADD / REMOVE COLUMNS / ROWS #####

    def duplicate_column(self, c, newcol):
        if self.verbose:
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
        if self.verbose:
            print 'Retained', len(
                self.df_train.columns), 'columns in dataframe'

    def drop_column(self, col):
        if col in self.df_train.columns:
            if self.verbose:
                print 'Dropping column', col
            self.df_train = self.df_train.drop([col], axis=1)
            self.df_test = self.df_test.drop([col], axis=1)

    def retain_columns_df(self, df, columns):
        for col in df.columns:
            if (not (col in columns)):
                df.drop(col, 1, inplace=True)

    def drop_row_by_id(self, rowid):
        idx = self.train_ids[self.train_ids == rowid].index[0]
        self.drop_row_by_index(idx)

    def drop_row_by_index(self, idx):
        indexes = self.df_train.index.values
        if idx in indexes:
            if self.verbose:
                print 'Dropping training row at index', idx
            self.df_train.drop(idx, inplace=True)
            self.train_ids.drop(idx, inplace=True)
            self.labels.drop(idx, inplace=True)

    def drop_rows_by_indexes(self, arr):
        for i in arr:
            self.drop_row_by_index(i)

    def remove_categoricals(self):
        for c in self.categorical_columns():
            self.drop_column(c)

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

    def scale(self, use_quantile_transformer=False):
        if self.verbose:
            print 'Scaling features ...'

        if use_quantile_transformer:
            scaler = QuantileTransformer(n_quantiles=10, random_state=0)
        else:
            #scaler = RobustScaler()
            scaler = StandardScaler()

        columns = self.df_train.columns
        scaler.fit(self.df_train[columns])

        self.df_train[columns] = scaler.transform(self.df_train[columns])
        self.df_test[columns] = scaler.transform(self.df_test[columns])

    def unscale_target(self,  values):
        if self.target_scaler is None:
            return values
        reshaped = np.reshape(values, (-1, 1))
        unscaled = self.target_scaler.inverse_transform(reshaped)
        return np.reshape(unscaled, (-1))

    def scale_target(self):
        self.target_scaler = StandardScaler()
        reshaped = np.reshape(self.labels.values, (-1, 1))
        self.target_scaler.fit(reshaped)
        scaled = self.target_scaler.transform(reshaped)
        reshaped = np.reshape(scaled, (-1))
        self.labels[:] = reshaped
        self.y_restore_transforms.append(
            lambda y: self.unscale_target(y))

    def untransform_target(self, predicted):
        for f in reversed(self.y_restore_transforms):
            predicted = f(predicted)
        return predicted

    def normalize_column_log1p(self, col):
        if col == self.targetcol:
            print 'Normalizing labels column', col
            self.labels = np.log1p(self.labels)
            self.y_restore_transforms.append(lambda y: np.expm1(y))
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
            singulars = self.df_train[c] == singurality
            if singulars.shape[0] == 0:
                return False
            regulars = self.df_train[c] != singurality
            test_regulars = self.df_test[c] != singurality

        has_train_singulars = (nrows - len(regulars) > 0)

        # curve fit
        if self.df_train[c][regulars].nunique() < 2:
            return False

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

    def pca(self, n_components):
        '''Helps decorrelating features'''
        pca = PCA(n_components=n_components)

        a = self.df_train.reset_index(drop=True)
        b = self.df_test.reset_index(drop=True)

        all_data = pd.concat([a, b], axis=0)

        if self.hasnan(all_data):
            raise Exception('found NANs')

        pca.fit(all_data.values)

        return pca.transform(self.df_train.values), pca.transform(self.df_test.values)

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

        self.convert_numerical_column_to_categorical(col)
        self.dummify_column(col)

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

    def df_numericals_as_string(self, df):
        for col in self.numerical_columns():
            nan_idx = df[col].isnull()
            df[col] = df[col].astype(float)
            df[col][nan_idx] = np.nan

    def numericals_as_float(self):
        '''Preserves the nan values'''
        self.df_transform(self.df_numericals_as_string)

    def convert_numerical_column_to_categorical(self, c):
        ''' Infers categories automatically form data '''
        train_categories = self.df_train[c].unique().tolist()
        test_categories = self.df_train[c].unique().tolist()
        categories = [str(x) for x in set(train_categories + test_categories)]
        self.convert_numerical_column_to_categorical_with_categories(
            c, categories)

    def convert_numerical_column_to_categorical_with_categories(self, c, categories):
        if self.verbose:
            print 'Transform numerical column', c, 'to categorical'
        self.df_train[c] = self.df_train[c].astype(str)

        if self.df_test is not None:
            self.df_test[c] = self.df_test[c].astype(str)

        self.schema['columns'][c] = {
            'type': 'CATEGORICAL', 'categories': categories}

    def convert_categorical_column_to_numerical(self, c):
        if self.verbose:
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
            c, self.schema, self.df_train, self.df_test, strict_check=self.strict_check)
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
            if len(unseen) > 0 and fail_on_unseen and self.strict_check:
                print 'ERROR: unseen values detected during label encoding for column', c
                print unseen[:5]
                sys.exit(1)
            unseen = 9999

            for idx, cat in enumerate(cats):
                df[c][df[c] == cat] = idx

        if self.df_train[c].nunique() == 1 and self.strict_check:
            print 'Label encoding of column', c, 'has only 1 value', self.df_train[c].iloc[0]
            return False

        self.convert_categorical_column_to_numerical(c)

        return True

    def replace_all_categoricals_with_mean(self):
        for c in self.categorical_columns():
            self.replace_categorical_with_mean(c)

    def replace_categorical_with_mean2(self, c):
        if self.verbose:
            print 'Replacing', c, 'with mean...'

        global_mean = self.labels.mean()

        _unique_train_values = self.df_train[c].unique()

        # we can't find the mean target for test values NOT in train set
        # so let's take global mean
        self.df_test[c][~ self.df_test[c].isin(
            _unique_train_values)] = global_mean

        for v in _unique_train_values:
            matching = (self.df_train[c] == v)
            mean = self.labels[matching].mean()
            self.df_train[c][matching] = mean
            self.df_test[c][self.df_test[c] == v] = mean

        self.df_train[c] = self.df_train[c].astype(float)
        self.df_test[c] = self.df_test[c].astype(float)

        self.schema['columns'][c]['type'] = 'NUMERIC'

        return True

    def replace_categorical_with_mean(self, c):
        return self.replace_categorical_with_labels_transform(c, 'mean', lambda ls: ls.mean())

    def replace_categorical_with_median(self, c):
        return self.replace_categorical_with_labels_transform(c, 'median', lambda ls: ls.median())

    def replace_categorical_with_labels_transform(self, c, tranform_name, labels_transform):
        if self.verbose:
            print 'Replacing', c, 'with ', tranform_name, '...'

        global_val = labels_transform(self.labels)

        _unique_train_values = self.df_train[c].unique()

        # we can't find the val target for test values NOT in train set
        # so let's take global val
        self.df_test[c][~ self.df_test[c].isin(
            _unique_train_values)] = global_val

        for v in _unique_train_values:
            matching = (self.df_train[c] == v)
            val = labels_transform(self.labels[matching])
            self.df_train[c][matching] = val
            self.df_test[c][self.df_test[c] == v] = val

        self.df_train[c] = self.df_train[c].astype(float)
        self.df_test[c] = self.df_test[c].astype(float)

        self.schema['columns'][c]['type'] = 'NUMERIC'

        return True

    ########## OUTLIERS TOOLBOX ###########

    def autoremove_ouliers2(self):
        print 'Starting outliers detection...'
        x = self.df_train.values
        y = self.labels.values
        regression = ols("data ~ x", data=dict(data=y, x=x)).fit()
        test = regression.outlier_test()
        print test
        print test.head()
        print test[test['bonf(p)'] < 0.5]
        sorted = test[test['bonf(p)'] < 0.5].sort_values('bonf(p)')
        print sorted.index

        for i in sorted.index:
            self.drop_row_by_index(i)

    def autoremove_ouliers(self):
        outlier_test = self.compute_outliers()
        bonf_test = outliers_test['bonf(p)']
        bonf_outliers = list(bonf_test[bonf_test < 1e-3].index)

        for i in bonf_outliers:
            self.drop_row_by_index(i)

    def compute_outliers(self):
        '''time consuming'''
        print 'Computing outliers...'
        x = self.df_train
        y = self.labels
        model = smapi.OLS(y, x)
        results = model.fit()
        outliers_test = results.outlier_test().sort_values('bonf(p)')
        return outliers_test

    def find_worst_predicted_points(self, model, score_fn):
        scores = {}
        for index, row in self.df_train.iterrows():
            y_pred = model.predict([row])
            y_true = np.array([self.labels[index]])
            score = score_fn(y_pred, y_true)
            #print 'index', index, 'pred',y_pred,'true',y_true,'score',score
            scores[index] = score

        return self.sort_dict_by_value(scores, reverse=True)

    def lasso_stats(self, lasso_model, print_n_first_important_cols=20, plot=False):
        col_to_coef = {}
        for i, col in enumerate(self.df_train.columns):
            col_to_coef[col] = np.abs(lasso_model.coef_[i])
        sorted_col_to_coef = self.sort_dict_by_value(
            col_to_coef, reverse=True)[:print_n_first_important_cols]

        for item in sorted_col_to_coef:
            col = item[0]
            s = len(self.df_train) - self.find_most_popular_value_count(col)
            print 'Lasso coef', item[0], item[1], '('+str(s)+')'

        print 'Lasso intercept', lasso_model.intercept_
        print 'Lasso iterations', lasso_model.n_iter_

        if plot:
            self.barplot_dict(sorted_col_to_coef)

    def barplot_dict(self, thedict):
        print thedict[:10]
        sns.barplot(x=['_'+str(i[0]) for i in thedict], y=[i[1]
                                                           for i in thedict])
        plt.show()

    def sort_dict_by_value(self, thedict, reverse=False):
        return sorted(thedict.items(), key=lambda x: x[1], reverse=reverse)

    def find_nearest_train_neighbor_indexes(self, row, dim_scaling_ratios=None, n_neighbors=5, ignore_first=False):
        neigh = NearestNeighbors()
        X = self.df_train.values

        if dim_scaling_ratios is not None:
            X = X*dim_scaling_ratios
            row = row*dim_scaling_ratios

        neigh.fit(self.df_train.values)

        if ignore_first:
            n_neighbors += 1

        neigh_distances, neigh_indexes = neigh.kneighbors(
            [row], n_neighbors=n_neighbors, return_distance=True)

        neigh_indexes = neigh_indexes[0]
        #neigh_distances = neigh_distances[0]
        #print neigh_distances

        return neigh_indexes[1:]

    def train_id_to_index(self, id):
        return self.train_ids.tolist().index(id)

    def train_index_to_id(self, index):
        return self.train_ids[index]

    def report_out_of_range_series(self, c, oor, expected):
        if oor.shape[0] > 0:
            print 'Out of range rows found in test set for column', c, 'expected:', expected
            for idx, row in oor.iterrows():
                print 'Index:', idx, 'value:', row[c]

    def find_out_of_range_test_data(self):
        for c in self.categorical_columns():
            values = self.df_train[c].unique()
            out_of_range_test = self.df_test[~self.df_test[c].isin(values)]
            self.report_out_of_range_series(c, out_of_range_test, values)

        for c in self.numerical_columns():
            min = self.df_train[c].min()
            out_of_range_test1 = self.df_test[self.df_test[c] < min]
            self.report_out_of_range_series(
                c, out_of_range_test1, 'min:'+str(min))

            max = self.df_train[c].max()
            out_of_range_test2 = self.df_test[self.df_test[c] > max]
            self.report_out_of_range_series(
                c, out_of_range_test2, 'max:'+str(max))

    def compute_scores(self, model0, scoreFn):
        model = clone(model0)
        model.fit(self.df_train.values, self.labels.values)

        predicted = model.predict(self.df_train.values)
        scores = [scoreFn(a, b) for a, b in zip(predicted, self.labels.values)]

        self.df_scores = pd.DataFrame(
            data={'score': scores}, index=self.df_train.index)
        return self.df_scores

    def print_row_stats_by_index(self, idx, max_categorical_pct=5, max_numerical_pct=5):
        row = self.df_train.loc[idx]

        print 'row idx', idx, 'id', self.train_ids.loc[idx], 'stats for NUMERIC columns:'
        col_to_stat = {}
        for c in self.numerical_columns():
            val = row[c]
            n_matches = self.df_train[self.df_train[c] == val].shape[0]
            pct = n_matches * 100 / self.df_train.shape[0]
            if pct > max_numerical_pct:
                pass
            else:
                n_above = self.df_train[self.df_train[c] >= val].shape[0]
                pct = n_above * 100 / self.df_train.shape[0]
                if pct <= max_numerical_pct:
                    print 'Column', c, '=', val, 'is in the top', pct, '% of the train set (', n_above, ')'
                    pass

                n_below = self.df_train[self.df_train[c] <= val].shape[0]
                pct = n_below * 100 / self.df_train.shape[0]
                if pct <= max_numerical_pct:
                    print 'Column', c, '=', val, 'is in the bottom', pct, '% of the train set (', n_below, ')'
                    pass

        for item in self.sort_dict_by_value(col_to_stat):
            print 'Column', item[0], '=', row[item[0]]

        print 'row idx', idx, 'stats for CATEGORICAL columns:'
        col_to_stat = {}
        for c in self.categorical_columns():
            matches = self.df_train[self.df_train[c] == row[c]]
            col_to_stat[c] = matches.shape[0]

        for item in self.sort_dict_by_value(col_to_stat):
            pct = item[1] * 100 / self.df_train.shape[0]
            if pct > max_categorical_pct:
                break
            print 'Column', item[0], '=', row[
                item[0]], 'matches', pct, '% of the train set (', item[1], ')'

    def find_most_popular_value_count(self, col):
        cmax = 0
        for v in self.df_train[col].unique():
            c = len(self.df_train[self.df_train[col] == v])
            if c > cmax:
                cmax = c
        return cmax
