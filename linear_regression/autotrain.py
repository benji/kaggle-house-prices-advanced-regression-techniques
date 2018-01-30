import pandas as pd
import math, json, sys
from tqdm import tqdm
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from utils.utils import *
from utils.Training import *

pd.options.mode.chained_assignment = None  # default='warn'

schema = json.loads(open('../schema2.json', 'r').read())
df_train = pd.read_csv('../data/train.csv')  #, dtype=get_pandas_types(schema))
df_test = pd.read_csv('../data/test.csv')  #, dtype=get_pandas_types(schema))

# special prep
df_train['DateSold'] = df_train['YrSold'] + df_train['MoSold'] / 12.0
df_test['DateSold'] = df_test['YrSold'] + df_test['MoSold'] / 12.0

df_train[
    'TotalSF'] = df_train['TotalBsmtSF'] + df_train['1stFlrSF'] + df_train['2ndFlrSF']
df_test[
    'TotalSF'] = df_test['TotalBsmtSF'] + df_test['1stFlrSF'] + df_test['2ndFlrSF']

# -4%!
#outliers_LotArea = df_train['LotArea'][df_train['LotArea'] > 100000]
#print outliers_LotArea
#df_train = df_train.drop(outliers_LotArea.index)

t = Training(df_train, df_test, schema=schema)

t.columns_na = {
    'PoolQC': 'None',
    'Alley': 'None',
    'Fence': 'None',
    'FireplaceQu': 'None',
    'GarageType': 'None',
    'GarageFinish': 'None',
    'GarageQual': 'None',
    'GarageCond': 'None',
    'GarageYrBlt': 0,
    'GarageArea': 0,
    'GarageCars': 0,
    'BsmtQual': 'None',
    'BsmtCond': 'None',
    'BsmtExposure': 'None',
    'BsmtFinType1': 'None',
    'BsmtFinType2': 'None',
    'MasVnrType': 'None',
    'MasVnrArea': 0,
    'MSZoning': 'RL',
    'Functional': 'Typ',
    'Electrical': 'SBrkr',
    'KitchenQual': 'TA',
    'Exterior1st': 'VinylSd',
    'Exterior2nd': 'VinylSd',
    'SaleType': 'WD',
    'MSSubClass': 'None',
    'MiscFeature': 'None'
}

t.replace_values.append(['Neighborhood', 'NAmes', 'NWAmes'])
t.replace_values.append(['BldgType', '2fmCon', '2FmCon'])
t.replace_values.append(['BldgType', 'Duplex', 'Duplx'])
t.replace_values.append(['BldgType', 'Twnhs', 'TwnhsE'])
t.replace_values.append(['Exterior2nd', 'Brk Cmn', 'BrkComm'])
t.replace_values.append(['Exterior2nd', 'CmentBd', 'CemntBd'])
t.replace_values.append(['Exterior2nd', 'Wd Shng', 'WdShing'])

t.separate_out_value('PoolArea', 0, 'NoPool')
t.logify_columns.append('SalePrice')  # +4%!
t.logify_columns.extend(('LotArea', 'GrLivArea', '1stFlrSF'))  # +0.4%!
t.fill_na_mean = False
t.remove_outliers.extend((524, 1299))  # +3%!
t.dummify_at_init = False
t.dummify_drop_first = False
t.use_label_encoding = True
#t.use_dummies_for_specific_columns = ['Neighborhood']

use_runtime_dummies = False

mssc_cats = schema['columns']['MSSubClass']['categories']
for c in mssc_cats:
    t.replace_values.append(['MSSubClass', int(c), c])

t.drop_columns = ['Utilities']  #,'MSSubClass']

t.prepare()

available_columns = list(t.df_train.columns.values)

n_passes = 10
validated_columns = []

best_accuracy = 0
making_progress = True
df_testcols = pd.DataFrame(columns=['ColName', 'Accuracy'])

while (making_progress):
    df_testcols = df_testcols[0:0]  # remove all

    for test_column in tqdm(available_columns):
        #print 'Testing column', test_column
        test_columns = validated_columns + [test_column]

        #print 'before',t.df_train.columns
        df_train = t.df_train[test_columns].copy()
        #print 'retaining',test_columns
        #print 'got',df_train.columns

        #df_train, _ = t.do_dummify(False)
        accuracy = test_accuracy(df_train, t.labels, passes=n_passes) * 100
        #print accuracy

        #print 'Accuracy for column', test_column, ':', accuracy
        df_testcols.loc[len(df_testcols)] = [test_column, accuracy]

    max_accuracy = df_testcols['Accuracy'].max()
    best_column = df_testcols['ColName'][df_testcols['Accuracy'] ==
                                         max_accuracy].max()[0]

    ranked_test_columns = df_testcols.sort_values('Accuracy', ascending=False)
    #print ranked_test_columns.head()
    best_column = ranked_test_columns.iloc[0, 0]

    diff_accuracy = max_accuracy - best_accuracy
    print 'Found best column', best_column, 'with accuracy', max_accuracy, '(diff=', diff_accuracy, '%)'

    if (diff_accuracy > 0):
        best_accuracy = max_accuracy
        validated_columns.append(best_column)
        available_columns.remove(best_column)
        print 'Retained columns:', validated_columns
    else:
        making_progress = False

print 'Final columns', validated_columns
print 'Final accuracy', best_accuracy

t.retain_columns(validated_columns)

df_train, df_test = dummify_with_schema(t.schema, t.df_train, t.df_test)

generate_predictions(t.labels, df_train, df_test, t.test_ids)
print "Predictions complete."