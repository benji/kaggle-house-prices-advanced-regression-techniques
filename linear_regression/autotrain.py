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

df_train = pd.read_csv('../data/train.csv')
df_test = pd.read_csv('../data/test.csv')
schema = json.loads(open('../schema.json', 'r').read())

# special prep
df_train['DateSold'] = df_train['YrSold'] + df_train['MoSold'] / 12.0
df_test['DateSold'] = df_test['YrSold'] + df_test['MoSold'] / 12.0

# -4%!
#outliers_LotArea = df_train['LotArea'][df_train['LotArea'] > 100000]
#print outliers_LotArea
#df_train = df_train.drop(outliers_LotArea.index)

t = Training(df_train, df_test, schema=schema)
t.separate_out_value('PoolArea', 0, 'NoPool')
t.logify_columns.append('SalePrice')  # +4%!
t.logify_columns.extend(('LotArea', 'GrLivArea', '1stFlrSF'))  # +0.4%!
t.fill_na_mean = True
t.remove_outliers.extend((524, 1299))  # +3%!
t.dummify_at_init = False
t.dummify_drop_first = False

t.prepare()

available_columns = list(t.df_train.columns.values)

n_passes = 10
validated_columns = [ ]

best_accuracy = 0
making_progress = True
df_testcols = pd.DataFrame(columns=['ColName', 'Accuracy'])

while (making_progress):
    df_testcols = df_testcols[0:0]  # remove all

    for test_column in tqdm(available_columns):
        #print 'Testing column', test_column
        test_columns = validated_columns + [test_column]

        df_train = t.df_train[test_columns].copy()

        df_train, _ = dummify_with_schema(schema,df_train)
        accuracy = test_accuracy(df_train, t.labels, passes=n_passes) * 100

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

generate_predictions(
    t.labels, df_train, df_test, t.test_ids)
print "Predictions complete."