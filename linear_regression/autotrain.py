import pandas as pd
import math, json, sys
from tqdm import tqdm
import numpy as np
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from utils.utils import *
from utils.Training import *
from utils.kaggle import *

t = training()

t.dummify_at_init = False
t.dummify_drop_first = False
t.use_label_encoding = False

t.prepare()

use_runtime_dummies = True

available_columns = list(t.df_train.columns.values)
validated_columns = []
best_error = 999
making_progress = True
df_testcols = pd.DataFrame(columns=['ColName', 'RMSE'])


def new_model():
    return Lasso(alpha=0.0005, random_state=1)


while (making_progress):
    df_testcols = df_testcols[0:0]  # remove all

    for test_column in tqdm(available_columns):
        #print 'Testing column', test_column
        test_columns = validated_columns + [test_column]

        df_train = t.df_train[test_columns].copy()

        if use_runtime_dummies:
            df_train, _ = t.do_dummify(df_train, None, False)

        rmse = test_accuracy_rmsle(new_model(), df_train, t.labels)

        df_testcols.loc[len(df_testcols)] = [test_column, rmse]

    min_error = df_testcols['RMSE'].min()
    best_column = df_testcols['ColName'][df_testcols['RMSE'] ==
                                         min_error].min()[0]

    ranked_test_columns = df_testcols.sort_values('RMSE', ascending=True)
    #print ranked_test_columns.head()
    best_column = ranked_test_columns.iloc[0, 0]

    diff_error = min_error - best_error
    print 'Found best column', best_column, 'with error', min_error, '(diff=', diff_error, ')'

    if (diff_error < 0):
        best_error = min_error
        validated_columns.append(best_column)
        available_columns.remove(best_column)
        print 'Retained columns:', validated_columns
    else:
        making_progress = False

print 'Final columns', validated_columns
print 'Final error', best_error

t.retain_columns(validated_columns)

if use_runtime_dummies:
    df_train, df_test = t.do_dummify(t.df_train, t.df_test, False)
else:
    df_train, df_test = t.df_train, t.df_test

generate_predictions(t.labels, df_train, df_test, t.test_ids)
print "Predictions complete."