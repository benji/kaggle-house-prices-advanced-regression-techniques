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
t.dummify_at_init = False
t.dummify_drop_first = False
t.use_label_encoding = False

t.prepare()

use_runtime_dummies = True
n_passes = 10

available_columns = list(t.df_train.columns.values)
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

        if use_runtime_dummies:
            df_train, _ = t.do_dummify(df_train, None, False)
        #print df_train.columns
        #print df_train.head()

        #accuracy = test_accuracy(df_train, t.labels, passes=n_passes) * 100
        accuracy = test_accuracy_kfolds(df_train, t.labels)

        #print 'Accuracy for column', test_column, ':', accuracy
        df_testcols.loc[len(df_testcols)] = [test_column, accuracy]

    max_accuracy = df_testcols['Accuracy'].max()
    best_column = df_testcols['ColName'][df_testcols['Accuracy'] ==
                                         max_accuracy].max()[0]

    ranked_test_columns = df_testcols.sort_values('Accuracy', ascending=False)
    #print ranked_test_columns.head()
    best_column = ranked_test_columns.iloc[0, 0]

    diff_accuracy = max_accuracy - best_accuracy
    print 'Found best column', best_column, 'with accuracy', max_accuracy, '(diff=', diff_accuracy, ')'

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

if use_runtime_dummies:
    df_train, df_test = t.do_dummify(t.df_train, t.df_test, False)
else:
    df_train, df_test = t.df_train, t.df_test

generate_predictions(t.labels, df_train, df_test, t.test_ids)
print "Predictions complete."