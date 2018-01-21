import pandas as pd
import math, json, sys
from tqdm import tqdm
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib import style

from train import test_accuracy, generate_predictions, retain_columns, prepare_data

pd.options.mode.chained_assignment = None  # default='warn'
style.use('ggplot')

df_train0 = pd.read_csv('train_transformed.csv')
df_test0 = pd.read_csv('test_transformed.csv')

y, df_train0, df_test0, test_ids = prepare_data(df_train0, df_test0)

print len(df_train0.columns), 'columns in dataframe'

available_columns = list(df_train0.columns.values)

n_passes = 100
validated_columns = []
best_accuracy = 0
making_progress = True
df_testcols = pd.DataFrame(columns=['ColName', 'Accuracy'])

while (making_progress):
    df_testcols = df_testcols[0:0]  # remove all

    for test_column in tqdm(available_columns):
        #print 'Testing column', test_column
        test_columns = validated_columns + [test_column]

        df_train, _ = retain_columns(test_columns, df_train0)

        accuracy = test_accuracy(df_train, y, passes=n_passes) * 100

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

df_train, df_test = retain_columns(validated_columns, df_train0, df_test0)

generate_predictions(y, df_train0, df_test0, test_ids)
print "Predictions complete."