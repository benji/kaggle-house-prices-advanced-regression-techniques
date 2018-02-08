import math, json, sys, os
import pandas as pd
from os import path

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from keras import optimizers

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from utils.utils import *
from utils.kaggle import *
from utils.Training import *

t = training()
t.logify_columns = []
t.prepare()

#for col in t.df_train.columns:
for col in []:
    if t.schema['columns'][col]['type'] == 'NUMERIC':
        print '=====', col, '====='
        newcol = col + '_zero'

        t.df_train[newcol] = (t.df_train[col] == 0) * 1
        t.df_test[newcol] = (t.df_test[col] == 0) * 1
        t.schema['columns'][newcol] = {'type': 'NUMERIC'}

t.df_train[t.idcol] = t.train_ids
t.df_train[t.targetcol] = t.labels
t.df_train.to_csv('newtrain.csv', index=False)
t.df_test[t.idcol] = t.test_ids
t.df_test.to_csv('newtest.csv', index=False)

with open('newschema.json', 'w') as outfile:
    json.dump(t.schema, outfile, indent=4)