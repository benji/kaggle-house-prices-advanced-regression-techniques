import pandas as pd
import math, json, sys, os
from tqdm import tqdm
import numpy as np
import keras
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

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from utils.utils import *
from utils.Training import *
from utils.kaggle import *
from deep_models import *

pd.options.mode.chained_assignment = None  # default='warn'

use_log_label = False
hm_manual_validation = 130

df_train = pd.read_csv('../categorical_to_numerical/newtrain.csv', index_col=0)
df_test = pd.read_csv('../categorical_to_numerical/newtest.csv', index_col=0)
schema = json.loads(
    open('../categorical_to_numerical/newschema.json', 'r').read())

t = Training(df_train, df_test, schema=schema)

t.dummify_at_init = True
t.dummify_drop_first = False
t.use_label_encoding = False
t.prepare()
t.labels = np.exp(t.labels)

#t.df_train.to_csv('temp.csv')

#t.df_train = pd.read_csv('../neural_network.bak/deep_train.csv', index_col=0)
#t.labels = pd.read_csv('../neural_network.bak/deep_labels.csv', header=None)

ncols = len(t.df_train.columns)
print 'columns: ', ncols

seed = 7
np.random.seed(seed)

X_train, y_train = X, y = shuffle(t.df_train.values, t.labels.values)

if hm_manual_validation > 0:
    X_train = X_train[:-hm_manual_validation]
    y_train = y_train[:-hm_manual_validation]

modelFn = model1
epochs = 150

#modelFn=model2
#epochs = 30


def build_model():
    return modelFn(ncols)


#acc = keras_deep_test_accuracy_for_model_using_kfolds(
#    build_model, t.df_train, t.labels, n_splits=4,epochs=epochs)
#print 'Cross validation R2:', acc

estimator = KerasRegressor(
    build_fn=build_model, nb_epoch=epochs, batch_size=128, verbose=True)

def test_accuracy_rmsle2(model, train, y, n_folds=5):
    kf = KFold(
        n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    score = np.sqrt(-cross_val_score(model, train.values, y.values, cv=kf))
    return score.mean()


print 'RMSE', test_accuracy_rmsle2(estimator, t.df_train, t.labels)


if False:
    model=None
    prediction = model.predict(t.df_test.values)

    df_predicted = pd.DataFrame(columns=['Id', 'SalePrice'])
    df_predicted['Id'] = t.test_ids
    df_predicted.set_index('Id')
    df_predicted['SalePrice'] = np.exp(prediction)
    df_predicted.to_csv('predicted.csv', sep=',', index=False)

    print 'Predictions done.'   