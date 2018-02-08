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

#t.df_train['SalePrice'] = np.exp(t.df_train['SalePrice'])


def numeric_feature_to_dual(series):
    train = pd.DataFrame(data={'num': series, 'bin': (series == 0) * 1})
    print train.head()
    return train

epochs=400

for col in t.df_train.columns:
#for col in ['TotalBsmtSF']:
    if t.schema['columns'][col]['type'] == 'NUMERIC':
        print '=====', col, '====='
        train = numeric_feature_to_dual(t.df_train[col])

        model = Sequential()
        model.add(Dense(2, input_dim=2, activation='linear'))
        #model.add(Dense(2, activation='linear'))
        model.add(Dense(1))

        model.summary()
        opt = optimizers.RMSprop(lr=1, rho=0.9, epsilon=None, decay=0.0)
        model.compile(loss='mean_squared_error', optimizer=opt)

        X_train, y_train = shuffle(train.values, t.labels.values)
        history = model.fit(
            X_train, y_train, epochs=epochs, batch_size=128, verbose=0)
        print 'loss $$', np.sqrt(history.history['loss'][-1])

        t.df_train[col] = model.predict(train.values)

        test = numeric_feature_to_dual(t.df_test[col])
        t.df_test[col] = model.predict(test.values)

t.df_train[t.idcol] = t.train_ids
t.df_train[t.targetcol] = t.labels
t.df_train.to_csv('newtrain.csv', index=False)
t.df_test[t.idcol] = t.test_ids
t.df_test.to_csv('newtest.csv', index=False)

#def build_model():
#    return model

#acc = keras_deep_test_accuracy_for_model_using_kfolds(
#    build_model, train, t.labels, n_splits=5,epochs=100)
#print 'Cross validation R2:', acc

#estimator = KerasRegressor(
#    build_fn=build_model, nb_epoch=epochs, batch_size=128, verbose=True)

#def test_accuracy_rmsle2(model, train, y, n_folds=5):
#    kf = KFold(
#        n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
#    score = np.sqrt(-cross_val_score(model, train.values, y.values, cv=kf))
#    return score.mean()