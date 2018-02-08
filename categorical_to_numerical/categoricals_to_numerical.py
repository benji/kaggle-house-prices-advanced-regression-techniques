import math, json, sys, os, json
import pandas as pd
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from utils.utils import *
from utils.kaggle import *
from utils.Training import *

df_train = pd.read_csv('../data/train.csv')
df_test = pd.read_csv('../data/test.csv')
schema = json.loads(open('../schema2.json', 'r').read())


def get_training(c, use_dummies, use_labelencoding):
    t = training()
    t.train_columns = [c]

    t.dummify_at_init = use_dummies
    t.dummify_drop_first = False
    t.use_label_encoding = use_labelencoding

    t.prepare()

    return t


def test_acc(c, use_dummies, use_labelencoding):
    print '=======', c, use_dummies, use_labelencoding, '========'

    t = get_training(c, use_dummies, use_labelencoding)

    if t.df_train.shape[1] > 0 and t.sanity(False):
        accuracy = test_accuracy_kfolds(t.df_train, t.labels)
        print 'COD', c, use_dummies, use_labelencoding, accuracy
        return accuracy
    else:
        return -1


t0 = training()
t0.prepare()

for c in schema['columns']:#['Neighborhood']:  #
    if schema['columns'][c]['type'] == 'NUMERIC':
        continue

    acc1 = test_acc(c, True, False)
    acc2 = test_acc(c, False, True)

    if acc1 < 0 and acc2 < 0:
        t0.do_drop_column(c)
        continue

    # Prediction using only that column

    use_dummies_for_pred = acc1 > acc2
    t = get_training(c, use_dummies_for_pred, not use_dummies_for_pred)

    X = np.array(t.df_train)
    X_predict = np.array(t.df_test)

    scaler = StandardScaler()
    scaler.fit(t.df_train)
    X = scaler.transform(X)
    X_predict = scaler.transform(X_predict)
    y = np.array(t.labels)

    # fit shuffle data
    ar = np.arange(len(X))
    clf.fit(X[ar], y[ar])

    # predict
    y_p_train = clf.predict(X)
    y_p_test = clf.predict(X_predict)

    t0.df_train[c] = y_p_train
    t0.df_test[c] = y_p_test
    t0.schema['columns'][c]['type'] = 'NUMERIC'

train = t0.df_train
train[t0.idcol] = t0.train_ids
train[t0.targetcol] = t0.labels

test = t0.df_test
test[t0.idcol] = t0.test_ids

train.to_csv('newtrain.csv',index=False)
test.to_csv('newtest.csv',index=False)

with open('newschema.json', 'w') as outfile:
    json.dump(t0.schema, outfile, indent=4)

print 'Done.'