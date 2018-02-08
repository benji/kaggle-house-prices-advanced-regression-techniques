import math, json, sys, os
import pandas as pd
from os import path

from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, BayesianRidge, LassoLarsIC, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold, cross_val_score, train_test_split

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from utils.utils import *
from utils.kaggle import *
from utils.Training import *

df_train = pd.read_csv('newtrain.csv', index_col=0)
df_test = pd.read_csv('newtest.csv', index_col=0)
schema = json.loads(open('newschema.json', 'r').read())

feature_augmentation = False

if feature_augmentation:
    for c in schema['columns'].copy():
        if c in df_train.columns and schema['columns'][c]['type'] == 'NUMERIC':
            print 'adding square for', c
            newcol = 'squared_{}'.format(c)
            df_train[newcol] = df_train[c]**2
            df_test[newcol] = df_test[c]**2
            schema['columns'][newcol] = {'type': 'NUMERIC'}

t = Training(df_train, df_test, schema=schema)

t.dummify_at_init = True
t.dummify_drop_first = False
t.use_label_encoding = False

t.prepare()

t.sanity(False)


def test_model(t, model, name, scaler=False):
    print '====', name, '===='
    model0 = model

    if scaler:
        model = make_pipeline(RobustScaler(), model)

    accuracy = test_accuracy_for_model_using_kfolds(model, t.df_train,
                                                    t.labels)
    print 'Coefficient of determination:', accuracy

    print 'RMSE', test_accuracy_rmsle(model, t.df_train, t.labels)

    if hasattr(model0, 'coef_'):
        d = {'feature': t.df_train.columns, 'coef': model0.coef_}
        df = pd.DataFrame(data=d)
        idxs = np.array(df.coef.abs().sort_values(
            inplace=False, ascending=False).index[:5])
        print df.iloc[idxs]


test_model(t, LinearRegression(), 'OLS')
test_model(t, Lasso(alpha=0.0005, random_state=1), 'Lasso', scaler=True)
test_model(
    t,
    Ridge(
        alpha=1.0,
        copy_X=True,
        fit_intercept=True,
        max_iter=None,
        normalize=False,
        random_state=None,
        solver='auto',
        tol=0.001),
    'Ridge',
    scaler=True)
test_model(
    t,
    ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3),
    'Elastic Net',
    scaler=True)
test_model(
    t,
    KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5),
    'Kernel Ridge',
    scaler=False)

#generate_predictions(t.labels, t.df_train, t.df_test, t.test_ids)
#print "Predictions complete."