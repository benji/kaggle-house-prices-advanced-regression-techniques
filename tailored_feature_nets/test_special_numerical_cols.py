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

t = training()

t.dummify_at_init = False
t.dummify_drop_first = False
t.use_label_encoding = False

t.prepare()

#t.sanity()

best_rmse = 0.110000260712
singularized_columns = []
making_progress = True
round = 0

while making_progress:
    round += 1
    print '#########################', round, '#######################'
    round_best_rmse = best_rmse
    round_best_col = None

    for col in t.df_train.columns:
        if t.schema['columns'][col]['type'] == 'NUMERIC' and not col in singularized_columns:
            print '================', col, '================'
            print t.schema['columns'][col]['type']
            print singularized_columns

            t2 = training()

            t2.dummify_at_init = True
            t2.dummify_drop_first = False
            t2.use_label_encoding = False

            t2.prepare()
            t2.numerical_singularities(col, 0)
            for c in singularized_columns:
                t2.numerical_singularities(c, 0)

            model0 = Lasso(alpha=0.0005, random_state=1)
            model = make_pipeline(RobustScaler(), model0)

            #accuracy = test_accuracy_for_model_using_kfolds(
            #    model, t2.df_train, t2.labels, scale=False)
            #print 'Coefficient of determination:', accuracy

            rmse = test_accuracy_rmsle(model, t2.df_train, t2.labels)

            print 'RMSE', rmse

            if rmse < round_best_rmse:
                print 'good col', col, 'diff', (round_best_rmse - rmse)
                round_best_col = col
                round_best_rmse = rmse

    if round_best_col is not None:
        print 'Best round col', round_best_col, 'diff', (best_rmse - round_best_rmse)
        singularized_columns.append(round_best_col)
        best_rmse = round_best_rmse
    else:
        making_progress = False

print 'Finished with singularized columns', singularized_columns
print 'Final RMSE', best_rmse
print 'Done'

#accuracy = test_accuracy_kfolds(t.df_train, t.labels)
#print 'Coefficient of determination:', accuracy

#generate_predictions(t.labels, t.df_train, t.df_test, t.test_ids)
#print "Predictions complete."