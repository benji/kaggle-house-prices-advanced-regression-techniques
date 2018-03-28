import math
import json
import sys
import os
import pandas as pd
from os import path
import yaml

from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, BayesianRidge, LassoLarsIC, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.base import clone

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from utils.utils import *
from utils.kaggle import *
from utils.Training import *
from utils.Variants import *

model0 = Lasso(alpha=0.0005, fit_intercept=True,
               random_state=seed(), warm_start=False, max_iter=10000)


t2 = training(exclude_outliers=False)
t2.explode_columns_possibilities()
t2.dummify_all_categoricals()


t2.ready_for_takeoff()
t2.scale()

t2.shuffle()

saleprice_model = clone(model0)
saleprice_model.fit(t2.df_train.values, t2.labels.values)

scores_map = t2.find_worst_predicted_points(saleprice_model, rmse)

t = training(exclude_outliers=False)


def find_rank_of_index(idx):
    i = 0
    for item in scores_map:
        if item[0] == idx:
            return i
        i += 1

def report_out_of_range_series( c, oor, expected):
    if oor.shape[0] > 0:
        print 'Out of range rows found in test set for column', c, 'expected:', expected
        for idx, row in oor.iterrows():
            print 'Index:', idx, 'value:', row[c],'score rank:',find_rank_of_index(idx)

for c in t.categorical_columns():
    values = t.df_train[c].unique()
    out_of_range_test = t.df_test[~t.df_test[c].isin(values)]
    report_out_of_range_series(c, out_of_range_test, values)

for c in t.numerical_columns():
    min = t.df_train[c].min()
    out_of_range_test1 = t.df_test[t.df_test[c] < min]
    report_out_of_range_series(c, out_of_range_test1, 'min:'+str(min))

    max = t.df_train[c].max()
    out_of_range_test2 = t.df_test[t.df_test[c] > max]
    report_out_of_range_series(c, out_of_range_test2, 'max:'+str(max))

# t.explode_columns_possibilities()
