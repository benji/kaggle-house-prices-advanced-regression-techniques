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


t= training(exclude_outliers=False)
t.explode_columns_possibilities()
t2 = t.copy()

t.dummify_all_categoricals()
t.ready_for_takeoff()
#t.scale()
#t.shuffle()

t.compute_scores(model0, rmse)
print t.df_scores.head()
t2.df_scores = t.df_scores
t2.save('tmp',with_scores=True)