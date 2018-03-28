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

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from utils.utils import *
from utils.kaggle import *
from utils.Training import *
from utils.Variants import *

np.random.seed(int(time.time()))


def seed():
    return np.random.randint(2**32-1)


t = training()
t.explode_columns_possibilities()
variants = Variants(t, verbose=False)

average_over_n = 500

for technique in [CATEGORICAL_ONEHOT, CATEGORICAL_TARGET_MEAN, CATEGORICAL_LABEL_ENCODING]:
    print '===================================', technique
    t2 = t.copy()
    t2.verbose = False

    if False:
        c = 'Neighborhood'
        t2.retain_columns([c])
        variants.apply_variants(t2, [[c, technique]])
    else:
        cols = t.categorical_columns()
        t2.retain_columns(cols)
        variants.apply_variants(t2,  [[c, technique] for c in cols])

    # t2.ready_for_takeoff()
    t2.scale()
    t2.shuffle()
    t2.remove_columns_with_unique_value()
    # t2.sanity()
    # t2.summary()
    # t2.save('tmp')

    scores = []
    for i in range(average_over_n):
        model = Lasso(alpha=0.0005, fit_intercept=True,
                      random_state=seed(), warm_start=True)
        scores.append(custom_score_using_kfolds(model.fit, model.predict, rmse,
                                                t2.df_train.values, t2.labels.values, n_splits=3, doShuffle=True, seed=seed(),average_over_n=average_over_n))
    print np.array(scores).mean()
