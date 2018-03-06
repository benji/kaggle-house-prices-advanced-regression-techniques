import time
import sys
from os import path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.base import clone

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from utils.utils import *
from utils.kaggle import *
from utils.Training import *
from utils.Variants import *
from utils.score_stats import *

np.random.seed(int(time.time()))


def seed():
    return np.random.randint(2**32-1)


t = training()
t.explode_columns_possibilities()

fit_intercept = True

if False:
    train_variants = [['TotalSF', 'linearize_order_2']]
    train_cols = [v[0] for v in train_variants]
    t.retain_columns(train_cols)
    variants = Variants(t, verbose=True)
    variants.apply_variants(t, train_variants)
else:
    t.dummify_all_categoricals()
    # t.label_encode_all_categoricals()

t.ready_for_takeoff()
t.scale()
if not fit_intercept:
    t.scale_target()
t.shuffle()
t.remove_columns_with_unique_value()
t.sanity()
t.summary()

X, X_test = t.df_train.values, t.df_test.values
y = t.labels.values

kfolds_splots = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
test_ratios = []  # .5, .6, .7, .8
strategies_max_evaluation_time = 60
display_every = 50

model0 = Lasso(alpha=0.0005, tol=0.000001, copy_X=True, selection='random',
               normalize=True, random_state=seed(), warm_start=False, fit_intercept=True)


def produce_score_fn(model, strategy):
    if strategy['type'] == 'test_ratio':
        def f():
            return score_using_test_ratio(model.fit, model.predict, rmse, X, y, test_ratio=strategy['test_ratio'], shuffle=True)
        return f
    elif strategy['type'] == 'kfolds':
        def f():
            return custom_score_using_kfolds(model.fit, model.predict, rmse, X, y, n_splits=strategy['n_splits'], doShuffle=True, seed=seed())
        return f


strategies = []
# Bench time is way higher and std way higher than kfolds scoring
for test_ratio in test_ratios:
    strategies.append({
        'type': 'test_ratio',
        'test_ratio': test_ratio,
        'name': 'test_ratio '+str(test_ratio),
        'plot_marker': '+'
    })
for kfold_n_splits in kfolds_splots:
    strategies.append({
        'type': 'kfolds',
        'n_splits': kfold_n_splits,
        'name': 'kfolds '+str(kfold_n_splits),
        'plot_marker': 'o'
    })

all_score_stats = []
for strategy in strategies:
    print 'Testing scoring strategy:', strategy
    model = clone(model0)

    score_fn = produce_score_fn(model, strategy)
    score_stats = ScoreStats(strategy['name'], score_fn,
                             plot_marker=strategy['plot_marker'])
    score_stats.run(max_time=strategies_max_evaluation_time,
                    display_every=display_every)
    score_stats.summary()
    all_score_stats.append(score_stats)

plot_scores_statistics(all_score_stats)
