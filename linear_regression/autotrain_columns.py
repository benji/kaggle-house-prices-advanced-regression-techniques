import pandas as pd
import math
import json
import sys
from tqdm import tqdm
import numpy as np
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from utils.utils import *
from utils.Training import *
from utils.Autotrain import *
from utils.kaggle import *

t = training()

t.dummify_at_init = False
t.dummify_drop_first = False
t.use_label_encoding = False

t.prepare()

use_runtime_dummies = True

available_columns = list(t.df_train.columns.values)


def new_model():
    return Lasso(alpha=0.0005, random_state=1)


def score_columns(cols):
    # print 'scoring columns', cols
    df_train = t.df_train[cols].copy()

    if use_runtime_dummies:
        df_train, _ = t.do_dummify(df_train, None, False)

    model = new_model()
    score_rmse = custom_score_using_kfolds(model.fit,
                                           model.predict,
                                           rmse,
                                           df_train.values,
                                           t.labels.values,
                                           n_splits=10,
                                           doShuffle=False,
                                           scale=True)

    return score_rmse


autotrain = Autotrain()


best_cols, score = autotrain.find_multiple_best(
    variants=available_columns, score_variants_fn=score_columns, goal='min')

print 'Final columns', best_cols
print 'Final score', score


t.retain_columns(best_cols)

if use_runtime_dummies:
    df_train, df_test = t.do_dummify(t.df_train, t.df_test, False)
else:
    df_train, df_test = t.df_train, t.df_test

generate_predictions(t.labels, df_train, df_test, t.test_ids)
print "Predictions complete."
