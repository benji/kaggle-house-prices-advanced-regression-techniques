import pandas as pd
import math
import json
import sys
import time
from tqdm import tqdm
import numpy as np
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from utils.utils import *
from utils.Training import *
from utils.Autotrain import *
from utils.kaggle import *
from utils.Variants import *

np.random.seed(int(time.time()))


def seed():
    return np.random.randint(2**32-1)


t = training()
t.shuffle()
t.explode_columns_possibilities()
t.health_check()

score_test_ratio = 0.6
score_kfolds = 8
n_scores_per_variant = 10
stop_if_no_progress = True

holdout = 400


def score(X, y_true):
    model = Lasso(alpha=0.001)
    scores = []
    for i in range(n_scores_per_variant):
        X_score, y_score = shuffle(X[:-holdout], y_true[:-holdout])

        score = custom_score_using_kfolds(
            model.fit, model.predict, rmse, X_score, y_score, n_splits=score_kfolds, doShuffle=True, seed=seed())
        #score_using_test_ratio(model.fit, model.predict, rmse, X, y_true, test_ratio=score_test_ratio)
        scores.append(score)
    return np.array(scores).mean()
    # return custom_score_using_kfolds(model.fit, model.predict, rmse, X, y_true, n_splits=10, doShuffle=False, scale=True)


def score_holdout(variants):
    t_h = t.copy()
    t_h.verbose = False
    t_h.retain_columns([v[0] for v in variants])
    variants_h = Variants(t_h, verbose=False)
    variants_h.apply_variants(t_h, variants)

    model = Lasso(alpha=0.001)
    model.fit(t_h.df_train.values[:-holdout], t_h.labels.values[:-holdout])
    predicted = model.predict(t_h.df_train.values[-holdout:])
    expected = t_h.labels.values[-holdout:]

    print 'holdout score:', rmse(predicted, expected)


autotrain = Autotrain(verbose=False, stop_if_no_progress=stop_if_no_progress)
variants = Variants(t, score_fn=score)
variants.remove_invalid_variants()

best_variants, score = autotrain.find_multiple_best(
    variants_fn=variants.generate_variants,
    score_variants_fn=variants.score_variants,
    on_validated_variant_fn=variants.commit_variant,
    on_round_finish_fn=score_holdout,
    goal='min', tqdm=True)

print 'Final variants', best_variants
print 'Final score', score
