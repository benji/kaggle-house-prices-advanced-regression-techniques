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

t = training()
t.explode_columns_possibilities()
t.health_check()

score_test_ratio = 0.1
n_scores_per_variant = 20


def score(X, y_true):
    model = Lasso(alpha=0.001)
    scores = []
    for i in range(n_scores_per_variant):
        scores.append(score_using_test_ratio(
            model.fit, model.predict, rmse, X, y_true, test_ratio=score_test_ratio))
    return np.array(scores[5:-5]).mean()
    # return custom_score_using_kfolds(model.fit, model.predict, rmse, X, y_true, n_splits=10, doShuffle=False, scale=True)


mean_target = t.labels.mean()
y_mean = np.ones(len(t.labels))*mean_target
mean_score = rmse(y_mean, t.labels.values)
print 'Score of the constant mean prediction:', mean_score

autotrain = Autotrain(verbose=False, stop_if_no_progress=False)
variants = Variants(t, score_fn=score)
variants.remove_invalid_variants()

best_variants, score = autotrain.find_multiple_best(
    variants_fn=variants.generate_variants,
    score_variants_fn=variants.score_variants,
    on_validated_variant_fn=variants.commit_variant,
    goal='min', tqdm=True)

print 'Final variants', best_variants
print 'Final score', score
