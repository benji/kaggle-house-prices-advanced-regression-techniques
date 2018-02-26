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

train_variants = train_variants = [['TotalSF', 'linearize_order_2'], ['OverallQual', 'target_mean'], ['Neighborhood', 'one_hot'], ['OverallCond', 'one_hot'], ['BsmtUnfSF', 'none'], ['Age', 'quantile_bin(10)'], ['LotArea', 'none'], ['GarageCars', 'none'], ['MSZoning', 'one_hot'], ['Fireplaces', 'none'], ['Functional', 'label_encoding'], ['SaleCondition', 'target_mean'], ['Condition1', 'one_hot'], ['GrLivArea', 'linearize_order_2'], ['YearBuilt', 'none'], ['KitchenQual', 'one_hot'], [
    'BsmtExposure', 'one_hot'], ['KitchenAbvGr', 'none'], ['YearsSinceRemodel', 'extract_0_linearize_order_2'], ['CentralAir', 'one_hot'], ['ScreenPorch', 'extract_0'], ['BsmtQual', 'one_hot'], ['BsmtFinType1', 'label_encoding'], ['PoolArea', 'none'], ['HeatingQC', 'target_mean'], ['LowQualFinSF', 'linearize_order_2'], ['OverallCond_numerical', 'none'], ['BsmtFinType2', 'target_mean'], ['WoodDeckSF', 'extract_0_linearize_order_2'], ['LotConfig', 'one_hot']]
train_cols = [v[0] for v in train_variants]
t.retain_columns(train_cols)
variants = Variants(t, verbose=True)
variants.apply_variants(t, train_variants)

t.health_check()

score_test_ratio = 0.1
n_scores_per_variant = 100


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

autotrain = Autotrain(verbose=False, stop_if_no_progress=True)

available_indexes = list(t.df_train.index)


def generate_variants(existing_variants=[]):
    return [i for i in available_indexes if i not in existing_variants]


def score_variants(outliers):
    #print 'score outliers', outliers
    t2 = t.copy()
    t2.verbose = False
    t2.drop_rows_by_indexes(outliers)
    thescore = score(t2.df_train.values, t2.labels.values)
    #print 'scored', thescore
    return thescore


def commit_variant(o):
    pass


best_variants, score = autotrain.find_multiple_best(
    variants_fn=generate_variants,
    score_variants_fn=score_variants,
    on_validated_variant_fn=commit_variant,
    goal='min', tqdm=True)

print 'Final variants', best_variants
print 'Final score', score
