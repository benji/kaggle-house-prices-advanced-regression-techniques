import pandas as pd
import math
import json
import sys
import operator
import time
from tqdm import tqdm
import numpy as np
from os import path
import seaborn as sns

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from utils.utils import *
from utils.Training import *
from utils.Autotrain import *
from utils.Variants import *
from utils.kaggle import *
import matplotlib.pyplot as plt

t = training()

t.dummify_at_init = False
t.dummify_drop_first = False
t.use_label_encoding = False
#t.train_columns = ['TotalBsmtSF']
t.prepare()

# seed numpy
_seed = int(time.time())
print 'numpy seed', _seed
np.random.seed(_seed)


def seed():
    _seed = np.random.randint(2**32-1)
    #print 'seed', _seed
    return _seed


def score_method_kfolds(n_splits=10):
    def score(X, y):
        model = Lasso(alpha=0.0005)  # , random_state=1)
        return custom_score_using_kfolds(model.fit, model.predict, rmse, X, y, n_splits=n_splits, doShuffle=True, scale=True)
    return score


def score_method_split(test_ratio=.2):
    def score(X, y):
        model = Lasso(alpha=0.0005)  # , random_state=1)
        X, y = doShuffle(X, y, seed=seed())
        return score_using_test_ratio(model.fit, model.predict, rmse, X, y, test_ratio=test_ratio)
    return score


# the mean predition

mean_target = t.labels.mean()
y_mean = np.ones(len(t.labels))*mean_target
mean_score = rmse(y_mean, t.labels.values)
print 'Score of the constant mean prediction:', mean_score

# Pick a variant

best_variants = [['TotalSF', 'linearize_order_2'], ['OverallQual', 'target_mean'], ['Neighborhood', 'one_hot'], ['OverallCond', 'one_hot'], ['BsmtUnfSF', 'none'], ['Age', 'quantile_bin(10)'], ['LotArea', 'none'], ['GarageCars', 'none'], ['MSZoning', 'one_hot'], ['Fireplaces', 'none'], ['Functional', 'label_encoding'], ['SaleCondition', 'target_mean'], ['Condition1', 'one_hot'], ['GrLivArea', 'linearize_order_2'], ['YearBuilt', 'none'], ['KitchenQual', 'one_hot'], ['BsmtExposure', 'one_hot'], ['KitchenAbvGr', 'none'], ['YearsSinceRemodel', 'extract_0_linearize_order_2'], ['CentralAir', 'one_hot'], ['ScreenPorch', 'extract_0'], ['BsmtQual', 'one_hot'], ['BsmtFinType1', 'label_encoding'], ['PoolArea', 'none'], ['HeatingQC', 'target_mean'], ['LowQualFinSF', 'linearize_order_2'], ['OverallCond_numerical', 'none'], ['BsmtFinType2', 'target_mean'], ['WoodDeckSF', 'extract_0_linearize_order_2'], ['LotConfig', 'one_hot'], ['1stFlrSF', 'quantile_bin(10)'], ['GarageQual', 'target_mean'], [
    'GarageFinish', 'one_hot'], ['Exterior1st', 'label_encoding'], ['YearRemodAdd', 'none'], ['RoofMatl', 'target_mean'], ['FullBath_categorical', 'target_mean'], ['BsmtFullBath', 'none'], ['TotalBsmtSF', 'quantile_bin(5)'], ['BsmtFinSF1', 'linearize_order_2'], ['HalfBath', 'linearize_order_2'], ['2ndFlrSF', 'extract_0_linearize_order_2'], ['GarageArea', 'quantile_bin(5)'], ['OpenPorchSF', 'linearize_order_2'], ['Remodeled', 'extract_0_linearize_order_2'], ['Foundation', 'label_encoding'], ['ExterCond', 'target_mean'], ['BsmtFinSF2', 'linearize_order_2'], ['BsmtHalfBath', 'none'], ['Street', 'one_hot'], ['HouseStyle', 'label_encoding'], ['Exterior2nd', 'label_encoding'], ['SaleType', 'label_encoding'], ['FullBath', 'linearize_order_2'], ['GarageCars_categorical', 'target_mean'], ['RecentRemodel', 'extract_0'], ['Fireplaces_categorical', 'label_encoding'], ['RoofStyle', 'label_encoding'], ['GarageYrBlt', 'extract_0_linearize_order_2'], ['KitchenAbvGr_categorical', 'label_encoding']]
best_variants_strs = [v[0]+'_'+v[1] for v in best_variants]

v = best_variants[0]

# Figure out reliable mean

n_scores_for_mean = 300
n_scores = 3000

variants = Variants(t, score_fn=score_method_split(test_ratio=.4))

scores = []
i = 0
while i < n_scores_for_mean:
    scores.append(variants.score_variants([v]))
    if i % 25 == 0:
        print i
    i += 1

mean = np.array(scores).mean()
print 'Mean  =', mean
main_scores = scores

# run tests

tests_data = []

for test_ratio in reversed([.05, .1, .25, .5, .75, .9]):
    print 'test_ratio', test_ratio
    test_data = {}

    variants = Variants(t, score_fn=score_method_split(test_ratio=test_ratio))

    # Collect scores data

    scores = []
    i = 0
    while i < n_scores:
        scores.append(variants.score_variants([v]))
        if i % 25 == 0:
            print i
        i += 1

    # Mean

    test_mean = np.array(scores).mean()
    test_data['mean'] = test_mean
    print 'Test mean  =', test_mean
    print 'Diff between real mean and test mean =', np.abs(test_mean - mean)

    # Compute diff variation around the mean score

    diffs = []
    avg_diffs = []
    avg_scores = []
    scores_so_far = []
    for s in scores:
        scores_so_far.append(s)
        diff = s - mean
        diffs.append(diff)
        avg_diff = np.abs(np.array(diffs).mean())
        avg_diffs.append(avg_diff)
        avg_scores.append(np.array(scores_so_far).mean())

    sigma = np.std(diffs)
    print 'Diff sigma =', sigma

    test_data['scores'] = scores
    test_data['diffs'] = diffs
    test_data['avg_diffs'] = avg_diffs
    test_data['avg_scores'] = avg_scores

    test_data['name'] = 'TR_'+str(test_ratio)+' s='+str(sigma)

    tests_data.append(test_data)

# Plot results


def showplot(title):
    plt.title(title)
    plt.show()


for td in tests_data:
    sns.distplot(td['diffs'], hist=False, label=td['name'])
plt.axvline(x=0)
showplot('comparing standard deviation of different scoring methods/paramers')

for td in tests_data:
    plt.plot(td['avg_diffs'], label=td['name'])
plt.legend(loc='upper left')
plt.ylim(ymin=0)
showplot('mean average convergence speed')


for td in tests_data:
    plt.plot(td['avg_scores'], label=td['name'])
#plt.plot(main_scores, label=td['name'])
plt.legend(loc='upper left')
showplot('scores')
