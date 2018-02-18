import pandas as pd
import math
import json
import sys
import operator
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


def score(X, y_true):
    model = Lasso(alpha=0.0005)  # , random_state=1)
    return custom_score_using_kfolds(model.fit, model.predict, rmse, X, y_true, n_splits=10, doShuffle=True, scale=True)


mean_target = t.labels.mean()
y_mean = np.ones(len(t.labels))*mean_target
mean_score = rmse(y_mean, t.labels.values)
print 'Score of the constant mean prediction:', mean_score

variants = Variants(t, score_fn=score)

best_variants = [['TotalSF', 'linearize_order_2'], ['OverallQual', 'target_mean'], ['Neighborhood', 'one_hot'], ['OverallCond', 'one_hot'], ['BsmtUnfSF', 'none'], ['Age', 'quantile_bin(10)'], ['LotArea', 'none'], ['GarageCars', 'none'], ['MSZoning', 'one_hot'], ['Fireplaces', 'none'], ['Functional', 'label_encoding'], ['SaleCondition', 'target_mean'], ['Condition1', 'one_hot'], ['GrLivArea', 'linearize_order_2'], ['YearBuilt', 'none'], ['KitchenQual', 'one_hot'], ['BsmtExposure', 'one_hot'], ['KitchenAbvGr', 'none'], ['YearsSinceRemodel', 'extract_0_linearize_order_2'], ['CentralAir', 'one_hot'], ['ScreenPorch', 'extract_0'], ['BsmtQual', 'one_hot'], ['BsmtFinType1', 'label_encoding'], ['PoolArea', 'none'], ['HeatingQC', 'target_mean'], ['LowQualFinSF', 'linearize_order_2'], ['OverallCond_numerical', 'none'], ['BsmtFinType2', 'target_mean'], ['WoodDeckSF', 'extract_0_linearize_order_2'], ['LotConfig', 'one_hot'], ['1stFlrSF', 'quantile_bin(10)'], ['GarageQual', 'target_mean'], [
    'GarageFinish', 'one_hot'], ['Exterior1st', 'label_encoding'], ['YearRemodAdd', 'none'], ['RoofMatl', 'target_mean'], ['FullBath_categorical', 'target_mean'], ['BsmtFullBath', 'none'], ['TotalBsmtSF', 'quantile_bin(5)'], ['BsmtFinSF1', 'linearize_order_2'], ['HalfBath', 'linearize_order_2'], ['2ndFlrSF', 'extract_0_linearize_order_2'], ['GarageArea', 'quantile_bin(5)'], ['OpenPorchSF', 'linearize_order_2'], ['Remodeled', 'extract_0_linearize_order_2'], ['Foundation', 'label_encoding'], ['ExterCond', 'target_mean'], ['BsmtFinSF2', 'linearize_order_2'], ['BsmtHalfBath', 'none'], ['Street', 'one_hot'], ['HouseStyle', 'label_encoding'], ['Exterior2nd', 'label_encoding'], ['SaleType', 'label_encoding'], ['FullBath', 'linearize_order_2'], ['GarageCars_categorical', 'target_mean'], ['RecentRemodel', 'extract_0'], ['Fireplaces_categorical', 'label_encoding'], ['RoofStyle', 'label_encoding'], ['GarageYrBlt', 'extract_0_linearize_order_2'], ['KitchenAbvGr_categorical', 'label_encoding']]
best_variants_strs = [v[0]+'_'+v[1] for v in best_variants]

v = best_variants[0]
scores = []

i = 0
while i < 1500:
    scores.append(variants.score_variants([v]))
    if i % 25 == 0:
        print i
    i += 1

mean = np.array(scores).mean()
diffs = []
avg_diffs = []
i = 0
for s in scores:
    diff = s-mean
    diffs.append(diff)
    avg_diff = np.abs(np.array(diffs).mean())
    avg_diffs.append(avg_diff)
    if i % 10 == 0:
        print '{:f} - {}'.format(avg_diff, i)
    i += 1

print np.std(diffs)

#ax = sns.distplot(diffs)
# plt.show()

#plt.plot(avg_diffs)
#plt.show()

abs_diffs = np.abs(np.array(scores)-mean)
plt.plot(abs_diffs)
plt.show()

def average(arr, n):
    end =  n * int(len(arr)/n)
    return np.mean(arr[:end].reshape(-1, n), 1)
    
plt.plot(average(abs_diffs,3))
plt.show()
plt.plot(average(abs_diffs,5))
plt.show()
plt.plot(average(abs_diffs,7))
plt.show()

#i = 0
#while True:
#    scores.append(variants.score_variants([v]))
#    if i % 10 == 0:
#        print np.max(np.abs(np.array(scores)-mean)).max()
#    i += 1
