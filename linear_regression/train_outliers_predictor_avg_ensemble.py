import math
import json
import sys
import os
import pandas as pd
from os import path
import yaml

from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, BayesianRidge, LassoLarsIC, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.base import clone
import xgboost as xgb
import lightgbm as lgb

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from utils.utils import *
from utils.kaggle import *
from utils.Training import *
from utils.Variants import *
from utils.AveragingModels import *

# Try to predict the error of our regular model.

# 1. Compute score of regular model for each observations
# 2. Build a model where that score (RMSE) is the target variable
 
# Result:
# There is no 'pattern' across the outliers.
# We end up with an estimator that predict a couple of the biggest outliers.

np.random.seed(int(time.time()))


def seed():
    return np.random.randint(2**32-1)


do_holdout = False
holdout = 350
kfold_average_over_n = 1

t = training(exclude_outliers=False)
t.explode_columns_possibilities()
t.dummify_all_categoricals()


t.ready_for_takeoff()
t.scale()

t.shuffle()

t.remove_columns_with_unique_value()
t.sanity()
t.summary()
#t.save('tmp', index=True)
# sys.exit(0)


def new_model():
    lasso = Lasso(alpha=0.0005, fit_intercept=True,
                  random_state=seed(), warm_start=False, max_iter=10000)
    KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
    ENet = ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=seed())

    GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                       max_depth=4, max_features='sqrt',
                                       min_samples_leaf=15, min_samples_split=10,
                                       loss='huber', random_state=seed())

    model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                                 learning_rate=0.05, max_depth=3,
                                 min_child_weight=1.7817, n_estimators=2200,
                                 reg_alpha=0.4640, reg_lambda=0.8571,
                                 subsample=0.5213, silent=1,
                                 random_state=7, nthread=-1)

    model_lgb = lgb.LGBMRegressor(objective='regression', num_leaves=5,
                                  learning_rate=0.05, n_estimators=720,
                                  max_bin=55, bagging_fraction=0.8,
                                  bagging_freq=5, feature_fraction=0.2319,
                                  feature_fraction_seed=seed(), bagging_seed=seed(),
                                  min_data_in_leaf=6, min_sum_hessian_in_leaf=11)

    return AveragingModels(models=(ENet,  KRR, lasso, GBoost, model_xgb, model_lgb))
    # return lasso


model = new_model()



saleprice_model = new_model()
saleprice_model.fit(t.df_train.values, t.labels.values)

scores = [rmse(y_pred, y_true) for y_pred, y_true in zip(
    saleprice_model.predict(t.df_train.values), t.labels.values)]
scores = np.array(scores)

print 'scores', scores[:5]

# try predict scores
do_scale = False

if do_scale:
    scaler = StandardScaler()
    tmp = [[v] for v in scores]
    tmp = scaler.fit_transform(tmp)
    tmp = [v[0] for v in tmp]
    scores = np.array(tmp)
    print 'after scaling', scores[:10]


# PRINT MEAN PREDICTION

model = new_model()
print 'mean prediction score:', mean_prediction_score(rmse, scores)

mean_pred = np.array([scores.mean() for _ in scores])
print 'mean pred 2', rmse(mean_pred, scores)

#t.labels = scores
# t.save('../scores')
# pd.DataFrame(index=t.df_train.index, data=scores).to_csv(
#   '../scores/scores.csv', sep=',')

model = new_model()
model.fit(t.df_train.values, scores)
print 'score on train', rmse(model.predict(t.df_train.values), scores)

# print most significant features to predict outliers
#t.lasso_stats(model, print_n_first_important_cols=20)

if True:
    score = custom_score_using_kfolds(model.fit, model.predict, rmse,
                                      t.df_train.values, scores, n_splits=10, doShuffle=False, seed=seed(), average_over_n=1)
    print 'Kfold score:', score

score_model = new_model()
score_model.fit(t.df_train.values, scores)

# try to predict an outlier
scores_map = t.find_worst_predicted_points(saleprice_model, rmse)

for item in scores_map[:20]:
    #print '-------------'
    print 'outlier:', item, 'id:', t.train_index_to_id(item[0])
    if False:
        idx = item[0]
        print 'index:', idx
        print 'id', t.train_index_to_id(idx)
        print 'scored:', item[1]
        label = t.labels[idx]
        print 'actual label:', label
        X = t.df_train.ix[idx]
        #X = t.df_train.loc[idx]
        lasso_predicted = saleprice_model.predict([X])[0]
        print 'lasso prediction label:', lasso_predicted
        lasso_error = rmse(lasso_predicted, label)
        print 'lasso prediction error:', lasso_error

        feats = t.df_train.loc[idx]
        score_pred = score_model.predict([X])
        print 'predicted error', score_pred

predicted_scores = {}
score_i = 0
for index, row in t.df_train.iterrows():
    y_pred = score_model.predict([row])
    y_true = np.array([scores[score_i]])
    score = rmse(y_pred, y_true)
    #print 'index', index, 'pred',y_pred,'true',y_true,'score',score
    predicted_scores[index] = score
    score_i += 1


def find_rank_of_index(idx):
    i = 0
    for item in scores_map:
        if item[0] == idx:
            return i
        i += 1


# compare predicted and actual ranks
i = 0
for item in t.sort_dict_by_value(predicted_scores, reverse=True)[:50]:
    idx = item[0]
    print i, find_rank_of_index(idx)
    i += 1

'''
outlier: (632, 0.608235208408237) id: 633
outlier: (462, 0.5128304572440161) id: 463
outlier: (1324, 0.5041258619268767) id: 1325
outlier: (1298, 0.42668444598458066) id: 1299
outlier: (30, 0.4054223424996177) id: 31
outlier: (970, 0.3466445550512294) id: 971
outlier: (688, 0.33870296616614404) id: 689
outlier: (1453, 0.32312960070677477) id: 1454
outlier: (968, 0.3175734196786877) id: 969
outlier: (1432, 0.30852026619016293) id: 1433
outlier: (495, 0.29936871665371) id: 496
outlier: (588, 0.29774856584744924) id: 589
outlier: (523, 0.2642279317224343) id: 524
outlier: (874, 0.2377015087883141) id: 875
outlier: (681, 0.23297502945138682) id: 682
outlier: (812, 0.22439822279858745) id: 813
outlier: (714, 0.22265729573304505) id: 715
outlier: (774, 0.21369178271234013) id: 775
outlier: (581, 0.21206202042251476) id: 582
outlier: (1122, 0.21090858846759275) id: 1123
0 0
1 2
2 1
3 6
4 7
5 4
6 5
7 9
8 11
9 8
10 20
11 16
12 13
13 3
14 17
15 31
16 14
17 21
18 10
19 34
20 39
21 18
22 15
23 46
24 27
25 29
26 25
27 19
28 35
29 1268
30 28
31 23
32 26
33 49
34 42
35 68
36 84
37 37
38 41
39 36
40 56
41 70
42 12
43 51
44 71
45 65
46 40
47 111
48 67
49 47
'''