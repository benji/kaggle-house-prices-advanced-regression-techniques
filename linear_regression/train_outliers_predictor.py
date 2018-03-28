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
from sklearn.base import clone

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from utils.utils import *
from utils.kaggle import *
from utils.Training import *
from utils.Variants import *

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
t.save('tmp', index=True)
# sys.exit(0)
model0 = Lasso(alpha=0.0005, fit_intercept=True,
              random_state=seed(), warm_start=False, max_iter=10000)

#model0 = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)


saleprice_model = clone(model0)
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

model = clone(model0)
print 'mean prediction score:', mean_prediction_score(rmse, scores)

mean_pred = np.array([scores.mean() for _ in scores])
print 'mean pred 2', rmse(mean_pred, scores)

#t.labels = scores
# t.save('../scores')
# pd.DataFrame(index=t.df_train.index, data=scores).to_csv(
#   '../scores/scores.csv', sep=',')

model = clone(model0)
model.fit(t.df_train.values, scores)
print 'score on train', rmse(model.predict(t.df_train.values), scores)

# print most significant features to predict outliers
#t.lasso_stats(model, print_n_first_important_cols=20)

if True:
    score = custom_score_using_kfolds(model.fit, model.predict, rmse,
                                      t.df_train.values, scores, n_splits=10, doShuffle=False, seed=seed(), average_over_n=1)
    print 'Kfold score:', score

score_model = clone(model0)
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
        #X = t.df_train.iloc[idx]
        lasso_predicted = saleprice_model.predict([X])[0]
        print 'lasso prediction label:', lasso_predicted
        lasso_error = rmse(lasso_predicted, label)
        print 'lasso prediction error:', lasso_error

        feats = t.df_train.iloc[idx]
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

t2 = t.copy()

saleprice_model2 = clone(model0)
saleprice_model2.fit(t2.df_train.values, t2.labels.values)

print rmse(saleprice_model2.predict(t2.df_train.values), t2.labels.values)

t2.df_train['score'] = score_model.predict(t2.df_train.values)
t2.df_test['score'] = score_model.predict(t2.df_test.values)
t2.schema['columns']['score'] = {'type': 'NUMERIC'}

saleprice_model2 = clone(model0)
saleprice_model2.fit(t2.df_train.values, t2.labels.values)

print rmse(saleprice_model2.predict(t2.df_train.values), t2.labels.values)
