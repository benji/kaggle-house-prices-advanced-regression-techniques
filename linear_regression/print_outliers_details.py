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

t = training(exclude_outliers=True,remove_partials=False)
t.shuffle()

for idx in [632,462,523,1298]:
        t.drop_row_by_index(idx)

t0 = t.copy()
t.explode_columns_possibilities()
t.dummify_all_categoricals()


t.ready_for_takeoff()
t.scale()


t.remove_columns_with_unique_value()
t.sanity()
t.summary()
#t.save('tmp', index=True)
# sys.exit(0)

#for i in partials:
#    t.drop_row_by_index(i)

model0 = Lasso(alpha=0.0005, fit_intercept=True,
               random_state=seed(), warm_start=False,max_iter=10000)  # , max_iter=10000)

model = clone(model0)
score = custom_score_using_kfolds(model.fit, model.predict, rmse,
                                    t.df_train.values, t.labels.values,
                                    n_splits=3, doShuffle=True, seed=seed(),
                                    average_over_n=1)
print 'Kfold score:', score

saleprice_model = clone(model0)
saleprice_model.fit(t.df_train.values, t.labels.values)

scores = [rmse(y_pred, y_true) for y_pred, y_true in zip(
    saleprice_model.predict(t.df_train.values), t.labels.values)]
scores = np.array(scores)

print 'DIFF scores', scores[:5]
print 'mean DIFF scores=',np.mean(scores)
#sys.exit(0)

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

print 'LABELS PREDICTION MODEL COEFS:'
t.lasso_stats(saleprice_model, print_n_first_important_cols=15)


if False:
    model = clone(model0)
    model.fit(t.df_train.values, scores)
    print 'score on train', rmse(model.predict(t.df_train.values), scores)

    # print most significant features to predict outliers
    print 'SCORES PREDICTION MODEL COEFS:'
    t.lasso_stats(model, print_n_first_important_cols=15)

if True:
    score = custom_score_using_kfolds(model.fit, model.predict, rmse,
                                      t.df_train.values, scores, n_splits=10, doShuffle=False, seed=seed(), average_over_n=1)
    print 'Kfold score:', score

score_model = clone(model0)
score_model.fit(t.df_train.values, scores)

# try to predict an outlier
scores_map = t.find_worst_predicted_points(saleprice_model, rmse)

for item in scores_map[:2]:
    print '-------------'
    idx = item[0]
    print 'outlier:', item, 'id:', t.train_index_to_id(idx)
    actual = t.labels[idx]
    row = t.df_train.loc[idx]
    pred = saleprice_model.predict([row])[0]
    print 'SalePrice Actual:', actual, 'Predicted:', pred, 'diff', str(
        pred-actual),'rmse',rmse(pred,actual)
    row0 = t0.df_train.loc[idx]
    print row0['SaleCondition'],row0['SaleType'],row['1stFlrSF']
    t0.print_row_stats_by_index(
        idx, max_categorical_pct=5, max_numerical_pct=5)
