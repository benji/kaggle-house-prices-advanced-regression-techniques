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


# 1. find outliers
# we first evaluate how far each train observation is from the trained model prediction
# the ones that score the worst are suspected to be outliers

# 2. find neighbors
# we use KNN to find the training observations closest to an outlier
# when calculating the distance, we use the lasso coefs to increase the significant of more significant features.

# 3. prediction
# use the neighbors' labels to predict the outliers label

# Results
# As expected, it doesn't work so well
# The outliers in this dataset don't have any data that is representative of them

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

t.df_train.index.name = 'foo'

id = 633

#print 'index before', t.train_id_to_index(id)
#print 'id before', t.train_ids.iloc[t.train_id_to_index(id)]
#print 'label before', t.labels.iloc[t.train_id_to_index(id)]

t.shuffle()

#print 'index after', t.train_id_to_index(id)
#print 'id after', t.train_ids.iloc[t.train_id_to_index(id)]
#print 'label after', t.labels.iloc[t.train_id_to_index(id)]
#idx = t.train_id_to_index(id)
#print 't.labels.iloc[',idx,'] =', t.labels.iloc[idx]

t.remove_columns_with_unique_value()
t.sanity()
t.summary()
t.save('tmp', index=True)
# sys.exit(0)
model0 = Lasso(alpha=0.0005, fit_intercept=True,
               random_state=seed(), warm_start=False, max_iter=10000)

model = clone(model0)
model.fit(t.df_train.values, t.labels.values)

print 'y_true', t.labels.values[:5]

if False:
    for i in [632]:  # range(len(t.df_train.values)):
        print "HEEREE index:", i
        print 'id', t.train_ids.iloc[i]
        y_pred = model.predict([t.df_train.values[i]])
        y_true = t.labels.values
        print y_pred[0], y_true[i]
        print i, y_pred, y_true[i], rmse(y_pred[0], y_true[i])
        print "HEEREE end"

scores = [rmse(y_pred, y_true) for y_pred, y_true in zip(
    model.predict(t.df_train.values), t.labels.values)]

print 'scores', scores[:5]

if False:
    # try predict scores
    do_scale = True

    if do_scale:
        scaler = StandardScaler()
        tmp = [[v] for v in scores]
        tmp = scaler.fit_transform(tmp)
        tmp = [v[0] for v in tmp]
        scores = np.array(tmp)
        print 'after scaling', scores[:10]

    #t.labels = scores
    # t.save('../scores')
    # pd.DataFrame(index=t.df_train.index, data=scores).to_csv(
    #   '../scores/scores.csv', sep=',')

    model = clone(model0)
    model.fit(t.df_train.values, scores)

for item in t.find_worst_predicted_points(model, rmse, limit=5):
    print '-------------'
    print 'outlier:', item
    idx = item[0]  # t.find_index( item[0] )
    print 'index:', idx
    print 'id', t.train_index_to_id(idx)
    label = t.labels.iloc[idx]
    print 't.labels.iloc[', idx, '] =', t.labels.iloc[idx]
    print 'actual label:', label
    X = t.df_train.iloc[idx]
    lasso_predicted = model.predict([X])[0]
    print 'lasso prediction label:', lasso_predicted
    lasso_error = rmse(lasso_predicted, label)
    print 'lasso prediction error:', lasso_error

    nearest_neighbor_indexes = t.find_nearest_train_neighbor_indexes(
        X, n_neighbors=3, dim_scaling_ratios=model.coef_, ignore_first=True)
    print 'neighbors', nearest_neighbor_indexes
    neigh_labels = []
    for nearest_neighbor_index in nearest_neighbor_indexes:
        nn_label = t.labels.iloc[nearest_neighbor_index]
        print '1nearest neighbor label:', nn_label
        neigh_labels.append(nn_label)
    avg_label = np.array(neigh_labels).mean()
    nn_error = rmse(label, avg_label)
    print 'error of outlier based on AVG nn label', nn_error
    print '*** error diff', nn_error-lasso_error

    if False:
        nearest_neighbor_index = t.find_nearest_train_neighbor_index(
            X, ignore_first=True)
        nn_label = t.labels.iloc[nearest_neighbor_index]
        print '1nearest neighbor label:', nn_label
        nn_error = rmse(label, nn_label)
        print '1error of outlier based on nn label', nn_error
        print '*** 1error diff', nn_error-lasso_error

        nearest_neighbor_index = t.find_nearest_train_neighbor_index(
            X, dim_scaling_ratios=model.coef_, ignore_first=True)
        nn_label = t.labels.iloc[nearest_neighbor_index]
        print '2nearest neighbor label:', nn_label
        nn_error = rmse(label, nn_label)
        print '2error of outlier based on nn label', nn_error
        print '*** 2error diff', nn_error-lasso_error


sys.exit(0)

model = clone(model0)

print 'mean prediction score:', mean_prediction_score(
    model.fit, model.predict, rmse, t.df_train.values, scores)

model = clone(model0)
model.fit(t.df_train.values, scores)
mean_pred = np.array([scores.mean() for _ in scores])
print 'mean pred 2', rmse(model.predict(t.df_train.values), mean_pred)

model = clone(model0)
model.fit(t.df_train.values, scores)
print 'score on train', rmse(model.predict(t.df_train.values), scores)

if False:
    for i in range(10):
        score = custom_score_using_kfolds(model.fit, model.predict, rmse,
                                          t.df_train.values, scores, n_splits=3, doShuffle=True, seed=seed(), average_over_n=1)
        print 'Kfold score:', score
