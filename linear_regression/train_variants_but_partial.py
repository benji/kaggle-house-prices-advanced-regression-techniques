import math
import json
import sys
import os
import pandas as pd
from os import path
import yaml

from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, BayesianRidge, LassoLarsIC, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.base import clone

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from utils.utils import *
from utils.kaggle import *
from utils.Training import *
from utils.Variants import *

np.random.seed(int(time.time()))


def seed():
    return np.random.randint(2**32-1)


t = training(exclude_outliers=True)
t.shuffle()

partials = []
for i in t.df_train[t.df_train['SaleCondition'] == 'Partial'].index:
    partials.append(i)

# t.save('tmp')
# sys.exit(0)
t.explode_columns_possibilities()

if False:
    variants = Variants(t, verbose=True)
    train_variants = variants.all_individual_variants
    # train_cols = [v[0] for v in train_variants]
    # t.retain_columns(train_cols)
    variants.remove_invalid_variants()
    variants.apply_variants(
        t, variants.all_individual_variants, copy_column=True)
    t.remove_categoricals()

fit_intercept = True
if False:
    train_variants = [['OverallQual', 'target_mean'], ['TotalSF', 'extract_0_linearize_order_2'], ['Neighborhood', 'one_hot'], ['BsmtUnfSF', 'quantile_bin(5)'], ['OverallCond_numerical', 'none'], ['YearBuilt', 'quantile_bin(20)'], ['LotArea', 'extract_0_linearize_order_2'], ['Fireplaces', 'linearize_order_2'], ['MSZoning', 'one_hot'], ['GarageArea', 'linearize_order_2'], ['Condition1', 'target_median'], [
        'KitchenQual', 'one_hot'], ['Functional', 'label_encoding'], ['HeatingQC', 'label_encoding'], ['GrLivArea', 'none'], ['SaleCondition', 'target_mean'], ['BsmtExposure', 'one_hot'], ['KitchenAbvGr', 'linearize_order_3'], ['Exterior1st', 'one_hot'], ['RecentRemodel', 'extract_0'], ['CentralAir', 'one_hot'], ['ScreenPorch', 'linearize_order_2'], ['BsmtFullBath_categorical', 'one_hot']]
    train_cols = [v[0] for v in train_variants]
    t.retain_columns(train_cols)
    variants = Variants(t, verbose=True)
    variants.apply_variants(t, train_variants)
else:
    t.dummify_all_categoricals()
    # t.label_encode_all_categoricals()

# t.autoremove_ouliers()
# t.drop_rows_by_indexes([462, 588, 632, 968, 1324])
# for i in [30,495,523,1298]:
#    t.drop_row_by_index(i)
# for i in [ 462, 523, 588, 632, 825, 968, 1324 ]:
#    t.drop_row_by_index(i)


t.ready_for_takeoff()
t.scale()
if not fit_intercept:
    t.scale_target()
# t.remove_columns_with_unique_value()
t.sanity()
t.summary()

model0 = Lasso(alpha=0.0005, fit_intercept=fit_intercept,
               random_state=seed(), warm_start=False, max_iter=10000)

holdout = 350
training_remove_partials = True

if True:
    n_times = 3
    for train_times in range(1, n_times+1):
        print '------'
        X, y = shuffle(t.df_train.values, t.labels.values)

        # 1 Kfold scoring
        model = clone(model0)
        score = custom_score_using_kfolds(model.fit, model.predict, rmse,
                                          X, y, n_splits=3, doShuffle=True, seed=seed(), average_over_n=1)
        print 'Kfold score:', score

        # 2 holdout scoring
        model = clone(model0)
        X_train, y_train = shuffle(
            X[:-holdout], y[:-holdout], random_state=seed())
        model.fit(X_train, y_train)
        t.lasso_stats(model, print_n_first_important_cols=10)

        y_pred = model.predict(X[-holdout:])
        s = rmse(y_pred, y[-holdout:])
        print 'Holdout score', s
        print 'kfold - holdout score diff:', np.abs(score-s)

        if training_remove_partials:
            # 3 SaleCondition=Partial scoring
            t1 = t.copy()
            for i in partials:
                t1.drop_row_by_index(i)

            model = clone(model0)
            model.fit(t1.df_train.values, t1.labels.values)
            y_pred = model.predict(X)
            s = rmse(y_pred, y)
            print 'Remove Partial on train score', s

            t1.lasso_stats(model, print_n_first_important_cols=10)

# todo see how a model fit against all non partial does with all including partial

else:
    if training_remove_partials:
        for i in partials:
            t.drop_row_by_index(i)
    X, X_test = t.df_train.values, t.df_test.values
    y = t.labels.values

    model = clone(model0)
    model.fit(X, y)
    t.lasso_stats(model)

    df_predicted = pd.DataFrame(columns=['Id', 'SalePrice'])
    df_predicted['Id'] = t.test_ids
    df_predicted.set_index('Id')
    df_predicted['SalePrice'] = t.untransform_target(model.predict(X_test))
    df_predicted.to_csv('predicted.csv', sep=',', index=False)

    print 'predictions done.'
