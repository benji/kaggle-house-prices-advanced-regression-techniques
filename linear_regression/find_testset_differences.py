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


do_holdout = False
holdout = 350
kfold_average_over_n = 1

t = training(exclude_outliers=False)
t.explode_columns_possibilities()

# t.save('tmp')
# sys.exit(0)
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
# t.scale()
if not fit_intercept:
    t.scale_target()
t.shuffle()
t.remove_columns_with_unique_value()
t.sanity()
t.summary()
# t.save('tmp')
# sys.exit(0)
model0 = Lasso(alpha=0.0005, fit_intercept=fit_intercept,
               random_state=seed(), warm_start=False, max_iter=10000)


if do_holdout:
    for train_times in range(1, 2):
        X, y = shuffle(t.df_train.values, t.labels.values)

        model = clone(model0)
        print 'kfold scoring...'
        score = custom_score_using_kfolds(model.fit, model.predict, rmse,
                                          X, y, n_splits=3, doShuffle=True, seed=seed(), average_over_n=kfold_average_over_n)
        print 'Kfold score:', score

        model = clone(model0)

        X_train, y_train = shuffle(
            X[:-holdout], y[:-holdout], random_state=seed())

        model.fit(X_train, y_train)
        # t.lasso_stats(model)

        y_pred = model.predict(X[-holdout:])
        s = rmse(y_pred, y[-holdout:])
        print 'Holdout score', s

else:
    X, X_test = t.df_train.values, t.df_test.values
    y = t.labels.values

    model = clone(model0)
    model.fit(X, y)

    df_predicted = pd.DataFrame(columns=['Id', 'SalePrice'])
    df_predicted['Id'] = t.test_ids
    df_predicted.set_index('Id')
    df_predicted['SalePrice'] = t.untransform_target(model.predict(X_test))
    df_predicted.to_csv('predicted.csv', sep=',', index=False)

    print 'predictions done.'
