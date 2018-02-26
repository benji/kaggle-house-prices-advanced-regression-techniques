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

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from utils.utils import *
from utils.kaggle import *
from utils.Training import *
from utils.Variants import *

t = training()
t.explode_columns_possibilities()

if False:
    train_variants = [['TotalSF', 'linearize_order_2'], ['OverallQual', 'target_mean'], ['Neighborhood', 'one_hot'], ['OverallCond', 'one_hot'], ['BsmtUnfSF', 'none'], ['Age', 'quantile_bin(10)'], ['LotArea', 'none'], ['GarageCars', 'none'], ['MSZoning', 'one_hot'], ['Fireplaces', 'none'], ['Functional', 'label_encoding'], ['SaleCondition', 'target_mean'], ['Condition1', 'one_hot'], ['GrLivArea', 'linearize_order_2'], ['YearBuilt', 'none'], ['KitchenQual', 'one_hot'], ['BsmtExposure', 'one_hot'], ['KitchenAbvGr', 'none'], ['YearsSinceRemodel', 'extract_0_linearize_order_2'], ['CentralAir', 'one_hot'], ['ScreenPorch', 'extract_0'], ['BsmtQual', 'one_hot'], ['BsmtFinType1', 'label_encoding'], ['PoolArea', 'none'], ['HeatingQC', 'target_mean'], ['LowQualFinSF', 'linearize_order_2'], ['OverallCond_numerical', 'none'], ['BsmtFinType2', 'target_mean'], ['WoodDeckSF', 'extract_0_linearize_order_2'], ['LotConfig', 'one_hot'], ['1stFlrSF', 'quantile_bin(10)'], [
        'GarageQual', 'target_mean'], ['GarageFinish', 'one_hot'], ['Exterior1st', 'label_encoding'], ['YearRemodAdd', 'none'], ['RoofMatl', 'target_mean'], ['FullBath_categorical', 'target_mean'], ['BsmtFullBath', 'none'], ['TotalBsmtSF', 'quantile_bin(5)'], ['BsmtFinSF1', 'linearize_order_2'], ['HalfBath', 'linearize_order_2'], ['2ndFlrSF', 'extract_0_linearize_order_2'], ['GarageArea', 'quantile_bin(5)'], ['OpenPorchSF', 'linearize_order_2'], ['Remodeled', 'extract_0_linearize_order_2'], ['Foundation', 'label_encoding'], ['ExterCond', 'target_mean'], ['BsmtFinSF2', 'linearize_order_2'], ['BsmtHalfBath', 'none'], ['Street', 'one_hot'], ['HouseStyle', 'label_encoding'], ['Exterior2nd', 'label_encoding'], ['SaleType', 'label_encoding'], ['FullBath', 'linearize_order_2'], ['GarageCars_categorical', 'target_mean'], ['RecentRemodel', 'extract_0'], ['Fireplaces_categorical', 'label_encoding'], ['RoofStyle', 'label_encoding'], ['GarageYrBlt', 'extract_0_linearize_order_2']]
    train_cols = [v[0] for v in train_variants]
    t.retain_columns(train_cols)
    variants = Variants(t, verbose=True)
    variants.apply_variants(t, train_variants)
else:
    t.dummify_all_categoricals()
    #t.label_encode_all_categoricals()

#t.autoremove_ouliers()
#t.drop_rows_by_indexes([462, 588, 632, 968, 1324])
#for i in [30,495,523,1298]:
#    t.drop_row_by_index(i)
for i in [ 462, 523, 588, 632, 825, 968, 1324 ]:
    t.drop_row_by_index(i)


t.ready_for_takeoff()
t.scale()
t.shuffle()
t.remove_columns_with_unique_value()
t.sanity()
t.summary()
# t.save('tmp')


assert t.diagnose_nas() == 0
X, X_test = t.df_train.values, t.df_test.values
#X,X_test = t.pca(n_components=100)
y = t.labels.values

model = Lasso(alpha=0.0005, random_state=1)


def train(x, y):
    model.fit(x, y)


def predict(x):
    return model.predict(x)


rmse_score = custom_score_using_kfolds(
    train,
    predict,
    rmse,
    np.array(X),
    np.array(y),
    n_splits=10,
    scale=False,
    doShuffle=False)

print 'KFold RMSE:', rmse_score


#print 'weigths',model0.coef_
#print 'bias', model0.intercept_

df_predicted = pd.DataFrame(columns=['Id', 'SalePrice'])
df_predicted['Id'] = t.test_ids
df_predicted.set_index('Id')
df_predicted['SalePrice'] = np.exp(model.predict(X_test))
df_predicted.to_csv('predicted.csv', sep=',', index=False)

print 'predictions done.'
