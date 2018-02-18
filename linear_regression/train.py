import math, json, sys, os
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

t = training()


t.dummify_at_init = True
t.dummify_drop_first = False
t.use_label_encoding = False
t.replace_all_categoricals_with_mean = True
t.categoricals_with_mean=['Neighborhood']

if True:
    t.train_columns = [
        'OverallQual', 'TotalSF', 'Neighborhood', 'OverallCond', 'BsmtQual',
        'MSSubClass', 'GarageArea', 'BsmtUnfSF', 'YearBuilt', 'LotArea',
        'MSZoning', 'Fireplaces', 'Functional', 'HeatingQC', 'SaleCondition',
        'Condition1', 'BsmtExposure', 'GrLivArea', 'BsmtFinType1',
        'KitchenQual', 'BsmtFinSF1', 'Exterior1st', '2ndFlrSF', 'GarageCars',
        'ScreenPorch', 'WoodDeckSF', 'BsmtFullBath', 'CentralAir', '1stFlrSF',
        'HalfBath', 'PoolArea', 'GarageYrBlt', 'MasVnrType', 'ExterQual',
        'KitchenAbvGr', 'FullBath', 'LotConfig', 'Foundation', 'LowQualFinSF',
        'BedroomAbvGr', 'BsmtFinSF2', 'Condition2', 'PoolQC'
    ]

if False:
    t.train_columns = [
        'OverallQual', 'TotalSF', 'Neighborhood', 'OverallCond', 'BsmtQual',
        'MSSubClass'
    ]

t.prepare()

#t.sanity()
t.save('tmp')

#t.df_train, t.df_test = t.do_dummify(t.df_train, t.df_test, False)

model0 = Lasso(alpha=0.0005, random_state=1)
model = make_pipeline(RobustScaler(), model0)


def train(x, y):
    model.fit(x, y)


def predict(x):
    return model.predict(x)


rmse_score = custom_score_using_kfolds(
    train,
    predict,
    rmse,
    np.array(t.df_train.values),
    np.array(t.labels.values),
    scale=True,
    doShuffle=True)

print 'KFold RMSE:', rmse_score

cod_score = custom_score_using_kfolds(
    train,
    predict,
    cod,
    np.array(t.df_train.values),
    np.array(t.labels.values),
    scale=True,
    doShuffle=True)

print 'COD:', cod_score

#print 'weigths',model0.coef_
#print 'bias', model0.intercept_

sys.exit(0)
print 'RMSE', test_accuracy_rmsle(model, t.df_train, t.labels)

df_predicted = pd.DataFrame(columns=['Id', 'SalePrice'])
df_predicted['Id'] = t.test_ids
df_predicted.set_index('Id')
df_predicted['SalePrice'] = np.exp(model.predict(t.df_test))
df_predicted.to_csv('predicted.csv', sep=',', index=False)

print 'predictions done.'