import math, json, sys, os
import pandas as pd
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from utils.utils import *
from utils.kaggle import *
from utils.Training import *

t = training()
t.train_columns = [
    'TotalSF', 'OverallQual', 'YearBuilt', 'OverallCond', 'LotArea',
    'Neighborhood', 'BsmtFullBath', 'TotalBsmtSF', 'MoSold'
]

t.prepare()

accuracy = test_accuracy(t.df_train, t.labels, passes=100) * 100
print 'Accuracy', accuracy

generate_predictions(t.labels, t.df_train, t.df_test, t.test_ids)
print "Predictions complete."