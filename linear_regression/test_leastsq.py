from scipy.optimize import leastsq
import matplotlib.pyplot as plt
import pandas as pd
import math
import json
import sys
import operator
from tqdm import tqdm
import numpy as np
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from utils.utils import *
from utils.Training import *
from utils.Autotrain import *
from utils.kaggle import *

#col = 'YearBuilt'
col = 'TotalBsmtSF'
#col = 'LotFrontage'

t = training()

t.dummify_at_init = False
t.dummify_drop_first = False
t.use_label_encoding = False
t.train_columns = [col]


# t.remove_outliers.append(935)

t.prepare()

df = t.df_train
labels = t.labels

#selected = df[col] < 300
#df = df[selected]
#labels = labels[selected]

#selected = df[col] != 0
#df = df[selected]
#labels = labels[selected]

selected = df[col] != 0
df = df[selected]
labels = labels[selected]


def main():
    x = df[col].values
    y = labels

    # transform
    #f2, coefs = generate_least_square_best_fit(x, y, 2)
    #x = f2(x)

    # fit
    f1, _ = generate_least_square_best_fit(x, y, 1)
    f2, coefs = generate_least_square_best_fit(x, y, 2)
    print coefs
    
    print np.flip(coefs,0)

    coefs2 = np.copy(coefs)
    coefs2 = np.flip(coefs2,0)
    coefs2[-1]=coefs2[-1]-12
    print np.roots(coefs2)

    # draw
    xs = np.linspace(x.min(), x.max(), 50)
    y1s = f1(xs)
    y2s = f2(xs)

    plt.plot(x, y, 'bo', xs, y1s, 'r-', xs, y2s, 'g-')
    plt.show()


if __name__ == "__main__":
    main()
