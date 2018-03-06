import time
import sys
import collections
from os import path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# map confidence interval desired in % -> z values
Zs = {
    80: 1.28,
    90: 1.645,
    95: 1.96,
    98: 2.33,
    99: 2.58
}
orderedZs = collections.OrderedDict(sorted(Zs.items(), key=lambda t: t[0]))

score_plus_or_minus_precision_desired = 0.001
standard_deviation = 0.0049341254653

for ci in orderedZs:
    z = Zs[ci]
    v = np.power(standard_deviation * z /
                 score_plus_or_minus_precision_desired, 2)
    min_samples = int(np.ceil(v))
    print 'For', ci, '% accuracy on a', score_plus_or_minus_precision_desired, 'confidence interval, need', min_samples, 'scores'
