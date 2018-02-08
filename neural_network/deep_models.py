import pandas as pd
import math, json, sys, os
from tqdm import tqdm
import numpy as np
import keras
from os import path

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from utils.utils import *
from utils.Training import *
from utils.kaggle import *

pd.options.mode.chained_assignment = None  # default='warn'


def model1(ncols):
    model = Sequential()
    model.add(Dense(2 * ncols, input_dim=ncols, activation='linear'))
    model.add(Dense(ncols, activation='linear'))
    model.add(Dense(ncols / 2, activation='linear'))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='rmsprop')
    return model

def model2(ncols):
    model = Sequential()
    model.add(Dense(ncols, input_dim=ncols, activation='linear'))
    model.add(Dense(ncols, activation='linear'))
    model.add(Dense(ncols, activation='linear'))
    model.add(Dense(ncols, activation='linear'))
    model.add(Dense(ncols, activation='linear'))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='rmsprop')
    return model
