# kaggle score: 0.15952

import tensorflow as tf
import pandas as pd
import math, json, sys, os
from tqdm import tqdm
import numpy as np
import keras
from os import path
import matplotlib.pyplot as plt
rng = np.random
from sklearn.preprocessing import StandardScaler

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from utils.utils import *
from utils.kaggle import *
from utils.Training import *

t = training()
t.logify_columns = []

t.dummify_at_init = True
t.dummify_drop_first = False
t.use_label_encoding = True

t.prepare()

train_X = t.df_train.values
train_Y = np.array(t.labels.values).flatten()

scaler = StandardScaler()
scaler.fit(t.df_train)
train_X = scaler.transform(train_X)

n_samples = t.df_train.shape[0]
n_dims = t.df_train.shape[1]

X = tf.placeholder("float", shape=(None, n_dims))
Y = tf.placeholder("float", shape=(None))

W = tf.Variable(tf.ones([1, n_dims]), name="weight")
b = tf.Variable(tf.zeros([1, n_dims]), name="bias")

pred = tf.reduce_sum(tf.add(tf.multiply(X, W), b), 1)
cost = tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * n_samples)
avgdiff = tf.reduce_sum(tf.abs(pred - Y)) / n_samples

learning_rate = 0.3
training_epochs = 10

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(training_epochs):
        print 'Epoch', epoch

        for i in range(len(train_X)):
            x = [train_X[i]]
            y = [train_Y[i]]
            sess.run(optimizer, feed_dict={X: x, Y: y})

        print 'cost all', sess.run(cost, feed_dict={X: train_X, Y: train_Y})

        print 'avg diff', sess.run(avgdiff, feed_dict={X: train_X, Y: train_Y})

    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print("Training cost=", training_cost, '\n')

    pred = sess.run(pred, feed_dict={X: scaler.transform(t.df_test.values)})

    df_predicted = pd.DataFrame(columns=['Id', 'SalePrice'])
    df_predicted['Id'] = t.test_ids
    df_predicted.set_index('Id')
    df_predicted['SalePrice'] = pred
    df_predicted.to_csv('predicted.csv', sep=',', index=False)

    print 'Predictions done.'