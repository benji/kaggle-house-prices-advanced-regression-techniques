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

best_cost = 2.22143e+08
learning_rate = 0.3
training_epochs = 10

train = t.df_train.copy()
allcols = list(t.schema['columns'])

for col in allcols:
    if t.schema['columns'][col]['type'] == 'NUMERIC':
        print 'Testing col', col
        newcol = col + '_zero'

        train[newcol] = (train[col] == 0) * 1
        t.schema['columns'][newcol] = {'type': 'NUMERIC'}

        train_X = train.values
        train_Y = np.array(t.labels.values).flatten()

        scaler = StandardScaler()
        scaler.fit(train)
        train_X = scaler.transform(train_X)

        n_samples = train.shape[0]
        n_dims = train.shape[1]

        X = tf.placeholder("float", shape=(None, n_dims))
        Y = tf.placeholder("float", shape=(None))

        W = tf.Variable(tf.ones([1, n_dims]), name="weight")
        b = tf.Variable(tf.zeros([1, n_dims]), name="bias")

        pred = tf.reduce_sum(tf.add(tf.multiply(X, W), b), 1)
        cost = tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * n_samples)
        avgdiff = tf.reduce_sum(tf.abs(pred - Y)) / n_samples

        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(
            cost)

        init = tf.global_variables_initializer()

        training_cost = 0
        with tf.Session() as sess:
            sess.run(init)

            for epoch in range(training_epochs):
                print 'Epoch', epoch

                for i in range(len(train_X)):
                    x = [train_X[i]]
                    y = [train_Y[i]]
                    sess.run(optimizer, feed_dict={X: x, Y: y})

                print 'cost all', sess.run(
                    cost, feed_dict={
                        X: train_X,
                        Y: train_Y
                    })

                print 'avg diff', sess.run(
                    avgdiff, feed_dict={
                        X: train_X,
                        Y: train_Y
                    })

            print("Optimization Finished!")
            training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
            print("Training cost=", training_cost, '\n')

        if training_cost < best_cost:
            print '============================================FOUND GOOD COL', newcol
        
        train.drop(newcol, 1, inplace=True)

print train.columns