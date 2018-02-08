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
#t.train_columns = [
#    'TotalSF', 'OverallQual', 'YearBuilt', 'OverallCond', 'LotArea',
#    'BsmtFullBath', 'TotalBsmtSF', 'MoSold'
#]

t.dummify_at_init = True
t.dummify_drop_first = False
t.use_label_encoding = False

t.prepare()

# Training Data
train_X = t.df_train.values
train_Y = np.array(t.labels.values).flatten()

scaler = StandardScaler()
scaler.fit(t.df_train)
train_X = scaler.transform(train_X)

print train_X.shape
print train_Y.shape
#sys.exit(0)
print 'x shape', train_X.shape
print 'y shape', train_Y.shape

n_samples = t.df_train.shape[0]
n_dims = t.df_train.shape[1]

# tf Graph Input
X = tf.placeholder("float", shape=(None, n_dims))
Y = tf.placeholder("float", shape=(None))

# Set model weights
W = tf.Variable(tf.ones([1, n_dims]), name="weight")
b = tf.Variable(tf.zeros([1, n_dims]), name="bias")

# Construct a linear model
pred = tf.reduce_sum(tf.add(tf.multiply(X, W), b), 1)

# Mean squared error
cost = tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * n_samples)

avgdiff = tf.reduce_sum(tf.abs(pred - Y)) / n_samples

learning_rate = 1
training_epochs = 10

# Gradient descent
#  Note, minimize() knows to modify W and b because Variable objects are trainable=True by default
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Fit all training data
    for epoch in range(training_epochs):
        print epoch

        for i in range(len(train_X)):  #range(4):#
            #print 'W',sess.run(W)
            #print 'b',sess.run(b)
            x = [train_X[i]]
            y = [train_Y[i]]

            #print 'pred',sess.run(pred, feed_dict={X: x, Y: y})

            #print 'pred - Y',sess.run(pred - Y, feed_dict={X: x, Y: y})

            if i % 100000 == 0:
                if False:
                    print 'x', x
                    print 'y', y
                    print 'W', sess.run(W)
                    print 'b', sess.run(b)

                    print 'cost row', sess.run(cost, feed_dict={X: x, Y: y})

                    train_Y2 = train_Y  # [[j] for j in train_Y]

                    print 'cost all (train_Y2)', train_Y2
                    print 'cost all (pred)', sess.run(
                        pred, feed_dict={
                            X: train_X,
                            Y: train_Y2
                        })
                    print 'cost all (Y)', sess.run(
                        Y, feed_dict={
                            X: train_X,
                            Y: train_Y2
                        })
                    print 'cost all (pred - Y)', sess.run(
                        pred - Y, feed_dict={
                            X: train_X,
                            Y: train_Y2
                        })
                    print 'cost all (pow2)', sess.run(
                        tf.pow(pred - Y, 2),
                        feed_dict={
                            X: train_X,
                            Y: train_Y2
                        })

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
                print '================='

            sess.run(optimizer, feed_dict={X: x, Y: y})

        #sys.exit(0)
        # Display logs per epoch step
        #if (epoch + 1) % display_step == 0:

        #print train_Y[:3]

        train_Y2 = train_Y  #[[y] for y in train_Y]

        #print sess.run(pred - train_Y2)

        c = sess.run(cost, feed_dict={X: train_X, Y: train_Y2})
        print c
        #print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
        #    "W=", sess.run(W), "b=", sess.run(b))

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