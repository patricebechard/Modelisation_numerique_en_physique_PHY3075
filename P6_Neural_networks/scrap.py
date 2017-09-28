#!/usr/bin/env python
import os				#no warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import numpy as np
import sys

filetraining = 'trainingset.txt'        #input file
nTraining = 1000

filetest = filetraining
nTest = 2000-nTraining

nIter = 10000                            #number of iterations
nNeurons = [13,10,1]                     #number of neurons by layer
nLayers = len(nNeurons)			#number of neuron layers
eta = 0.0001


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, w_h, w_o):
    h = tf.nn.sigmoid(tf.matmul(X, w_h)) # this is a basic mlp, think 2 stacked logistic regressions
    return tf.nn.sigmoid(tf.matmul(h, w_o)) # note that we dont take the softmax at the end because our cost fn does that for us

def init_set(f,nSample):
    """Initialize training set and answers for supervized learning"""
    tset = np.zeros([nSample,nNeurons[0]])            #training set empty array
    oset = np.zeros([nSample,nNeurons[-1]])           #answers for training set
    for i in range(nSample):
        temp = f.readline().strip().split()
        tset[i] = [float(elem) for elem in temp[:-1]]
        oset[i] = int(temp[-1])
    return tset,oset

f = open(filetraining)
f.readline()
train_x, train_y = init_set(f,nTraining)
test_x, test_y = init_set(f,nTest)

X = tf.placeholder("float", [None, nNeurons[0]])
Y = tf.placeholder("float", [None, nNeurons[-1]])

w_h = init_weights([nNeurons[0], nNeurons[1]]) # create symbolic variables
w_o = init_weights([nNeurons[1], nNeurons[2]])

py_x = model(X, w_h, w_o)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y)) # compute costs
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost) # construct an optimizer
predict_op =tf.less(py_x,tf.constant([0.5]))


# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.global_variables_initializer().run()

    for i in range(nIter):

        order = np.random.permutation(np.arange(nTraining))
        for j in order:
            x = np.array([train_x[j]]) ; y = np.array([train_y[j]])
            sess.run(train_op, feed_dict={X: x, Y: y})

        print(sess.run(py_x,feed_dict={X:train_x,Y:train_y}))
        print(i)
        #print(i, np.mean(test_y == sess.run(predict_op, feed_dict={X: test_x})))
        #print(np.mean(sess.run(predict_op, feed_dict={X: test_x})))
        #print(sess.run(cost,feed_dict={X: train_x,Y: train_y}))
