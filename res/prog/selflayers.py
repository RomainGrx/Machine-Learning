#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 16:23:42 2019

@author: romaingraux
"""

import tensorflow as tf  
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/Users/romaingraux/Documents/Python/Machine-Learning/res/prog')
import useful as u


def Convolution_activate(input, neurons_in, neurons_out, filter_height=5, filter_width=5, pool=True, padding='SAME', name='Convolution'):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([filter_height, filter_width, neurons_in, neurons_out], stddev=0.1), name = 'W')
        b = tf.Variable(tf.constant(0.1, shape = [neurons_out]), name = 'B')
        conv = tf.nn.conv2d(input, w, strides = [1,1,1,1], padding = padding)
        act = tf.nn.relu(conv + b)
        tf.summary.histogram('Weights', w)
        tf.summary.histogram('Biases', b)
        tf.summary.histogram('Activations', act)
        if pool:
            pooling = tf.nn.max_pool(act, ksize = [1,2,2,1], strides = [1,2,2,1], padding = padding) 
    return pooling

def Fully_connected(input, neurons_in, neurons_out, activation = None, name='Fully connected'):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([neurons_in, neurons_out], stddev=0.1), name = 'W')
        b = tf.Variable(tf.constant(0.1, shape = [neurons_out]), name = 'B')
        mul = tf.matmul(input, w) + b
        if activation == 'relu':
            act = tf.nn.relu(mul)
        elif activation == 'softmax' :
            act = tf.nn.softmax(mul)
        else : 
            act = mul
        tf.summary.histogram('Weights', w)
        tf.summary.histogram('Biases', b)
        tf.summary.histogram('Activations', act)
    return act