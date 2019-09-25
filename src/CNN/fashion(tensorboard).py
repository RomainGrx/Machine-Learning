# -*- coding: utf-8 -*-
import gym 
import tensorflow as tf  
from collections import deque
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/Users/romaingraux/Documents/Python/Machine-Learning/res/prog')
import useful as u


tensorboard_path = '/Users/romaingraux/Documents/Python/Machine-Learning/res/tensorboard/fashion/run2/'
categories = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images, test_images = train_images.reshape(train_images.shape[0], 28, 28, 1), test_images.reshape(test_images.shape[0], 28, 28, 1)


def Convolution_activate(input, neurons_in, neurons_out, filter_height = 5, filter_width = 5, name = 'conv'):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([filter_height, filter_width, neurons_in, neurons_out], stddev=0.1), name = 'W')
        b = tf.Variable(tf.constant(0.1, shape = [neurons_out]), name = 'B')
        conv = tf.nn.conv2d(input, w, strides = [1,1,1,1], padding = 'SAME')
        act = tf.nn.relu(conv + b)
        tf.summary.histogram('weights', w)
        tf.summary.histogram('biases', b)
        tf.summary.histogram('activations', act)
        pool = tf.nn.max_pool(act, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
    return pool

def Fully_connected(input, neurons_in, neurons_out, name = 'fully', activation = None):
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
        tf.summary.histogram('weights', w)
        tf.summary.histogram('biases', b)
        tf.summary.histogram('activations', act)
    return act


tf.reset_default_graph()
tf.keras.backend.clear_session()
sess = tf.Session()

x = tf.placeholder(tf.float32, shape=[None, 28, 28], name = 'x')
y = tf.placeholder(tf.uint8, shape=[None, 10], name = 'labels')
x_reshape = tf.reshape(x, [-1, 28, 28, 1])
tf.summary.image('input', x_reshape, 30)

conv1 = Convolution_activate(x_reshape, 1, 32, name = 'conv1')
conv2 = Convolution_activate(conv1, 32, 64, name = 'conv2')
flatten = tf.reshape(conv2, [-1, 7*7*64])
fc1 = Fully_connected(flatten, 7*7*64, 1024, name = 'fully1', activation='relu')
fc2 = Fully_connected(fc1, 1024, 512, name = 'fully2', activation='relu')
fc3 = Fully_connected(fc1, 1024, 256, name = 'fully3', activation='relu')
logits = Fully_connected(fc3, 256, 10, name = 'fully2')
tf.summary.histogram('Fully connected 1', fc1)
tf.summary.histogram('Fully connected 2', fc2)
tf.summary.histogram('Fully connected 2', fc3)
tf.summary.histogram('Logits', logits)

train_images = train_images.reshape(60000, 28, 28)
labels = np.zeros((60000, 10))
for i in range(60000):
    labels[i,train_labels[i]] = 1
train_labels = labels
with tf.name_scope('loss'):
    loss = tf.math.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
    tf.summary.scalar('loss', loss)
with tf.name_scope('optimizer'):
    Adam = tf.train.AdamOptimizer(1e-4).minimize(loss)
    
with tf.name_scope('accuracy'):
    correct = tf.equal(tf.arg_max(logits, 1), tf.arg_max(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

merged_summary = tf.summary.merge_all()
sess.run(tf.global_variables_initializer())
writer = tf.summary.FileWriter(tensorboard_path)
writer.add_graph(sess.graph)
config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer, config)


run = 0
acc = 0
for i in range(2000):
    index = np.random.choice(np.arange(60000),
                                size = 100,
                                replace = False)
    batch_images = [train_images[i] for i in index]
    batch_labels = [train_labels[i] for i in index]
    
    sess.run(Adam, feed_dict={x : batch_images,
                                  y : batch_labels})
    if i % 5 == 0:
        [acc, s, lo] = sess.run([accuracy, merged_summary, loss], feed_dict={x : batch_images,
                                                                    y : batch_labels})
        writer.add_summary(s, i)
    
        print('Episode {} Accuracy {} Loss {}'.format(i, acc, lo))


