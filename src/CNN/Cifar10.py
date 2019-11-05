#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 20:15:56 2019

@author: romaingraux
"""

# In[0]
# Initialisation générale

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
import sys
sys.path.insert(0, '../../res/prog')
import useful as u

data = keras.datasets.cifar10
(train_images, train_labels), (test_images, test_labels) = data.load_data()
img_cols, img_rows, channels = 32, 32, 3
train_images, test_images = train_images.reshape(train_images.shape[0], img_cols, img_rows, channels)/255, test_images.reshape(test_images.shape[0], img_cols, img_rows, channels)/255
input_shape = (img_cols, img_rows, channels)

modelpath = "../../res/models/cifar10.h5"
categories = np.array(['Airplane','Automobile','Bird','Cat','Deer','Dog','Frog','Horse','Ship','Truck'])

# In[1]
# Initialisation et entrainement du modèle + save

model = keras.Sequential([
        keras.layers.Conv2D(16, input_shape = input_shape, kernel_size=(3, 3), activation = 'relu', padding='same'),
#        keras.layers.Conv2D(16, (3, 3), padding='same'),
        keras.layers.MaxPooling2D(pool_size = (2,2)),
        keras.layers.Conv2D(32, (3, 3), padding='same'),
#        keras.layers.Conv2D(32, (3, 3), padding='same'),
        keras.layers.MaxPooling2D(pool_size = (2,2)),
#        keras.layers.Conv2D(64, (3, 3), padding='same'),
#        keras.layers.Conv2D(64, (3, 3), padding='same'),
#        keras.layers.MaxPooling2D(pool_size = (2,2)),
#        keras.layers.Conv2D(128, (3, 3), padding='same'),
#        keras.layers.Conv2D(128, (3, 3), padding='same'),
#        keras.layers.MaxPooling2D(pool_size = (2,2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation=tf.nn.relu),
#        keras.layers.Dropout(0.5),
        keras.layers.Dense(64, activation=tf.nn.relu),
#        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation=tf.nn.sigmoid)
        ])

model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=4)
test_loss, test_acc = model.evaluate(train_images, train_labels)

u.labeled_modelsave(modelpath, model, train_images, train_labels)

# In[2]
# Prédictions et affichage des résultats

category_predictions, category_accuracies  = u.prediction(modelpath, test_images, categories)
plt.figure(figsize=(25,25))
shift = 10
for n in range(10):
    plt.subplot(5,5,n+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[shift+n])
    plt.xlabel("{} {:2.0f}%\n{}{}".format(category_predictions[shift+n],
                                100*category_accuracies[shift+n],
                                'REAL : ',
                                categories[test_labels[shift+n][0]]))

# In[3]
# Practice
