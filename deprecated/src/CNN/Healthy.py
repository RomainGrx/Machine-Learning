#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 09:54:21 2019

@author: romaingraux
"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model

# In[1]
# Training data
Categorie = np.array(['malade','bonne santé'])
Healthy = 3 * np.random.rand(10,2) + 2
Illy = 3 * np.random.rand(10,2) 
Training = np.vstack((Illy, Healthy))
Target = np.append(np.zeros(10),np.ones(10)).reshape((20,1))


# In[2]
# Représentation graphique 
plt.plot(Healthy[:,0],Healthy[:,1], 'go')
plt.plot(Illy[:,0],Illy[:,1], 'ro')
plt.show()

# In[3]
# Déclaration du modèle

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(2, ), dtype = 'float32', name = 'main_input'),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(2, activation=tf.nn.softmax)
])

model.compile(optimizer='rmsprop', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# In[4]
# Training du modèle

model.fit(Training, Target, epochs=100)
test_loss, test_acc = model.evaluate(Training, Target)

# In[5]
# Prédiction 

prediction = model.predict(Training)
print (Categorie[np.argmax(prediction, axis = 1)])

# In[6]
# Graphique et prédiction

Case = np.random.rand(10,2) * 3 + 1
Pred = model.predict(Case)
Cat  = Categorie[np.argmax(Pred, axis = 1)]
fig  = plt.figure()
plt.plot(Case[:,0],Case[:,1], 'o', color = 'purple')
i = 0
for a,b in zip(Case[:,0], Case[:,1]): 
    plt.text(a, b, str(Cat[i]))
    i += 1
plt.show()

# In[7]

test = np.array([[1, 1],
                [4, 2]])
predict = model.predict(test)
print(Categorie[np.argmax(predict, axis = 1)])






