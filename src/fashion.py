#
# Predict pictures in a ten categories array
#
# Romain Graux 
# May 24, 2019
#
# -------------------------------------------------------------------------

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

categories = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

model = keras.models.load_model('../res/models/fashion.h5')
model.summary()

predictions = model.predict(test_images)

category_predictions = np.argmax(predictions, axis = 1)
category_accuracies  = np.max(predictions, axis = 1)

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i])
    plt.xlabel("{} {:2.0f}%".format(categories[category_predictions[i]],
                                100*category_accuracies[i]))
plt.show()

