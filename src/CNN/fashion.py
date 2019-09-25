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
import sys
sys.path.insert(0, '/Users/romaingraux/Documents/Python/Machine-Learning/res/prog')
import useful as u

modelpath = "/Users/romaingraux/Documents/Python/Machine-Learning/res/models/fashion.h5"
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

categories = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images, test_images = train_images.reshape(train_images.shape[0], 28, 28, 1)/255, test_images.reshape(test_images.shape[0], 28, 28, 1)/255

# In[1]
model = keras.Sequential([
    keras.layers.Conv2D(32, input_shape = (28, 28, 1), kernel_size=(3, 3), activation = 'relu', padding='same'), 
    keras.layers.MaxPooling2D(pool_size = (2,2)),      
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs = 1)
test_loss, test_acc = model.evaluate(train_images, train_labels)

u.labeled_modelsave(modelpath, model, train_images, train_labels)


# In[2]

predictions = model.predict(test_images)
category_predictions = np.argmax(predictions, axis = 1)
category_accuracies  = np.max(predictions, axis = 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28)

#shift = np.random.randint(100)
shift = 0
plt.figure(figsize=(10,10))
for _ in range(10):
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(test_images[shift+i])
        plt.xlabel("{} {:2.0f}%".format(categories[category_predictions[shift+i]],
                                    100*category_accuracies[shift+i]))
    plt.show()
    shift += 25
    txt = input()
    if(txt == 'q'):
        break