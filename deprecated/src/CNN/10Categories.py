# -------------------------------------------------------------------------
#
# Predict pictures of transport vehicles
#
# Romain Graux
# May 24, 2019
#
# -------------------------------------------------------------------------
# In[1]
# Initialisation générale

data = "../../res/datasets/transport-datasets/data_batch_1"
test_data = "../../res/datasets/transport-datasets/test_batch"
metafile = "../../res/datasets/transport-datasets/batches.meta"
modelpath = "../../res/models/transport.h5"
input_shape = (32, 32, 3)
import _pickle as cPickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

unpickled_df = pd.read_pickle(data)
meta = pd.read_pickle(metafile)

# In[2]
# Traitement des photos

photos = unpickled_df['data'].reshape((10000, 32, 32, 3))
labels = unpickled_df['labels']
categories = np.array(meta['label_names'])
img_cols, img_rows, channels = 32, 32, 3
categories_size = 10
photos = photos.reshape(photos.shape[0], img_cols, img_rows, channels)/255


# In[3]
# Déclaration et entrainement du modèle

model = keras.Sequential([
    keras.layers.Conv2D(32, input_shape = input_shape, kernel_size=(3, 3), activation = 'relu', padding='same'),
    keras.layers.MaxPooling2D(pool_size = (2,2)),
    keras.layers.Conv2D(64, (3, 3), padding='same'),
    keras.layers.MaxPooling2D(pool_size = (2,2)),
#    keras.layers.Conv2D(128, (3, 3), padding='same'),
#    keras.layers.MaxPooling2D(pool_size = (2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation=tf.nn.relu),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(categories_size, activation=tf.nn.softmax)
])

model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(photos, labels, epochs=4)

test_loss, test_acc = model.evaluate(photos, labels)

txt1 = input("Want to know accuracy of the last saved model? [yes/no] ")
if(txt1 == "yes"):
    oldmodel = keras.models.load_model(modelpath)
    print('Old model : ')
    oldmodel.evaluate(photos,labels)
    print('New model : ')
    model.evaluate(photos[:],labels[:])
    txt2 = input("Want to save the new model? [yes/no] ")
    if(txt2 == "yes"):
        model.save(modelpath)
        print("Model saved")


# In[4]
# Prédictions des photos

model = keras.models.load_model(modelpath)
predictions = model.predict(photos)
category_predictions = np.argmax(predictions, axis = 1)
category_accuracies  = np.max(predictions, axis = 1)
plt.figure(figsize=(25,25))
shift = 10
for n in range(10):
    plt.subplot(5,5,n+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(photos[shift+n])
    plt.xlabel("{} {:2.0f}%\n{}{}".format(categories[category_predictions[shift+n]],
                                100*category_accuracies[shift+n],
                                'REAL : ',
                                categories[labels[shift+n]]))

# In[6]
