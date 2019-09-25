#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 10:54:19 2019

@author: romaingraux

Program with differents useful functions.
"""

from termcolor import colored
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np


# --- Text coloring ---

def newprint(txt):
    print(colored('->', 'red'), txt)

# --- Model ---
    
def modelsave(modelpath, model):
    txt2 = input("Want to save the new model? [yes/no] ")
    if(txt2 == "yes"):
        keras.models.save_model(model, modelpath)
        print("Model saved")
        return
    print("Model not saved")
    
def loadmodel(modelpath, model = None):
    try :
        txt2 = input("Want to load the last model? [yes/no] ")
        if(txt2 == "yes"):
            model = keras.models.load_model(modelpath)
            print("Model loaded")
            return model
        print("Model not loaded")
        return model
    except :
        print("Can not be loaded")
        return model
    
def labeled_modelsave(modelpath, newmodel, files, labels):
    txt1 = input("Want to know accuracy of the last saved model? [yes/no] ")
    if(txt1 == "yes"):
        oldmodel = keras.models.load_model(modelpath)
        print('Saved model : ')
        oldmodel.evaluate(files,labels)
        print('New model : ')
        newmodel.evaluate(files,labels)
        txt2 = input("Want to save the new model? [yes/no] ")
        if(txt2 == "yes"):
            newmodel.save(modelpath)
            print("Model saved")
            return
        print("Model not saved")
   
         
def prediction(modelpath, files, categories):
    model = keras.models.load_model(modelpath)
    predictions = model.predict(files)
    category_predictions = categories[np.argmax(predictions, axis = 1)]
    category_accuracies  = np.max(predictions, axis = 1)
    return category_predictions, category_accuracies