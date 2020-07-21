#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 02:37:49 2020

@author: wisnu
"""


import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model
import time
from keras import backend as K
import cv2


#Prediction Function
def predict(model1, file):

    img_size = 150
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (img_size, img_size))
    img = np.array(img) / 255
    img = img.reshape(-1, img_size, img_size, 1)
    predictions = model1.predict_classes(img)
    predictions = predictions.reshape(1,-1)[0]
    return predictions


if __name__ == "__predict__":
    predict()