#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 17:15:56 2018

@author: levimcclenny
"""

import matplotlib.pyplot as plt
#from ksvd import ApproximateKSVD
from sklearn.feature_extraction import image
import numpy as np
from sklearn.linear_model import orthogonal_mp_gram
import os, os.path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from PIL import Image
#from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
import tensorflow as tf



data = pd.read_csv("/Users/levimcclenny/Desktop/Materials Informatics//classifiers/DictLearn/micrograph.csv")

imgs = []
paths = []
img_name = []
img_new_name = []
processed_imgs = []
dir = os.path.dirname(os.path.realpath('__file__'))
path = os.path.join(dir, 'micrographs/')
#path = os.path.join(dir, 'micrographs/')
#path = "../micrographs/micrograph"
valid_images = [".tif",".png"]

for f in os.listdir(path):
    img_name.append(f)

img_name.sort()
for f in img_name:
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    # use PIL to get all uint8 type images, then take the center and convert to float64, 
    # since apparently matpltlib doesnt care about float32 vs uint8 
    img = np.asarray(Image.open(os.path.join(path,f)))
    #paths.append(os.path.join(path,f))
    #img_new_name.append(f)
    x = img[0:224, 0:224]
    #img = image.load_img(os.path.join(path,f))
    #x = image.img_to_array(img)
    x = np.expand_dims(x, axis=2)
    #x = preprocess_input(x)
    #subtract mean for VGG preprocessing
    processed_imgs.append(x-np.mean(x))

l_idx = np.searchsorted(img_name, data['path'])

image_set = np.asarray(processed_imgs)
imgs_ordered = image_set[l_idx]

lb = LabelBinarizer()
lb.fit(np.asarray(data['primary_microconstituent']))
y = lb.transform(np.asarray(data['primary_microconstituent']))


X_train, X_test, y_train, y_test = train_test_split(
    imgs_ordered, y, test_size=0.1, random_state=42)


 
input_shape = (224, 224, 1)

model = Sequential([
Conv2D(64, (3, 3), input_shape=input_shape, padding='same',
activation='relu'),
Conv2D(64, (3, 3), activation='relu', padding='same'),
MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
Conv2D(128, (3, 3), activation='relu', padding='same'),
Conv2D(128, (3, 3), activation='relu', padding='same',),
MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
Conv2D(256, (3, 3), activation='relu', padding='same',),
Conv2D(256, (3, 3), activation='relu', padding='same',),
Conv2D(256, (3, 3), activation='relu', padding='same',),
MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
Conv2D(512, (3, 3), activation='relu', padding='same',),
Conv2D(512, (3, 3), activation='relu', padding='same',),
Conv2D(512, (3, 3), activation='relu', padding='same',),
MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
Conv2D(512, (3, 3), activation='relu', padding='same',),
Conv2D(512, (3, 3), activation='relu', padding='same',),
Conv2D(512, (3, 3), activation='relu', padding='same',),
MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
Flatten(),
Dense(4096, activation='relu'),
Dense(4096, activation='relu'),
Dense(7, activation='softmax')
])

model.summary()
model.compile(loss = 'categorical_crossentropy', optimizer = 'sgd', metrics = ['accuracy'])

model.fit(X_train, y_train, epochs = 5, batch_size = 32)

score = model.evaluate(X_test, y_test)
