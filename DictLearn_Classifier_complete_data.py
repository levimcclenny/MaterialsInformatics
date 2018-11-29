#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 22:34:56 2018

@author: levimcclenny
"""

import matplotlib.pyplot as plt
from ksvd import ApproximateKSVD
from sklearn.feature_extraction import image
import numpy as np
from sklearn.linear_model import orthogonal_mp_gram
import os, os.path
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image


def getDict(signals, n):
    # Set K-SVD paprameters to maximize classification rate according to fannjiang
    print(n)
    aksvd = ApproximateKSVD(n_components=64, max_iter = 50, transform_n_nonzero_coefs=4)
    dictionary = aksvd.fit(signals[np.random.choice(range(signals.shape[0]), int(round(n)), replace = True)]).components_
    #gamma = aksvd.transform(signals) 
    return dictionary

def train_dict(img_batch, p):
    all_signals = np.empty((0, 256))
    print('getting signals')
    for img in img_batch:
        patches = image.extract_patches_2d(img, (16,16))
        signals = patches.reshape(patches.shape[0], -1)
        all_signals = np.concatenate([all_signals, signals])
    print('training dictionary')
    output_dict = getDict(all_signals, p*all_signals.shape[0])
    return output_dict

def test(test_img):
    patches = image.extract_patches_2d(test_img, (16,16))
    signals = patches.reshape(patches.shape[0], -1)
    sample = signals[np.random.choice(range(signals.shape[0]), 1000, replace = True)]
    dictionary_error = []
    for dictionary in dictionaries:
        sample_error = []
        for signal in sample:
            X = signal
            D = dictionary
            gram = D.dot(D.T)
            Xy = D.dot(X.T)
            out = orthogonal_mp_gram(gram, Xy, n_nonzero_coefs=4).T
            sample_error.append(np.linalg.norm(X - out.dot(D)))
        dictionary_error.append(sample_error)
    return np.argmin(np.mean(dictionary_error, axis = 1))



steph = 161
stepv = 215
imgs = []
dir = os.path.dirname(os.path.realpath('__file__'))
path = os.path.join(dir, 'micrographs/')
#path = "../micrographs/micrograph"
valid_images = [".tif",".png"]

for f in os.listdir(path):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    # use PIL to get all uint8 type images, then take the center and convert to float64, 
    # since apparently matpltlib doesnt care about float32 vs uint8 
    img = np.asarray(Image.open(os.path.join(path,f)))
    imgs.append(img[steph:2*steph, stepv:2*stepv]/255)
    
data = pd.read_csv("/Users/levimcclenny/Desktop/DictLearn/micrograph.csv")

#import preprocessed data with labels, labels are as follows:
# 1 = pearlite
# 2 = Spheroidite
# 3 = pearlite+spheroidite
# 4 = spheroidite+widmanstatten
# 5 = network
# 6 = martensite
# 7 = pearlite+widmanstatten


X_train, X_test, y_train, y_test = train_test_split(
    imgs, data['primary_microconstituent'], test_size=0.33, random_state=42)

train = np.asarray(X_train)
pear_train = train[np.where(y_train == 'pearlite')]
spher_train = train[np.where(y_train == 'spheroidite')]
PS_train = train[np.where(y_train == 'pearlite+spheroidite')]
SW_train = train[np.where(y_train == 'spheroidite+widmanstatten')]
net_train = train[np.where(y_train == 'network')]
mart_train = train[np.where(y_train == 'martensite')]
PW_train = train[np.where(y_train == 'pearlite+widmanstatten')]

pear_train_sub = pear_train[np.random.choice(range(pear_train.shape[0]), round(.5*pear_train.shape[0]), replace = True)]
spher_train_sub = spher_train[np.random.choice(range(spher_train.shape[0]), round(.5*spher_train.shape[0]), replace = True)]
PS_train_sub = PS_train[np.random.choice(range(PS_train.shape[0]), round(.5*PS_train.shape[0]), replace = True)]
SW_train_sub = SW_train[np.random.choice(range(SW_train.shape[0]), round(.5*SW_train.shape[0]), replace = True)]
net_train_sub = net_train[np.random.choice(range(net_train.shape[0]), round(.5*net_train.shape[0]), replace = True)]
mart_train_sub = mart_train[np.random.choice(range(mart_train.shape[0]), round(.5*mart_train.shape[0]), replace = True)]
PW_train_sub = PW_train[np.random.choice(range(PW_train.shape[0]), round(.5*PW_train.shape[0]), replace = True)]

     
pear_dict = train_dict(pear_train_sub, .05)
print('pear done')
spher_dict = train_dict(spher_train_sub, .05)
print('spher done')
PS_dict = train_dict(PS_train_sub, .05)
print('PS done')
SW_dict = train_dict(SW_train_sub, .05)
print('SW done')
net_dict = train_dict(net_train_sub, .05)
print('net done')
mart_dict = train_dict(mart_train_sub, .05)
print('mart done')
PW_dict = train_dict(PW_train_sub, .05)
print('PW done')

dictionaries = [spher_dict, PS_dict, net_dict, mart_dict, SW_dict, pear_dict, PW_dict]

targets = np.array(y_test)
targets[np.where(targets == 'spheroidite')] = 0
targets[np.where(targets == 'pearlite+spheroidite')] = 1
targets[np.where(targets == 'network')] = 2
targets[np.where(targets == 'martensite')] = 3
targets[np.where(targets == 'spheroidite+widmanstatten')] = 4
targets[np.where(targets == 'pearlite')] = 5
targets[np.where(targets == 'pearlite+widmanstatten')] = 6


# Get trained dictionaries
#dictionaries = train()

test_class = []
for img in X_test[0:60]:
    test_class.append(test(img))
    
np.sum(np.array(test_class) == np.array(targets[0:60]))/60