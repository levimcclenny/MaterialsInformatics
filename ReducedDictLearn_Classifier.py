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


def getDict(img_chunk):
    patches = image.extract_patches_2d(img_chunk, (16,16))
    signals = patches.reshape(patches.shape[0], -1)
    
    # Set K-SVD paprameters to maximize classification rate according to fannjiang
    aksvd = ApproximateKSVD(n_components=64, max_iter = 100, transform_n_nonzero_coefs=4)
    dictionary = aksvd.fit(signals[np.random.choice(range(signals.shape[0]), 1000, replace = True)]).components_
    #gamma = aksvd.transform(signals) 
    return dictionary

def get_decomp(input_image):
    patches = image.extract_patches_2d(input_image, (16,16))
    signals = patches.reshape(patches.shape[0], -1)
    aksvd = ApproximateKSVD(n_components=64, max_iter = 100, transform_n_nonzero_coefs=4)
    dictionary = aksvd.fit(signals[np.random.choice(range(signals.shape[0]), 1000, replace = True)]).components_
    gamma = aksvd.transform(signals)
    return gamma, dictionary


def getReducedDict(img_chunk, n_atoms):
    gamma, dictionary = get_decomp(img_chunk)
    counts = np.count_nonzero(gamma, axis = 0)
    reducedIndex = counts.argsort()[-n_atoms:][::-1]
    reducedDict = dictionary[reducedIndex, :]
    return reducedDict


def train(n_atoms):
    spher_dict = getReducedDict(center2, n_atoms)
    print('spher complete')
    PS_dict = getReducedDict(center4, n_atoms)
    print('PS complete')
    net_dict = getReducedDict(center9, n_atoms)
    print('net complete')
    mart_dict = getReducedDict(center20, n_atoms)
    print('mart complete')
    SW_dict = getReducedDict(center26, n_atoms)
    print('SW complete')
    pear_dict = getReducedDict(center773, n_atoms)
    print('pear complete')
    PW_dict = getReducedDict(center911, n_atoms)
    print('PW complete')
    # make an array for looping later
    dictionary_array = [spher_dict, PS_dict, net_dict, mart_dict, SW_dict, pear_dict, PW_dict]
    return dictionary_array


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


def show_dict(dictionary, hsize, vsize, save = False, filename = 'file.png'):
    out=[]
    for i in range(dictionary.shape[0]):
        out.append(dictionary[i].reshape(16,16))
    fig=plt.figure(figsize=(16, 16))
    columns = vsize
    rows = hsize
    for i in range(1, columns*rows +1):
        img = out[i-1]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img, cmap = 'gray')   
    plt.show()
    if save :
        fig.savefig(filename)


# pull in
#MG2 - spherodite
#MG4 - pearlite+spheroidite
#MG9 - network
#MG20 - martensite
#MG26 - spheroidite+widmanstatten
#MG773 - pearlite
#MG911 - pearlite+widmanstatten


# pull in images
img_array2 = plt.imread('/Users/levimcclenny/Desktop/DictLearn/micrographs/micrograph2.tif')
img_array4 = plt.imread('/Users/levimcclenny/Desktop/Dictlearn/micrographs/micrograph4.tif')
img_array9 = plt.imread('/Users/levimcclenny/Desktop/Dictlearn/micrographs/micrograph9.tif')
img_array20 = plt.imread('/Users/levimcclenny/Desktop/Dictlearn/micrographs/micrograph20.tif')
img_array26 = plt.imread('/Users/levimcclenny/Desktop/Dictlearn/micrographs/micrograph26.tif')
img_array773 = plt.imread('/Users/levimcclenny/Desktop/Dictlearn/micrographs/micrograph773.tif')
img_array911 = plt.imread('/Users/levimcclenny/Desktop/Dictlearn/micrographs/micrograph911.tif')

#remove info bar at bottom, convert to float64
img_array2 = img_array2[0:483, 0:645]/255
img_array4 = img_array4[0:483, 0:645]/255
img_array9 = img_array9[0:483, 0:645]/255
img_array20 = img_array20[0:483, 0:645]/255
img_array26 = img_array26[0:483, 0:645]/255
img_array773 = img_array773[0:483, 0:645]/255
img_array911 = img_array911[0:483, 0:645]/255

#generate step sizes for 3x3 image chunks
steph = img_array2.shape[0]//3
stepv = img_array2.shape[1]//3

#grab center images to train
center2 = img_array2[steph:2*steph, stepv:2*stepv]
center4 = img_array4[steph:2*steph, stepv:2*stepv]
center9 = img_array9[steph:2*steph, stepv:2*stepv]
center20 = img_array20[steph:2*steph, stepv:2*stepv]
center26 = img_array26[steph:2*steph, stepv:2*stepv]
center773 = img_array773[steph:2*steph, stepv:2*stepv]
center911 = img_array911[steph:2*steph, stepv:2*stepv]


image_dict = []
targets = []
for i in range(3):
    for j in range(3):
        image_dict.append(img_array2[(i)*steph:(i+1)*steph, (j)*stepv:(j+1)*stepv])
        targets.append(0)
        image_dict.append(img_array4[(i)*steph:(i+1)*steph, (j)*stepv:(j+1)*stepv])
        targets.append(1)
        image_dict.append(img_array9[(i)*steph:(i+1)*steph, (j)*stepv:(j+1)*stepv])
        targets.append(2)
        image_dict.append(img_array20[(i)*steph:(i+1)*steph, (j)*stepv:(j+1)*stepv])
        targets.append(3)
        image_dict.append(img_array26[(i)*steph:(i+1)*steph, (j)*stepv:(j+1)*stepv])
        targets.append(4)
        image_dict.append(img_array773[(i)*steph:(i+1)*steph, (j)*stepv:(j+1)*stepv])
        targets.append(5)
        image_dict.append(img_array911[(i)*steph:(i+1)*steph, (j)*stepv:(j+1)*stepv])
        targets.append(6)
        
   

     
# Get trained dictionaries
dictionaries = train(10)

test_class = []
for img in image_dict:
    test_class.append(test(img))
    
np.sum(np.array(test_class) == np.array(targets))/len(image_dict)

show_dict(dictionaries[0], 2,5, save = True, filename = 'spher.png')


        
        
        




