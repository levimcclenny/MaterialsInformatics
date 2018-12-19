#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 19:42:26 2018

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
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF


# utils functions
def getDict(img_chunk, N_patches):
    patches = image.extract_patches_2d(img_chunk, (16,16))
    signals = patches.reshape(patches.shape[0], -1)  
    # Set K-SVD paprameters to maximize classification rate according to fannjiang
    aksvd = ApproximateKSVD(n_components=64, max_iter = 100, transform_n_nonzero_coefs=4)
    dictionary = aksvd.fit(signals[np.random.choice(range(signals.shape[0]), N_patches, replace = True)]).components_
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


def get_features(img, dictionary):
    patches = image.extract_patches_2d(img, (16,16))
    signals = patches.reshape(patches.shape[0], -1)
    sample = signals[np.random.choice(range(signals.shape[0]), 100, replace = True)]
    sparse = []
    for signal in sample:
        X = signal
        D = dictionary
        gram = D.dot(D.T)
        Xy = D.dot(X.T)
        out = orthogonal_mp_gram(gram, Xy, n_nonzero_coefs=4).T
        sparse.append(out)
    return sparse


data = pd.read_csv("micrograph.csv")



imgs = []
paths = []
img_name = []
img_new_name = []
dir = os.path.dirname(os.path.realpath('__file__'))
path = os.path.join(dir, 'micrographs/')
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
    paths.append(os.path.join(path,f))
    img_new_name.append(f)
    imgs.append(img[0:484, :]/255)

l_idx = np.searchsorted(img_name, data['path'])

image_set = np.asarray(imgs)
imgs_ordered = image_set[l_idx]


spher_index = np.where((data['micron_bar'] == 10.0) & (data['primary_microconstituent'] == 'spheroidite'))
pear_index = np.where((data['micron_bar'] == 10.0) & (data['primary_microconstituent'] == 'pearlite'))
sw_index = np.where((data['micron_bar'] == 10.0) & (data['primary_microconstituent'] == 'spheroidite+widmanstatten'))
net_index = np.where((data['micron_bar'] == 100.0) & (data['primary_microconstituent'] == 'network'))

spher_imgs = imgs_ordered[spher_index, :,:][0]
pear_imgs = imgs_ordered[pear_index, :,:][0]
sw_imgs = imgs_ordered[sw_index, :,:][0]
net_imgs = imgs_ordered[net_index, :,:][0]

steph = 161
stepv = 215
image_data = []
targets = []

for img in spher_imgs[0:20]:
    for i in range(3):
        for j in range(3):
            image_data.append(img[(i)*steph:(i+1)*steph, (j)*stepv:(j+1)*stepv])
            targets.append([0]*100)
            

for img in pear_imgs:
    for i in range(3):
        for j in range(3):
            image_data.append(img[(i)*steph:(i+1)*steph, (j)*stepv:(j+1)*stepv])
            targets.append([0]*100)
            
for img in sw_imgs:
    for i in range(3):
        for j in range(3):
            image_data.append(img[(i)*steph:(i+1)*steph, (j)*stepv:(j+1)*stepv])
            targets.append([1]*100)
            

for img in net_imgs:
    for i in range(3):
        for j in range(3):
            image_data.append(img[(i)*steph:(i+1)*steph, (j)*stepv:(j+1)*stepv])
            targets.append([0]*100)


X_train, X_test, y_train, y_test = train_test_split(
    image_data, targets, test_size=0.33, random_state=42)


total_img_set = np.append(spher_imgs[0:20], pear_imgs, axis = 0)


#MG2 - spherodite

#MG773 - pearlite



# pull in images
spher_dit_train_img = plt.imread('/Users/levimcclenny/Desktop/DictLearn/micrographs/micrograph2.tif')
pear_dit_train_img = plt.imread('/Users/levimcclenny/Desktop/Dictlearn/micrographs/micrograph773.tif')
sw_dit_train_img = plt.imread('/Users/levimcclenny/Desktop/Dictlearn/micrographs/micrograph862.tif')
net_dit_train_img = plt.imread('/Users/levimcclenny/Desktop/Dictlearn/micrographs/micrograph8.tif')


#remove info bar at bottom, convert to float64
spher_dit_train_img = spher_dit_train_img[0:483, 0:645]/255
pear_dit_train_img = pear_dit_train_img[0:483, 0:645]/255
sw_dit_train_img = sw_dit_train_img[0:483, 0:645]/255
net_dit_train_img = net_dit_train_img[0:483, 0:645]/255


#grab center images to train
spher_dit_train_img_center = spher_dit_train_img[steph:2*steph, stepv:2*stepv]
pear_dit_train_img_center = pear_dit_train_img[steph:2*steph, stepv:2*stepv]
sw_dit_train_img_center = sw_dit_train_img[steph:2*steph, stepv:2*stepv]
net_dit_train_img = net_dit_train_img[steph:2*steph, stepv:2*stepv]





spher_dict = getDict(spher_dit_train_img_center, 1000)
pear_dict = getDict(pear_dit_train_img_center, 1000)
sw_dict = getDict(sw_dit_train_img_center, 1000)
net_dict = getDict(net_dit_train_img, 1000)

comb_dict = np.vstack((pear_dict, sw_dict))

#X_train = [center2, center773]
train_feats = []
for img in X_train: 
    train_feats.append(get_features(img, comb_dict))
    
    

X = [item for sublist in train_feats for item in sublist]

y = [item for sublist in y_train for item in sublist]

#y = np.array(([0]*1000, [1]*1000)).flatten()



classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="rbf", C=0.025, probability=True),
    LinearSVC(random_state=0, tol=1e-5),
    NuSVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    MLPClassifier(),
    #GaussianProcessClassifier(1.0*RBF(1.0)),
    ]


for clf in classifiers:
    name = clf.__class__.__name__
      
    print("="*30)
    print("Training:")
    print(name)
    clf.fit(X, y) 
    print('****Results****')
    train_predictions =[]
    for img in X_test:
        feat = get_features(img, comb_dict)
        train_predictions.append(np.round(np.mean(clf.predict(feat))))
    
    acc = 1- sum(abs(np.mean(y_test, axis = 1) - train_predictions))/len(y_test)
    print("Accuracy: {:.4%}".format(acc))
    
# =============================================================================
#     train_predictions = clf.predict_proba(X_test)
#     ll = log_loss(y_test, train_predictions)
#     print("Log Loss: {}".format(ll))
#     
#     log_entry = pd.DataFrame([[name, acc*100, ll]], columns=log_cols)
#     log = log.append(log_entry)
# =============================================================================
    
print("="*30)






# =============================================================================
# #z_dict = []
# Z_SVM_1 =[]
# for img in X_test:
#     feat = get_features(img, comb_dict)
#     #z_dict
#     Z_SVM_1.append(np.round(np.mean(clf.predict(feat))))
# 1- sum(abs(np.mean(y_test, axis = 1) - Z_SVM_1))/len(y_test)
# =============================================================================
    
dictionaries = [pear_dict, sw_dict]  
Z_Dict_1 = []
for img in X_test:
    Z_Dict_1.append(test(img))
    
1- sum(abs(np.mean(y_test, axis = 1) - Z_Dict_1))/len(y_test)
    
