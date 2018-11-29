#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 19:19:01 2018

@author: levimcclenny
"""

import matplotlib.pyplot as plt
from ksvd import ApproximateKSVD
from sklearn.feature_extraction import image
import numpy as np
from sklearn.linear_model import orthogonal_mp_gram
from sklearn.svm import LinearSVC


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



# utils functions
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


def getReducedDict(img_chunk, n_atoms ):
    gamma, dict2 = get_decomp(img_chunk)
    counts = np.count_nonzero(gamma, axis = 0)
    reducedDict = counts.argsort()[-n_atoms:][::-1]
    return reducedDict
    


spher_dict = getDict(center2)
PS_dict = getDict(center4)
net_dict = getDict(center9)
mart_dict = getDict(center20)
SW_dict = getDict(center26)
pear_dict = getDict(center773)
PW_dict = getDict(center911)

# make an array for looping later
dictionaries = [spher_dict, PS_dict, net_dict, mart_dict, SW_dict, pear_dict, PW_dict]

## Test images

spher_test_img = img_array2[0:steph, 0:stepv]


patches = image.extract_patches_2d(spher_test_img, (16,16))
signals = patches.reshape(patches.shape[0], -1)
aksvd = ApproximateKSVD(n_components=64, max_iter = 100, transform_n_nonzero_coefs=4)
dictionary = aksvd.fit(signals[np.random.choice(range(signals.shape[0]), 1000, replace = True)]).components_
gamma = aksvd.transform(signals)

plt.imshow(image.reconstruct_from_patches_2d(
gamma.dot(dictionary).reshape((29200, 16, 16)), (161, 215)))

#generate test images and target vector at the same time
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
        





gamma, dict2 = get_decomp(center2)
counts = np.count_nonzero(gamma, axis = 0)
reducedDict = counts.argsort()[-10:][::-1]
show_dict(dict2, 8,8, save = True, filename = 'try2')

newdict = dict2[try2, :]


X = signals[0]
D = newdict

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

error_64_redo = []
for i in range(5):
    print i
    dictionaries = train()
    test_class = []
    for img in image_dict:
        test_class.append(test(img))
        
    error_64_redo.append(np.sum(np.array(test_class) == np.array(targets))/len(image_dict))
    
image_64 = img_array / 255


center_plot = image_64[213:426, 213:426]
plt.imshow(center_plot, cmap = 'gray')
#plt.savefig('M15.png')
plt.savefig('original.png')


img = center2


patches = image.extract_patches_2d(center2, (16,16))

signals = patches.reshape(patches.shape[0], -1)

aksvd = ApproximateKSVD(n_components=, max_iter = 100, transform_n_nonzero_coefs=5)
dictionary = aksvd.fit(signals[:1000]).components_
gamma = aksvd.transform(signals)
aksvd = ApproximateKSVD(n_components=64, max_iter = 100, transform_n_nonzero_coefs=4)
aksvd.fit()
gamma = aksvd.transform(signals[np.random.choice(range(signals.shape[0]), 10000, replace = True)])

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


    
    
show_dict(dict2, 8,8, save = True, filename = 'test')
fig.savefig('M15_100iter_10000Samples_C5.png')

purt = gamma + np.random.randn(gamma.shape[0], gamma.shape[1])
purt[np.where(gamma == 0)] = 0
gamma = aksvd.transform(signals)
rem = gamma
sel = np.random.choice(rem.shape[0], round(.8*rem.shape[0]), replace = False)
rem[[sel]] = 0
#sub = rem[np.random.choice(gamma.shape[0], round(.1*gamma.shape[0]), replace = False)]
#sub[[1, 5, 7]] = 0
reduced = rem.dot(dictionary)   
reduced_img = image.reconstruct_from_patches_2d(
    reduced.reshape(patches.shape), img.shape)
plt.imshow(reduced_img, cmap = 'gray')
plt.savefig('reconstructed_80_percent_removed.png')

gamma = aksvd.transform(signals)
shuffle = gamma
np.random.shuffle(shuffle)
reduced = purt.dot(dictionary)
reduced_img = image.reconstruct_from_patches_2d(
    reduced.reshape(patches.shape), img.shape)
plt.imshow(reduced_img, cmap = 'gray')
plt.savefig('shuffle.png')


def get_features(img, dictionary):
    patches = image.extract_patches_2d(img, (16,16))
    signals = patches.reshape(patches.shape[0], -1)
    sample = signals[np.random.choice(range(signals.shape[0]), 1000, replace = True)]
    sparse = []
    for signal in sample:
        X = signal
        D = dictionary
        gram = D.dot(D.T)
        Xy = D.dot(X.T)
        out = orthogonal_mp_gram(gram, Xy, n_nonzero_coefs=4).T
        sparse.append(out)
    return sparse
        


#train on spher and net
spher_dict = getDict(center2)
net_dict = getDict(center9)

train_dict = np.vstack((spher_dict, net_dict))

sparse_spher = get_features(center2, train_dict)
sparse_net = get_features(center9, train_dict)

y = np.array(([0]*1000, [1]*1000)).flatten()

X = np.vstack((sparse_spher, sparse_net))

clf = LinearSVC(random_state=0, tol=1e-5)
clf.fit(X, y)


z_dict = []
Z_SVM =[]
for img in image_dict:
    feat = get_features(img, train_dict)
    z_dict
    z_SVM.append(np.round(np.mean(clf.predict(feat))))
    




image_dict = []
targets = []
for i in range(3):
    for j in range(3):
        image_dict.append(img_array2[(i)*steph:(i+1)*steph, (j)*stepv:(j+1)*stepv])
        targets.append(0)
        image_dict.append(img_array9[(i)*steph:(i+1)*steph, (j)*stepv:(j+1)*stepv])
        targets.append(1)





