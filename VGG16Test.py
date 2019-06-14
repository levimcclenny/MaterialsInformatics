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
from keras.applications.vgg16 import VGG16
#from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
import tensorflow as tf
from progress.bar import Bar
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
keras.backend.set_session(sess)


img_new_name = []
processed_imgs = []
#dir = os.path.dirname(os.path.realpath('__file__'))
paths = ['images/Unknown/']
#path = os.path.join(dir, 'micrographs/')
#path = "../micrographs/micrograph"
valid_images = [".tif",".png",".jpg"]
for k in range(len(paths)):
    imgs = []
    img_name = []
    for f in os.listdir(paths[k]):
        img_name.append(f)

    #labels = []
    for f in img_name:
        print(f)
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        # use PIL to get all uint8 type images, then take the center and convert to float64, 
        # since apparently matpltlib doesnt care about float32 vs uint8 
        img = np.asarray(Image.open(os.path.join(paths[k],f)).convert('RGB'))
        imgs.append(img)

#    l_idx = np.searchsorted(img_name, data['path'])

#    image_set = np.asarray(imgs)
#    imgs_ordered = image_set[l_idx]
    #bar = Bar('Processing', len(imgs_ordered))
    print('Generating Images\n')
    for i in range(len(imgs)):
        patches = image.extract_patches_2d(imgs[i], (224,224), max_patches = 10)
        for patch in patches:
            x = np.expand_dims(patch, axis = 0)
            processed_imgs.append(preprocess_input(x)[0,:,:,:])



# lb = LabelBinarizer()
# lb.fit(np.asarray(all_labels))
print('\nLabels Binarized, converting array')


input = np.asarray(processed_imgs)
 
input_shape = (224, 224, 1)

model = VGG16(weights='results/Vahid_VGG16_Weights.h5', classes=2)

yhat = model.predict(input)

print(np.hstack(img_name,yhat))
