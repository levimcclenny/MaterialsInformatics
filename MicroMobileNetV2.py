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
#from keras.preprocessing import image
from keras.applications.xception import preprocess_input
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.applications.mobilenet_v2 import MobileNetV2
import tensorflow as tf
#from progress.bar import Bar
import time

class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

def progbar(curr, total, full_progbar):
    frac = curr/total
    filled_progbar = round(frac*full_progbar)
    print('\r', '#'*filled_progbar + '-'*(full_progbar-filled_progbar), '[{:>7.2%}]'.format(frac), end='')

data = pd.read_csv("/home/Projects/Materials/micrograph.csv")

imgs = []
paths = []
img_name = []
img_new_name = []
processed_imgs = []
#dir = os.path.dirname(os.path.realpath('__file__'))
path = '/home/Projects/Materials/micrographs/'
#path = os.path.join(dir, 'micrographs/')
#path = "../micrographs/micrograph"
valid_images = [".tif",".png"]

for f in os.listdir(path):
    img_name.append(f)

img_name.sort()



labels = []
for f in img_name:
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    # use PIL to get all uint8 type images, then take the center and convert to float64, 
    # since apparently matpltlib doesnt care about float32 vs uint8 
    img = np.asarray(Image.open(os.path.join(path,f)))
    imgs.append(img[0:483, 0:645])
    

l_idx = np.searchsorted(img_name, data['path'])

image_set = np.asarray(imgs)
imgs_ordered = image_set[l_idx]

processed_imgs = []
labels = []
#bar = Bar('Processing', len(imgs_ordered))
print('Generating Images\n')
for i in range(len(imgs_ordered)):
    patches = image.extract_patches_2d(imgs_ordered[i], (224,224), max_patches = 100)
    label = data['primary_microconstituent'][i]
    for patch in patches:
        x = Image.fromarray(patch).convert('RGB')
        x = np.asarray(x)
        #x = np.expand_dims(patch, axis = 2)
        x = preprocess_input(x)
        processed_imgs.append(x)
        labels.append(label)
    progbar(i, (len(imgs_ordered)-1), 20)



lb = LabelBinarizer()
lb.fit(np.asarray(data['primary_microconstituent']))
y = lb.transform(labels)
print('\nLabels Binarized, converting array')


input = np.asarray(processed_imgs)

X_train, X_test, y_train, y_test = train_test_split(
    input, y, test_size=0.1, random_state=42)


model = MobileNetV2(weights=None, classes = 7)

model.summary()
model.compile(loss = 'categorical_crossentropy', optimizer = 'sgd', metrics = ['accuracy'])
time_callback = TimeHistory()
model.fit(X_train, y_train, epochs = 5, batch_size = 32, validation_data=(X_test, y_test), callbacks=[time_callback])
name = 'results/UHCS_MobileNetV2_Weights'
score = model.evaluate(X_test, y_test)
print('Test score:', score[0])
print('Test accuracy:', score[1])
model.save_weights(name+'.h5')

file = open('MobileNetV2.txt', 'w')
file.write('Test score:'+ str(score[0])+'\n')
file.write('Test accuracy:'+ str(score[1])+'\n')
file.write(str(times))
file.close()
