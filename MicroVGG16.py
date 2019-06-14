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


#data = pd.read_csv("/home/Projects/Materials/micrograph.csv")





# for both folders
#     read in images 
#     assign labels
#     for each indifivual image
#         generate 10 images
#         assign labes to each image










img_new_name = []
processed_imgs = []
all_labels = []
#dir = os.path.dirname(os.path.realpath('__file__'))
paths = ['images/Precipitate/', 'images/Bicontinuous/']
label_options = [-1,1]
#path = os.path.join(dir, 'micrographs/')
#path = "../micrographs/micrograph"
valid_images = [".tif",".png",".jpg"]
for k in range(len(paths)):
    imgs = []
    img_name = []
    img_labels = []
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
        img_labels.append(label_options[k])

#    l_idx = np.searchsorted(img_name, data['path'])

#    image_set = np.asarray(imgs)
#    imgs_ordered = image_set[l_idx]
    #bar = Bar('Processing', len(imgs_ordered))
    print('Generating Images\n')
    for i in range(len(imgs)):
        patches = image.extract_patches_2d(imgs[i], (224,224), max_patches = 10)
        label = img_labels[i]
        for patch in patches:
            x = np.expand_dims(patch, axis = 0)
            processed_imgs.append(preprocess_input(x)[0,:,:,:])
            all_labels.append(label)
        progbar(i, (len(imgs)-1), 20)



# lb = LabelBinarizer()
# lb.fit(np.asarray(all_labels))
y = keras.utils.np_utils.to_categorical(all_labels)
print('\nLabels Binarized, converting array')


input = np.asarray(processed_imgs)


X_train, X_test, y_train, y_test = train_test_split(
    input, y, test_size=0.1, random_state=42)
 
input_shape = (224, 224, 1)

model = VGG16(weights=None, classes = 2)

model.summary()
model.compile(loss = 'categorical_crossentropy', optimizer = 'sgd', metrics = ['accuracy'])

model.fit(X_train, y_train, epochs = 5, batch_size = 32, validation_data=(X_test, y_test))
name = 'results/Vahid_VGG16_Weights'
score = model.evaluate(X_test, y_test)
model.save_weights(name+'.h5')
print('Test score:'+ score[0])
print('Test accuracy:'+ score[1])


file = open('VGG16.txt', 'w')
file.write('Test score:'+ str(score[0])+'\n')
file.write('Test accuracy:'+ str(score[1])+'\n')
file.write(str(times))
file.close()
