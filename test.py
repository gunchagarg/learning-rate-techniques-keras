import sys
import os
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.optimizers import Adam, SGD
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Activation, GlobalAveragePooling2D, BatchNormalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras import callbacks
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.datasets import cifar10
from keras import backend as K
import pandas as pd
import numpy as np
import glob
import h5py
import cv2
import matplotlib.pyplot as plt
%matplotlib inline

# Importing additional functionalities
from dlr_implementation import Adam_dlr
from clr_implementation import *

batch_size = 8
num_classes = 10
epochs = 5

# input image dimensions
img_rows, img_cols = 32, 32

# Specifying the layer names at which split is to be made
model_split_1 = 'res4a_branch2a'
model_split_2 = 'fc_start'

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

base_model = ResNet50(include_top=False, weights='imagenet')
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu', name = 'fc_start')(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
out = Dense(num_classes, activation='softmax', name = 'output')(x)  #num_classes: number of classes in the dataset

model = Model(inputs=base_model.input, outputs=[out])

# Extracting layers at which split is made
split_layer_1 = [layer for layer in model.layers if layer.name == model_split_1][0]
split_layer_2 = [layer for layer in model.layers if layer.name == model_split_2][0]

# Implementing Differential Learning
opt = Adam_dlr(split_1 = split_layer_1, split_2 = split_layer_2,
               lr = [1e-7, 1e-4, 1e-2])

# Implementing SGDR
sched = LR_Cycle(iterations = np.ceil(x_train.shape[0]/batch_size),
                 cycle_mult = 2)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=["accuracy"])

history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks = [sched] #Adding scheduler in Callbacks
                   )

# Plot Historical LR values
sched.plot_lr()

