# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 21:02:50 2018

@author: CS
"""

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
import tensorflow as tf 
from keras import backend as k

'''def categorical_hinge(y_true, y_pred):
    pos = k.sum(y_true * y_pred, axis=-1)
    neg =k.max((1.0 - y_true) * y_pred, axis=-1)
    return k.mean(k.maximum(0.0, neg - pos + 1), axis=-1)
'''


# Data Preparing

batch_size = 128
nr_classes = 10 #62
nr_iterations = 100
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784) #Done
X_test = X_test.reshape(10000, 784) #Done
X_train = X_train.astype('float32') #Done
X_test = X_test.astype('float32') #Done
X_train /= 255 #Done
X_test /= 255 #Done


Y_train = np_utils.to_categorical(y_train, nr_classes) # ??
Y_test = np_utils.to_categorical(y_test, nr_classes) # ??

model = Sequential()
model.add(Dense(10, input_shape=(784,)))

X_val=X_train[0:10000,:]
Y_val=Y_train[0:10000,:]

X_train=X_train[10000:60000,:]
Y_train=Y_train[10000:60000,:]


model.add(Activation('softmax'))
model.summary()
model.compile(loss='hinge',
              optimizer='sgd',
              metrics=['accuracy'])

saved_weights_name='SVMWeights.h5'

checkpoint = ModelCheckpoint(saved_weights_name, 
                                     monitor='val_acc', 
                                     verbose=1, 
                                     save_best_only=True, 
                                     mode='max')

history = model.fit(X_train, Y_train,
                    batch_size = batch_size, nb_epoch = nr_iterations,
                    verbose = 1, validation_data = (X_val, Y_val) ,callbacks=[checkpoint])

score = model.evaluate(X_test, Y_test, verbose = 0)