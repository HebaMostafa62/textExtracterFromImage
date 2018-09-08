# -*- coding: utf-8 -*-

#importing Keras, Library for deep learning 
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.preprocessing.image import  img_to_array
from keras import backend as K
K.set_image_dim_ordering('th')
import numpy as np
import os
from PIL import Image
import theano
theano.config.optimizer="None"
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.models import model_from_json

# input image dimensions
m,n = 200,200

path="F:\Projects\Pattern"

os.chdir(path);
dataset="Dataset";
foldrs=os.listdir(dataset)
x=[]
y=[]
for fold in foldrs:
    classes=os.listdir(dataset+"\\"+fold)
    for Class in classes:
        imgfiles=os.listdir(dataset+'\\'+fold+'\\'+Class);
        for img in imgfiles:
            im=Image.open(dataset+'\\'+fold+'\\'+Class+'\\'+img);
            im=im.convert(mode='RGB')
            imrs=im.resize((m,n))
            imrs=img_to_array(imrs)/255;
            imrs=imrs.transpose(2,0,1);
            imrs=imrs.reshape(3,m,n);
            x.append(imrs)
            y.append(Class)
        
x=np.array(x);
y=np.array(y);
print("**********successfully uploaded**********")

batch_size=32
nb_classes=62
nb_epoch=20
nb_filters=32
nb_pool=2
nb_conv=3

x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.2,random_state=4)

uniques, id_train=np.unique(y_train,return_inverse=True)
Y_train=np_utils.to_categorical(id_train,nb_classes)
uniques, id_test=np.unique(y_test,return_inverse=True)
Y_test=np_utils.to_categorical(id_test,nb_classes)

print("**********Start Building model**********")


model= Sequential()
model.add(Convolution2D(nb_filters,nb_conv,nb_conv,border_mode='same',input_shape=x_train.shape[1:]))
model.add(Activation('relu'));
model.add(Convolution2D(nb_filters,nb_conv,nb_conv));
model.add(Activation('relu'));
model.add(MaxPooling2D(pool_size=(nb_pool,nb_pool)));
model.add(Dropout(0.5));
model.add(Flatten());
model.add(Dense(128));
model.add(Dropout(0.5));
model.add(Dense(nb_classes));
model.add(Activation('softmax'));
model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])

nb_epoch=5;
batch_size=5;
model.fit(x_train,Y_train,batch_size=batch_size,nb_epoch=nb_epoch,verbose=1,validation_data=(x_test, Y_test))

print("**********successfully fitted**********")

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

imgpath="F:\Projects\Pattern\Image"
os.chdir(imgpath);
im = Image.open(imgpath+'\\'+"6.png");
im=im.convert(mode='RGB')
imrs=im.resize((m,n))
imrs=img_to_array(imrs)/255;
imrs=imrs.transpose(2,0,1);
imrs=imrs.reshape(3,m,n);

x=[]
x.append(imrs)
x=np.array(x);
predictions = loaded_model.predict(x)
print("Done")
