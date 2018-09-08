import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras import optimizers
from sklearn.preprocessing import StandardScaler
import cv2

def removeBackground(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    th, threshed = cv2.threshold(img, 200, 200, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)

    _, cnts, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = sorted(cnts, key=cv2.contourArea)[-1]

    x, y, w, h = cv2.boundingRect(cnt)
    dst = img[y:y + h, x:x + w]
    dst = cv2.resize(dst, (200, 200))
    #print(dst.shape)
    #dst = dst.reshape(40000)
    cv2.imwrite("001after.png", dst)
    return dst

def SVM(lines):

    model = Sequential()
    model.add(Dense(62, input_shape=(40000,)))
    model.add(Activation('softmax'))
    sgd = optimizers.SGD(lr=0.005)#0.005
    model.load_weights('SVMWeights2.h5')
    model.compile(loss='hinge', optimizer=sgd, metrics=['accuracy'])

    txt = []
    for word in lines:
        for img in word:
            if img == ',':
                txt.append(" ")
            else:
                #img = removeBackground(img)
                img = cv2.resize(img , (200,200))
                img = img.astype('float32')
                std = StandardScaler()
                img = std.fit_transform(img)
                img = img.reshape((40000,))
                prediction = model.predict(np.expand_dims(img,axis=0))
                predictedClass = np.argmax(prediction) + 1
                #print(predictedClass)
                if (predictedClass <= 10):
                    # print(chr(predictedClass+47)) #0==>48
                    txt.append(chr(predictedClass + 47))
                elif (predictedClass >= 11 and predictedClass <= 36):  # A ==> 11  Z==> 36
                    # print(chr(predictedClass +54)) #A==>65
                    txt.append(chr(predictedClass + 54))
                else:  # a ==> 37  z ==>62
                    # print(chr(predictedClass +60)) #a==>97
                    txt.append(chr(predictedClass + 60))
                #img = np.expand_dims(img, axis=0)

    s = ""
    s = s.join(txt)
    #print(s)
    return s
    #prediction = model.predict(images)
    #print(prediction)



