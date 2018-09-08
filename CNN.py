

from keras.models import model_from_json
import difflib
import preprocessing
import numpy as np
from keras.preprocessing.image import  img_to_array
from PIL import Image


def CNN (lines):
    m = 200
    n = 200
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")

    #imgpath="F:\Projects\Pattern\Image"
    #os.chdir(imgpath);
    #im = Image.open(imgpath+'\\'+"6.png");
    txt = []
    for word in lines:
        for im in word:
            if im == ',':
                txt.append(" ")
            else:
                im = Image.fromarray(im)
                im=im.convert(mode='RGB')
                imrs=im.resize((m,n))
                imrs=img_to_array(imrs)/255;
                imrs=imrs.transpose(2,0,1);
                imrs=imrs.reshape(3,m,n); #m,n

                x=[]
                x.append(imrs)
                x=np.array(x);
                predictions = loaded_model.predict(x)
                predictedClass = np.argmax(predictions) + 1
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
    s = ""
    s = s.join(txt)
    #print(s)
    return s
    #print("Done")
