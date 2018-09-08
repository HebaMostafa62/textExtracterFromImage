
import numpy as np
import importlib
import cv2
import difflib
import CNN
SVM = importlib.import_module('SVMload')
ModelName = 'SVM'
def prediction(ModelName , lines):


    string= ''
    wordList=[]
    for image in lines:
        #if type(image) is str:  # case of ","
        if image ==',':
            wordList.append(string)
            string=''
        else: #calling model
            if ModelName == 'SVM':
                prediction = SVM.SVM(image)
            if ModelName == 'CNN':
                 prediction = CNN.CNN(image)
            # if ModelName == 'KNN':
            #     prediction = SVM.SVM(image)
            #
            string+=prediction
    return wordList


# word_list=prediction()
# print(word_list)