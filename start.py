from tkinter import *
from tkinter import filedialog
import importlib
import difflib
import CNN
import post
import preprocessing
import SVMload
import numpy as np
prediction = importlib.import_module('prediction')
global filename
root = Tk() # Create the root (base) window where all widgets go
root.geometry("500x500")

w = Label(root, text="Upload your photo") # Create a label with words
w.pack() # Put the label into the window

x=[]
i = -1
def callSVM():
    allwords = SVMload.SVM(lines)
    post.generateDoc.generate(allwords ,'SVMoutput.txt')
    print(allwords)



# def callKnn():
#     print(prediction.prediction('KNN',x[i]))


def callCnn():
    allwords = CNN.CNN(lines)
    post.generateDoc.generate(allwords, 'CNNoutput.txt')
    print(allwords)

def browsefunc():
    global lines
    lines = []
    filename = filedialog.askopenfilename()
    x.append(filename)
    print(x[i])
    lines = preprocessing.pre_processing(x[i])
    print("preparing image .... Done")
    global i
    i+=1

browsebutton = Button(root, text="Browse", command=browsefunc)
browsebutton.pack()


btnSvm = Button(root, width=20, text="SVM",command=callSVM)

#btnKnn = Button(root, width=20, text="Knn", command=callKnn)

btnCnn = Button(root, width=20, text="CNN", command=callCnn)




#btnKnn.pack()
btnSvm.pack()
btnCnn.pack()

btnSvm.place(x=170,y=150, height=30, width=150)
#btnKnn.place(x=170,y=250, height=30, width=150)
btnCnn.place(x=170,y=340, height=30, width=150)

root.mainloop() # Start the event loop