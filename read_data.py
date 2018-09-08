import cv2
import numpy as np
import csv
from PIL import Image


def removeBackground(img):
    #img = cv2.imread(path)
    ## (1) Convert to gray, and threshold
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    th, threshed = cv2.threshold(img, 200, 200, cv2.THRESH_BINARY_INV)

    ## (2) Morph-op to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)

    ## (3) Find the max-area contour
    _, cnts, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = sorted(cnts, key=cv2.contourArea)[-1]

    ## (4) Crop and save it
    x, y, w, h = cv2.boundingRect(cnt)
    dst = img[y:y + h, x:x + w]
    dst = cv2.resize(dst, (200, 200))
    #print(dst.shape)
    #dst = dst.reshape(40000)
    cv2.imwrite("001after.png", dst)
    return dst


if __name__ == "__main__":
    filename = 'data.csv'
    file = open(filename , 'r' ,newline='')
    reader = csv.reader(file)
    header = next(reader)
    X = []
    y = []
    for row in reader:
        path = row[0]
        X.append(path)
        lable = row [1]
        y.append(lable)

    print("here")
    images = []
    for imagepath in X:
        print(imagepath + "not yet")
        #input_img = cv2.imread(imagepath)
        #input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        input_img_resize = removeBackground(imagepath)
        cv2.imwrite(imagepath, input_img_resize)
        images.append(input_img_resize)
        #print(imagepath + " Done")

    print("Done")
    y = np.array(y)
    print(y.shape)

    images = np.array(images)
    print(type(images))
    print(images.shape)

    images = images.reshape(21700, 40000)
    print("Done reshaping")
    print(images.shape)

    #np.save('X',images)
    #np.save('y',y)

    X_train = images[:13888 , :]#.astype('float32')
    #X_train /= 255
    np.save('X_train',X_train)


    y_train = y[:13888]
    np.save('y_train',y_train)


    print("Done train")


    X_valid = images[13888:17360 , :]#.astype('float32')
    #X_valid /= 255
    print("valid ")
    np.save('X_valid',X_valid)


    y_valid = y[13888:17360]
    np.save('y_valid',y_valid)
    print("Done valid")

    X_test = images[17360:,:]#.astype('float32')
    #X_test /= 255
    np.save('X_test',X_test)

    y_test = y[17360:]
    np.save('y_test',y_test)

    print("Done test")
