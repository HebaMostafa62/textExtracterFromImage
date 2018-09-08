import os, cv2
import numpy as np

import tensorflow as tf
#from keras import backend as k
#k.set_image_dim_ordering('tf')

#from sklearn.utils import shuffle
#from sklearn.cross_validation import train_test_split

PATH = os.getcwd()
# Define data path
data_path = PATH + '\Img'
data_dir_list = os.listdir(data_path)
print (data_dir_list)

img_rows=128
img_cols=128
num_channel=1


img_data_list=[]

for dataset in data_dir_list:
    img_list=os.listdir(data_path+'\\'+ dataset)
    print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
    for images in img_list:
        list = os.listdir(data_path + '\\' + dataset+ '\\')
        print('Loaded the images of dataset-' + '{}\n'.format(images))
        for img in list:
            if img=='Thumbs.db':
                continue
            else:
                print (data_path + '/'+ dataset + '/'+ images+ '/'+ img)
                input_img=cv2.imread(data_path + '/'+ dataset + '/'+ images+ '/'+ img )
                input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
                input_img_resize=cv2.resize(input_img,(200,200))
                img_data_list.append(input_img_resize)

#convert images into np array
img_data = np.array(img_data_list)
#convert to float
img_data = img_data.astype('float32')
#normalization  in order to get it in the range  0 to 1
img_data /= 255

print (img_data.shape) #(number of samples ,numb of rows, numb of cols )
'''

if num_channel==1:
	if k.image_dim_ordering()=='tf':
		img_data= np.expand_dims(img_data, axis=4)  #(samples,cols,rows ,channels)
		print (img_data.shape)


# Assigning Labels

# Define the number of classes
num_classes = 4

num_of_samples = img_data.shape[0]

labels = np.ones((num_of_samples,), dtype='int64')

labels[0:202] = 0
labels[202:404] = 1
labels[404:606] = 2
labels[606:] = 3

names = ['cats', 'dogs', 'horses', 'humans']

# convert class labels to on-hot encoding
encoding 
[1,0,0,0]  = 0
[0,1,0,0]  = 1
[0,0,1,0]  = 2 
[0,0,0,1]  = 3

Y = np_utils.to_categorical(labels, num_classes)

# Shuffle the dataset
x, y = shuffle(img_data, Y, random_state=2)  #set random_state to use the same shuffle each time give you fixed value of shuffling

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
'''