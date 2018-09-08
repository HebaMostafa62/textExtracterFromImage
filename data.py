import csv
import os
from random import shuffle
path = 'Img'



data = []
def get_data(path):
    foldernames = [x[0] for x in os.walk(path)]
    foldernames = foldernames[1:]
    label = 0
    for i in foldernames:
        images = os.listdir(i)
        for j in images:
            imagepath = i+'\\'+j
            imagelabel = label
            #if(label<=9):
            string = [imagepath , str(label)]
            data.append(string)
         #   else:
          #      string = [imagepath , str(chr(label))]
           #     data.append(string)
#        if(label == 9):
 #           label = 64
  #      if (label == 90):
   #         label = 96
        label = label + 1
get_data('Last_Dataset')
#get_data('Fnt')
shuffle(data)
#for i in data:
    #print(i)

filename = 'data.csv'
file = open(filename , 'w' ,newline='')
writer = csv.writer(file)
writer.writerow(["image path", "Label"])
for i in data:
    writer.writerow(i)
file.close()