# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 17:41:44 2020

@author: salman
"""

from PIL import Image 
import pandas as pd
import numpy as np
import cv2
import os


data = pd.read_csv('E:\\fyp data\\CocoStuff\\labels.txt', sep=" ",header=None)

arr=np.zeros(183)

for point in data.values:
    (key,name,val)=point[0],point[1],point[2]
    arr[key]=int(val)

print(arr)
print(np.unique(arr))

train_file= pd.read_csv('E:\\fyp data\\CocoStuff\\Coco Stuff Dataset\\imageLists\\validation.txt')
print(train_file)

train_lst=list(train_file["images"])

path="E:\\fyp data\\CocoStuff\\Coco Stuff Dataset\\annotations\\png\\"
saved="E:\\fyp data\\CocoStuff\\new_annotations\\validation\\"
person=0
for img in train_lst:
    imgPath=path+img+'.png'
    image=np.array(cv2.imread(imgPath,0))
    #print(type(image))
    
    image=arr[image]

    uniques=np.unique(image)
    
    if(len(uniques)>2):
        cv2.imwrite(saved+img+'.png',image)
# f.close()