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

d={}
data = pd.read_csv('E:\\fyp data\\ADEK-20\\new_se_new\\new.txt', sep="\t")

arr=np.zeros(151)
print(arr)
for point in data.values:
    (key,name,val)=point[0],point[-2],point[-1]
    arr[key]=val
print(arr)

print(arr)


train_file= pd.read_csv('E:\\fyp data\\ADEK-20\\validation_images.txt', sep="\t")

train_lst=list(train_file["images"])


path="E:\\fyp data\\ADEK-20\\ADEChallengeData2016\\ADEChallengeData2016\\annotations\\validation\\"
saved="E:\\fyp data\\ADEK-20\\new_se_new\\adk_annotations\\validation\\"


for img in train_lst:
    imgPath=path+img+'.png'
    image=np.array(cv2.imread(imgPath,0))
    image=arr[image]
    uniques=np.unique(image)
    
    if len(uniques>0):
        cv2.imwrite(saved+img+'.png',image)

print("Done")