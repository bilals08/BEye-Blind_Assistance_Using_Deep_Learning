# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 21:28:25 2020

@author: salman
"""

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




path="E:\\fyp data\\ADEK-20\\ADEChallengeData2016\\ADEChallengeData2016\\annotations\\anotations\\training\\"
saved="E:\\fyp data\\ADEK-20\\ADEChallengeData2016\\Object_info.txt"


d={}
data = pd.read_csv('E:\\fyp data\\ADEK-20\\objectInfoUpdated.txt', sep="\t")


for point in data.values:
    (key,name,val)=point[0],point[-2],point[-1]
    if val!=0:
        d[val]=name
d[0]="unlabeled"
print(d)


lst=np.zeros((51,2))
print(lst)

for i in range(len(lst)):
    lst[i][0]=i

train_lst=os.listdir(path)

print(train_lst)
for img in train_lst:
    imgPath=path+img
    image=np.array(cv2.imread(imgPath,0))
    uniques=np.unique(image)
    #print(uniques)
    for data in uniques:
        lst[data][1]+=1

print(lst)

with open(saved, "w+") as file:
    file.write("\n".join([str(int(i[0]))+": "+d[int(i[0])]+" - "+str(int(i[1])) for i in lst]))




   

# f.close()