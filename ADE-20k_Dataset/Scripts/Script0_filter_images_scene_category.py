##Choose images from scene cetegoty 


import numpy as np
import pandas as pd

d = {}
lst=[]

data=pd.read_csv("E:\\fyp data\\ADEK-20\\uniqueLabels.txt",sep=" ")


for point in data.values:
    (key,val)=point[0],point[1]
    d[key]=val
    

category_data = pd.read_csv('E:\\fyp data\\ADEK-20\\scene_category.txt', sep=" ")

arr=[]
for point in category_data.values:
    if d[point[1]]==1 or d[point[1]]==2:
        arr.append(point[0])

print(len(arr))

with open("E:\\fyp data\\ADEK-20\\result1.txt","w+") as file:
      file.write('\n'.join(arr))
      




