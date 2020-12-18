# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 15:33:11 2020

@author: Bilal
"""
'''
for file in files:
    img=cv2.imread(bA2 + file)
    ls=str(file[:-4])+'.png'
    if type(img)!=None and (ls in file2)==True :
         cv2.imwrite(baseAddress+str(file[:-4])+'.jpg',img)'''
import cv2
import numpy as np
import random
from skimage.transform import rotate

import os

baseAddress='C:\\Users\\Bilal\\Desktop\\images1\\customdataset\\training\\'
bA2='C:\\Users\\Bilal\\Desktop\\images1\\customannotations\\training\\'
files= os.listdir(baseAddress)


for file in files:
    img=cv2.imread(baseAddress + file)
    imgAnnot=cv2.imread(bA2+str(file[:-4])+'.png')
    
    if type(img)!=None and type(imgAnnot)!=None:
        
        #angle= random.randint(0,20)
        #ag=rotate(img, angle)

#        ag=np.fliplr(img)
#        agA=np.fliplr(imgAnnot)

#        
#        cv2.imshow('png',agA)
#        cv2.waitKey(0)
        
        cv2.imwrite(baseAddress+file[:-4]+'hflip.jpg',ag)
        cv2.imwrite(bA2+file[:-4]+'hflip.png',agA)
        
        nimg=np.zeros((img.shape[0],img.shape[1]))
#        
#        
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if(img[i,j,0]==255 and img[i,j,1]==255 and img[i,j,2]==255):
                    nimg[i,j]==4
                else:
                    nimg[i,j]=23
        print(np.unique(nimg))
        cv2.imwrite('C:\\Users\\Bilal\\Desktop\\images1\\customannotations\\training\\' + str(file[:-4] + '.png'),nimg)
#                
      
                    
#        print('C:\\Users\\Bilal\\Desktop\\images1\\labels\\tdlabel\\' + str(file[:-4] + '.jpg'))           
#        cv2.imwrite('C:\\Users\\Bilal\\Desktop\\images1\\customdataset\\training\\' + str(file[:-4] + '.jpg'),img)
        #cv2.waitKey(0)
    #print(img.shape)
    