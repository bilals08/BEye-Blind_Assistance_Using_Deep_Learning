# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 10:58:51 2020

@author: salman
"""
import os
import numpy as np

anotations_path="E:\\fyp data\\CocoStuff\\new_annotations\\validation\\"
training_path="E:\\fyp data\\CocoStuff\\Coco Stuff Dataset\\images\\"

anotations=os.listdir(anotations_path)
images=os.listdir(training_path)

for anotaion in anotations:
    anno=anotaion[:-4]+".jpg"
    if anno not in images:
        os.remove(os.path.join(anotations_path,anotaion))
        print("Delete")
        

