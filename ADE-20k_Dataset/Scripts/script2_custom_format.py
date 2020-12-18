
import pandas as pd
import numpy as np
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 19:04:34 2020

@author: salman
"""



name=[
"unlabelled",
"wall",
"building",
"sky",
"sidewalk",
"vegitation",
"tree",
"person",
"mountain",
"stairs",
"bench",
"pole",
"car",
"bike",
"animal",
"ground",
"fence",
"water",
"road",
"mountain",
"bag",
]
color=[
(0,0,0),
(191, 153, 220),
(243, 168, 132),
(102, 38, 199),
(178, 103, 200),
(230, 57, 44),
(59, 5, 184),
(170, 120, 203),
(139, 218, 51),
(202, 251, 254),
(108, 246, 107),
(76, 159, 104),
(150,200,122),
(11,55,233),
(70,176,200),
(170, 55, 148),
(100, 100, 204),
(144, 238, 51),
(101, 250, 211),
(180, 264, 11),
(159, 64, 104),
]

f = open("E:\\fyp data\\ADEK-20\\ade20_labels_updated_new.txt",'w+')
for i in range(len(color)):
    print("adad")
    f.write("CustomDataSet('{}',            {}, {}, 'void', 0, False, True, {}),\n".format(name[i],i,i,str(color[i])))

