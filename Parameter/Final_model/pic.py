# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 18:07:16 2020

@author: YLD
"""

from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from lib.evaluate import evaluate,ACC,ACC_1
import numpy as np

path='/home/yinld/ganomaly/ganomaly/gve_1.txt'
path_1='/home/yinld/ganomaly/ganomaly/gve_2.txt'
path_2='/home/yinld/ganomaly/ganomaly/gve_3.txt'
file=open(path)
y_pre=[]
lines = file.readlines()    
for line in lines:
    y_pre.append(float(line))  

file=open(path_1)
y_true=[]
lines = file.readlines()    
for line in lines:
    y_true.append(float(line[0:3]))  

file=open(path_2)
name=[]
lines = file.readlines()    
for line in lines:
    name.append(line) 
    
ACC_1(y_true,y_pre,name)