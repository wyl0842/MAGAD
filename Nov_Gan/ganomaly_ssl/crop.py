# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 16:26:04 2020

@author: YLD
"""

import numpy as np
import os
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import shutil
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image





img_path='/home/yinld/luntai/CDT/train/0.normal'

'''
for i in os.listdir(img_path): 
    if i.split('.')[0][-2:]=='_1':

'''
for i in os.listdir(img_path):    
    path=os.path.join(img_path,i)
    im=Image.open(path)
    im1=im.transpose(Image.FLIP_LEFT_RIGHT) 
    new_name=i.split('.')[0]+'_1'+'.png'
    #new_name_1=i.split('.')[0]+'_1'+'.png'
    name=os.path.join(img_path,new_name)
    #name_1=os.path.join(img_path,new_name_1)
    im1.save(name)


    
'''
for i in os.listdir(img_path):    
    path=os.path.join(img_path,i)
    im=Image.open(path)
    a,b=im.size
    x1=np.random.randint(low=0,high=50)
    y1=np.random.randint(low=0,high=50)
    x2=np.random.randint(low=a-50,high=a)
    y2=np.random.randint(low=b-50,high=b)
    im1=im.crop([x1,y1,x2,y2])
    im1=im1.resize((a,b))
    new_name=i.split('.')[0]+'_1'+'.png'
    #new_name_1=i.split('.')[0]+'_1'+'.png'
    name=os.path.join(img_path,new_name)
    #name_1=os.path.join(img_path,new_name_1)
    im1.save(name)
'''