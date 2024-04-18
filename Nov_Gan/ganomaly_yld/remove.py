# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 15:31:49 2020

@author: YLD
"""

import os

path='/home/yinld/luntai/gan_data/val/1.abnormal'
ls = os.listdir(path)    
for i in range(len(ls)):
    if i <500:
        line=ls[i]
        filePath = os.path.join(path, line)
        os.remove(filePath)
