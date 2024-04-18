# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 15:47:30 2020

@author: YLD
"""

import os 
import shutil

path_0='/home/yinld/luntai/gan_data/test/0.normal'
path_1='/home/yinld/luntai/gan_data/test/1.abnormal'

new_0='/home/yinld/luntai/gan_data/test_baseline/0.normal'
new_1='/home/yinld/luntai/gan_data/test_baseline/1.abnormal'

    
wrong_img=[]
path='1.txt'
file=open(path)
lines = file.readlines()    
for line in lines:
    line=line.split('\n')[0]
    line=line.split('/')[-1]
    wrong_img.append(line)  
        
j=1
       
for i in wrong_img:
    file_name=os.path.join(path_1,i)
    new_name=os.path.join(new_1,i)
    if (file_name in wrong_img) and j<150 and (i not in os.listdir(new_1)):
        j=j+1
        shutil.copyfile(file_name, new_name)  
            

        