# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 15:44:32 2020

@author: YLD
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import glob
import seaborn as sns

label_img=np.zeros((1,21))
label_img[0,0]=886
label_img[0,1]=113
label_img[0,2]=137
label_img[0,3]=14
label_img[0,4]=4
label_img[0,5]=3
label_img[0,6]=14
label_img[0,7]=2
label_img[0,8]=1
label_img[0,9]=1
label_img[0,10]=47
label_img[0,11]=490
label_img[0,12]=316
label_img[0,13]=1
label_img[0,14]=2
label_img[0,15]=1
label_img[0,16]=5
label_img[0,17]=1
label_img[0,18]=4
label_img[0,19]=1
label_img[0,20]=14

'''
label_img[0,7]=2
label_img[0,9]=1
label_img[0,18]=4
label_img[0,19]=1
'''
def hot_map(result):
    a=result
    '''
    for i in range(a.shape[1]):
        if a[0][i]==0:
            a[0][i]=a[0][i]+np.random.randint(1,5)
    '''
    fig, ax = plt.subplots(figsize = (24,2))
    sns.heatmap(a,annot=True,cmap=plt.cm.Blues,fmt='.0f',annot_kws={'size':14, 'color':'gray'})
    #sns.heatmap(np.round(a,2), annot=True, vmax=1,vmin = 0, xticklabels= True, yticklabels= True, 
    #            square=True, cmap="YlGnBu")
    ax.set_xlabel('Label', fontsize = 12)
    ax.set_ylabel('Number', fontsize = 12) 

    ax.set_xticklabels(['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21'], fontsize = 12, rotation = 360, horizontalalignment='right')
    ax.set_yticklabels([''], fontsize = 12, horizontalalignment='right')
    plt.title('Picture Number')
    plt.savefig("pic_result.png")
    
    plt.show()
    


img_set= glob.glob('/home/yinld/luntai/gan_data/test_0/1.abnormal/*.png')
np.random.seed(5)
np.random.shuffle(img_set)
img_set=img_set[0:1046]

for i in range(len(img_set)):
    j=int(img_set[i].split('/')[-1][0:2])-int('61')
    label_img[0,j]=label_img[0,j]+1



#label_img[0,21]=26000
hot_map(label_img)