# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 11:49:08 2020

@author: YLD
"""

import os
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.interpolate import interp1d

def compute_auc(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    AUC=auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.figure(figsize=(8,8))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.3f)' % AUC) ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig("AUC_result.png")
    plt.show()
    return AUC

'''
path='D://control//study_example//tyre_pictures//picture//changguicaozuo//demo/auc_result/ae.txt'
file=open(path)
y_pre=[]
y_true=[]
lines = file.readlines()    
for line in lines:
    y_pre.append(float(line.split(' ')[0]))  
    y_true.append(float(line.split(' ')[1][0:3]))  


'''
path='/home/yinld/ganomaly/ganomaly/ar_1.txt'
path_1='/home/yinld/ganomaly/ganomaly/ar_2.txt'
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

    
path='/home/yinld/ganomaly/ganomaly/aur_1.txt'
path_1='/home/yinld/ganomaly/ganomaly/aur_2.txt'
file=open(path)
y_pre_1=[]
lines = file.readlines()    
for line in lines:
    y_pre_1.append(float(line))  

file=open(path_1)
y_true_1=[]
lines = file.readlines()    
for line in lines:
    y_true_1.append(float(line[0:3]))  


path='/home/yinld/ganomaly/ganomaly/a_1.txt'
path_1='/home/yinld/ganomaly/ganomaly/a_2.txt'
file=open(path)
y_pre_2=[]
lines = file.readlines()    
for line in lines:
    y_pre_2.append(float(line))  


file=open(path_1)
y_true_2=[]
lines = file.readlines()    
for line in lines:
    y_true_2.append(float(line[0:3]))  


path='/home/yinld/ganomaly/ganomaly/au_1.txt'
path_1='/home/yinld/ganomaly/ganomaly/au_2.txt'
file=open(path)
y_pre_3=[]
lines = file.readlines()    
for line in lines:
    y_pre_3.append(float(line))  


file=open(path_1)
y_true_3=[]
lines = file.readlines()    
for line in lines:
    y_true_3.append(float(line[0:3]))  

    
path='/home/yinld/ganomaly/ganomaly/xbp_1.txt'
path_1='/home/yinld/ganomaly/ganomaly/xbp_2.txt'
file=open(path)
y_pre_4=[]
lines = file.readlines()    
for line in lines:
    y_pre_4.append(float(line))  


file=open(path_1)
y_true_4=[]
lines = file.readlines()    
for line in lines:
    y_true_4.append(float(line[0:3]))  
  
    
fpr, tpr, thresholds = roc_curve(y_true, y_pre, pos_label=1)
eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
fpr_1, tpr_1, thresholds = roc_curve(y_true_1, y_pre_1, pos_label=1)
eer_1 = brentq(lambda x: 1. - x - interp1d(fpr_1, tpr_1)(x), 0., 1.)
fpr_2, tpr_2, thresholds = roc_curve(y_true_2, y_pre_2, pos_label=1)
eer_2 = brentq(lambda x: 1. - x - interp1d(fpr_2, tpr_2)(x), 0., 1.)
fpr_3, tpr_3, thresholds = roc_curve(y_true_3, y_pre_3, pos_label=1)
eer_3 = brentq(lambda x: 1. - x - interp1d(fpr_3, tpr_3)(x), 0., 1.)
fpr_4, tpr_4, thresholds = roc_curve(y_true_4, y_pre_4, pos_label=1)
eer_4 = brentq(lambda x: 1. - x - interp1d(fpr_4, tpr_4)(x), 0., 1.)
AUC=auc(fpr, tpr)
AUC_1=auc(fpr_1, tpr_1)
AUC_2=auc(fpr_2, tpr_2)
AUC_3=auc(fpr_3, tpr_3)
AUC_4=auc(fpr_4, tpr_4)
plt.figure()
lw = 2
plt.figure(figsize=(8,8))
plt.plot(fpr_4, tpr_4, color='darkgrey',
         lw=lw, label='ModelA(area = %0.3f)' % AUC_4) ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([eer_4], [1-eer_4], marker='o', markersize=5, color="navy")
plt.plot(fpr, tpr, color='palevioletred',
         lw=lw, label='ModelB(area = %0.3f)' % AUC) ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([eer], [1-eer], marker='o', markersize=5, color="navy")
plt.plot(fpr_1, tpr_1, color='lightsalmon',
         lw=lw, label='ModelC(area = %0.3f)' % AUC_1) ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([eer_1], [1-eer_1], marker='o', markersize=5, color="navy")
plt.plot(fpr_2, tpr_2, color='steelblue',
         lw=lw, label='ModelD(area = %0.3f)' % AUC_2) ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([eer_2], [1-eer_2], marker='o', markersize=5, color="navy")
plt.plot(fpr_3, tpr_3, color='brown',
         lw=lw, label='ModelE(area = %0.3f)' % AUC_3) ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([eer_3], [1-eer_3], marker='o', markersize=5, color="navy")


plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
#plt.title('AUC_result')
plt.legend(loc="lower right")
plt.savefig("AUC_result.png")
plt.show()

print(1-eer)
print(1-eer_1)
print(1-eer_2)
#print(1-eer_3)






