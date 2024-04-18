# -*- coding: utf-8 -*-
"""
读取pred和true，作出ROC曲线
"""

import os
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from scipy.interpolate import spline
import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from scipy.signal import savgol_filter
import seaborn as sns

# def meanFilter(y, window=5):
#     res = []
#     for i, val in enumerate(y):
#         if i < window + 3:
#             res.append(y[i])
#         else:
#             res.append(np.mean(y[i-window:i]))
#     return res

pal_husl = sns.husl_palette(5,h=15/360,l=.65,s=1).as_hex()
pal_husl[0], pal_husl[4] = pal_husl[4], pal_husl[0]

def smooth(tpr):
    tpr[3:] = savgol_filter(tpr[3:], 53, 1, mode= 'nearest')
    # tpr = tpr
    return tpr

matplotlib.rcParams.update({'font.size': 14})

pred_data_path = '/home/wangyl/Code/Reconstruction/Dataprocess/output_data/{}_an_scores.txt'
true_data_path = '/home/wangyl/Code/Reconstruction/Dataprocess/output_data/{}_gt_labels.txt'
# methods = ['PCA', 'AE', 'GAN', 'GANY', 'GANS']
# color = ['skyblue', 'orange', 'slateblue', 'darkseagreen', 'firebrick']
# model = ['ModelA', 'ModelB', 'ModelC', 'ModelD', 'ModelE']
methods = ['AE', 'GAN', 'GANS']
linetype = ['--', '-.', '-']
color = pal_husl
# ['skyblue', 'orange', 'slateblue', 'darkseagreen', 'firebrick']
model = ['DSEBM', 'f-AnoGAN', 'Autoencoder', 'Ganomaly', 'MAGAD(ours)']

plt.figure()
lw = 2
plt.figure(figsize=(8,8))

# DSEBM
a = np.loadtxt('/home/wangyl/Code/MAGAD/Comparison/DSEBM/sorted_labels.txt')
b = np.loadtxt('/home/wangyl/Code/MAGAD/Comparison/DSEBM/sorted_energies.txt')
fpr, tpr, _ = roc_curve(a, b)
eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
roc_auc = auc(fpr, tpr)
tpr = smooth(tpr)
# tpr = meanFilter(tpr)
# plt.plot(fpr, tpr, label=f"AUC = {roc_auc:3f}")

plt.plot(fpr, tpr, color=color[0], linestyle=':', lw=lw, label='{}(area = {:.3f})'.format(model[0], roc_auc))
# plt.plot([eer], [1-eer], marker='o', markersize=4, color="navy")
# plt.plot([0, 1], [0, 1], linestyle="--")
# plt.title("ROC-AUC2")

# f-AnoGAN
df = pd.read_csv("/home/wangyl/Code/MAGAD/Comparison/f-anogan/score.csv")# 读csv
#df = pd.to_csv("score.csv")
trainig_label = 1
# labels = np.where(df["label"].values == trainig_label, 0, 1) # 라벨이 1이면 0을 반환, 아니면 1을 반환.
labels = np.where(df["label"].values == trainig_label, 1, 0)
# anomaly_score = df["anomaly_score"].values# 把一整列提取出来
img_distance = df["img_distance"].values
# z_distance = df["z_distance"].values
fpr, tpr, _ = roc_curve(labels, img_distance)#绘制ROC曲线
eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
# precision, recall, _ = precision_recall_curve(labels, img_distance)#计算准确率和召回率
roc_auc = auc(fpr, tpr)
tpr = smooth(tpr)
# pr_auc =  auc(recall, precision)
plt.plot(fpr, tpr, color=color[1], linestyle=(0,(4,1.5,1,1.5,1,1.5)), lw=lw, label='{}(area = {:.3f})'.format(model[1], roc_auc))
# plt.plot([eer], [1-eer], marker='o', markersize=4, color="navy")

i = 2
for method in methods:
    pred_path = pred_data_path.format(method)
    true_path = true_data_path.format(method)
    pre_file = open(pred_path)
    y_pre = []
    lines = pre_file.readlines()
    for line in lines:
        y_pre.append(float(line))  
    true_file = open(true_path)
    y_true = []
    lines = true_file.readlines()    
    for line in lines:
        y_true.append(float(line[0:3])) 
    fpr, tpr, thresholds = roc_curve(y_true, y_pre, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    AUC = auc(fpr, tpr)
    tpr = smooth(tpr)
    # tpr = meanFilter(tpr)
    plt.plot(fpr, tpr, color=color[i],
         lw=lw, linestyle=linetype[i-2], label='{}(area = {:.3f})'.format(model[i], AUC)) ###假正率为横坐标，真正率为纵坐标做曲线
    # plt.plot([eer], [1-eer], marker='o', markersize=4, color="navy")
    i = i + 1

# plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='-')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
#plt.title('AUC_result')
plt.legend(loc="lower right")
plt.grid()
# plt.savefig("AUC_result.png")
plt.savefig('AUC_result.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
plt.show()



