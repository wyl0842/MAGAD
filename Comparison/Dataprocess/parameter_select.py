# -*- coding: utf-8 -*-
"""
作出参数选择曲线
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

import matplotlib.pyplot as plt
from pylab import *  

# pal_husl = '#0ca8da'
# names = ['500', '1000', '1500', '2000', '2500', '3000']
# y = [0.845, 0.873, 0.873, 0.880, 0.875, 0.872]

# # names = ['5', '10', '15', '20', '25']
# x = range(len(names))
# # y = [0.855, 0.84, 0.835, 0.815, 0.81]
# # y1=[0.86,0.85,0.853,0.849,0.83]
# # #plt.plot(x, y, 'ro-')
# # #plt.plot(x, y1, 'bo-')
# # #pl.xlim(-1, 11)  # 限定横轴的范围
# plt.ylim(0.82, 0.89)  # 限定纵轴的范围
# plt.plot(x, y, 'ro-', color='#4169E1', lw=2)
# # plt.plot(x, y1, marker='*', ms=10,label=u'y=x^3曲线图')
# # for a, b in zip(x, y):  
# #     plt.text(a, b, (a,b),ha='center', va='bottom', fontsize=10)  
# # plt.legend()  # 让图例生效
# plt.xticks(x, names)
# # plt.margins(0)
# # plt.subplots_adjust(bottom=0.15)
# plt.xlabel("Memory Size") #X轴标签
# plt.ylabel("AUC") #Y轴标签
# plt.grid()
# plt.savefig('MEM_SIZE.png', bbox_inches='tight', pad_inches=0.1, dpi=300)

# plt.show()

pal_husl = '#0ca8da'
names = ['0', '32', '64', '128', '160', '192', '224', '256']
y = [0.845, 0.843, 0.855, 0.869, 0.872, 0.873, 0.87, 0.62]

# names = ['5', '10', '15', '20', '25']
x = range(len(names))
# y = [0.855, 0.84, 0.835, 0.815, 0.81]
# y1=[0.86,0.85,0.853,0.849,0.83]
# #plt.plot(x, y, 'ro-')
# #plt.plot(x, y1, 'bo-')
# #pl.xlim(-1, 11)  # 限定横轴的范围
plt.ylim(0.5, 0.9)  # 限定纵轴的范围
plt.plot(x, y, 'ro-', color='#4169E1', lw=2)
# plt.plot(x, y1, marker='*', ms=10,label=u'y=x^3曲线图')
# for a, b in zip(x, y):  
#     plt.text(a, b, (a,b),ha='center', va='bottom', fontsize=10)  
# plt.legend()  # 让图例生效
plt.xticks(x, names)
# plt.margins(0)
# plt.subplots_adjust(bottom=0.15)
plt.xlabel("Corrupted Block Size") #X轴标签
plt.ylabel("AUC") #Y轴标签
plt.grid()
plt.savefig('INP_SIZE.png', bbox_inches='tight', pad_inches=0.1, dpi=300)

plt.show()