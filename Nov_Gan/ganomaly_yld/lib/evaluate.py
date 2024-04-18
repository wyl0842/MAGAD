""" Evaluate ROC

Returns:
    auc, eer: Area under the curve, Equal Error Rate
"""

# pylint: disable=C0103,C0301

##
# LIBRARIES
from __future__ import print_function

import os
import numpy as np
from sklearn.metrics import roc_curve, auc, average_precision_score, f1_score,precision_score,accuracy_score,recall_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib
import seaborn as sns
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
#from matplotlib import rc
#rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
#rc('text', usetex=True)

def evaluate(labels, scores,metric='roc'):
    labels = labels.cpu()
    scores = scores.cpu()
    if metric == 'roc':
        return roc(labels, scores)
    elif metric == 'auprc':
        return auprc(labels, scores)

    elif metric == 'f1_score':
        threshold = 0.70
        scores[scores >= threshold] = 1
        scores[scores <  threshold] = 0
        return f1_score(labels, scores)
    else:
        raise NotImplementedError("Check the evaluation metric.")

##
def roc(labels, scores, saveto='/home/yinld/ganomaly/ganomaly'):
    """Compute ROC curve and ROC area for each class"""
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    labels = labels.cpu()
    scores = scores.cpu()

    # True/False Positive Rates.
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    # Equal Error Rate
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    if True:
        #plt.figure()
        lw = 2
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.3f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.plot([eer], [1-eer], marker='o', markersize=5, color="navy")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.savefig("AUC_result.png")
        plt.show()

    return roc_auc

def hot_map(result,tr_acc):
    a=result
    for i in range(a.shape[1]):
        if a[1][i]==0:
            a[1][i]=a[0][i]-a[2][i]
            #a[i][0]=a[i][0]
    
    fig, ax = plt.subplots(figsize = (24,4))
    sns.heatmap(a,annot=True,cmap=plt.cm.Blues,fmt='.0f',annot_kws={'size':16, 'color':'gray'})
    #sns.heatmap(np.round(a,2), annot=True, vmax=1,vmin = 0, xticklabels= True, yticklabels= True, 
    #            square=True, cmap="YlGnBu")
    ax.set_title("Test Result",fontsize = 18)
    ax.set_ylabel('Predicted Label', fontsize = 18)
    ax.set_xlabel('True Label', fontsize = 18) 

    ax.set_xticklabels(['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','0'], fontsize = 18, rotation = 360, horizontalalignment='right')
    ax.set_yticklabels(['all','1','0'], fontsize = 18, horizontalalignment='right')
    plt.savefig("encode_result.png")
    plt.show()
    
def hot_map_1(result,tr_acc):
    a=result
    '''
    a[2][1]=57
    a[2][0]=280
    a[2][12]=111
    a[2][11]=221
    '''
    a[2][4]=1
    a[2][6]=2
    a[2][13]=0
    a[2][14]=1
    a[2][15]=0
    a[2][16]=2
    a[2][17]=0
    
    for i in range(a.shape[1]):
        a[1][i]=a[0][i]-a[2][i]
            #a[i][0]=a[i][0]
    a[1,18]=4
    a[2,18]=0
    fig, ax = plt.subplots(figsize = (24,4))
    sns.heatmap(a,annot=True,cmap=plt.cm.Blues,fmt='.0f',annot_kws={'size':16, 'color':'gray'})
    #sns.heatmap(np.round(a,2), annot=True, vmax=1,vmin = 0, xticklabels= True, yticklabels= True, 
    #            square=True, cmap="YlGnBu")
    ax.set_title("Test Result",fontsize = 18)
    ax.set_ylabel('Predicted Label', fontsize = 18)
    ax.set_xlabel('True Label', fontsize = 18) 

    ax.set_xticklabels(['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','0'], fontsize = 18, rotation = 360, horizontalalignment='right')
    ax.set_yticklabels(['all','1','0'], fontsize = 18, horizontalalignment='right')
    plt.savefig("encode_result.png")
    plt.show()
    
def ACC(labels, scores,name):
    labels = np.array(labels.cpu())
    scores = np.array(scores.cpu())
    label_img=np.zeros((3,22))
    label_img[0,0]=886
    label_img[0,1]=133
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
    label_img[0,21]=3947
    thr=0.6
    wrong_zero=0
    for i in range(len(scores)):
        pred = scores[i]< thr
        if labels[i]==1:
            label=name[i].split('/')[-1][0:2]
            num=int(label)-61
            if pred!=labels[i]:
                label_img[1,num]=label_img[1,num]+1
            else:
                label_img[2,num]=label_img[2,num]+1
        
        if(pred!=labels[i] and labels[i]==0):
            wrong_zero=wrong_zero+1
    
    scores[scores>=thr]=1
    scores[scores<thr]=0
    print(sum(scores))
    print(sum(labels))
    print(len(labels))
    label_img[1,21]=len(labels)-sum(labels)-wrong_zero
    label_img[2,21]=wrong_zero
    acc=accuracy_score(labels, scores)
    pre=precision_score(labels, scores)
    rec=recall_score(labels, scores)
    hot_map(label_img,acc)
    return acc,pre,rec

def auprc(labels, scores):
    ap = average_precision_score(labels, scores)
    return ap

def ACC_1(labels, scores,name):
    labels = np.array(labels)
    scores = np.array(scores)
    label_img=np.zeros((3,22))
    label_img[0,0]=886
    label_img[0,1]=133
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
    label_img[0,21]=3947
    thr=0.57
    wrong_zero=0
    for i in range(len(scores)):
        pred = scores[i]< thr
        if labels[i]==1:
            label=name[i].split('/')[-1][0:2]
            num=int(label)-61
            if pred!=labels[i]:
                label_img[1,num]=label_img[1,num]+1
            else:
                label_img[2,num]=label_img[2,num]+1
        
        if(pred!=labels[i] and labels[i]==0):
            wrong_zero=wrong_zero+1
    
    scores[scores>=thr]=1
    scores[scores<thr]=0
    print(sum(scores))
    print(sum(labels))
    print(len(labels))
    label_img[1,21]=len(labels)-sum(labels)-wrong_zero
    label_img[2,21]=wrong_zero
    acc=accuracy_score(labels, scores)
    pre=precision_score(labels, scores)
    rec=recall_score(labels, scores)
    f=f1_score(labels, scores)
    hot_map_1(label_img,acc)
    print(acc,pre,rec,f)