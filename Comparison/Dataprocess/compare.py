# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 11:49:08 2020

@author: YLD
"""

import os
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import seaborn as sns
from sklearn.metrics import roc_curve, auc, average_precision_score, f1_score, precision_score, accuracy_score, recall_score

# pred保存路径,真实label(0/1)保存路径,真实类别保存路径
resultpath = '/home/wangyl/data_wyl/Reconstruction/Output/PCA/test_result/{}'

def evaluate(labels, scores, metric='roc'):
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
def roc(labels, scores):
    """Compute ROC curve and ROC area for each class"""
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # labels = labels.cpu()
    # scores = scores.cpu()

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

def auprc(labels, scores):
    ap = average_precision_score(labels, scores)
    return ap

def ACC(labels, scores, name, method):
    # labels = np.array(labels.cpu())
    # scores = np.array(scores.cpu())
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
    label_img[0,21]=3939
    thr=0.5
    wrong_zero=0
    for i in range(len(scores)):
        pred = scores[i]< thr
        if labels[i]==1:
            if len(name[i]) > 2:
                label = name[i].split('/')[-1][0:2]
            else:
                label = name[i]
            num = int(label)-61
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
    hot_map(label_img, acc, method)
    return acc,pre,rec

def hot_map(result, tr_acc, method):
    a=result
    for i in range(a.shape[1]):
        if a[1][i]==0:
            a[1][i]=a[0][i]-a[2][i]
            #a[i][0]=a[i][0]
    
    fig, ax = plt.subplots(figsize = (24,4))
    sns.heatmap(a,annot=True,cmap=plt.cm.Blues,fmt='.0f',annot_kws={'size':16, 'color':'gray'})
    #sns.heatmap(np.round(a,2), annot=True, vmax=1,vmin = 0, xticklabels= True, yticklabels= True, 
    #            square=True, cmap="YlGnBu")

    # 计算分类别准确率
    cls_pre = np.zeros((22,), dtype=np.float32)
    for i in range(len(cls_pre)):
        cls_pre[i] = a[1][i]/a[0][i]
    np.savetxt(('save_data/{}_class_precision.txt'.format(method)), cls_pre, fmt="%f", delimiter="\n")

    ax.set_title("Test Result",fontsize = 18)
    ax.set_ylabel('Predicted Label', fontsize = 18)
    ax.set_xlabel('True Label', fontsize = 18) 

    ax.set_xticklabels(['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','0'], fontsize = 18, rotation = 360, horizontalalignment='right')
    ax.set_yticklabels(['all','1','0'], fontsize = 18, horizontalalignment='right')
    plt.savefig("save_data/{}_encode_result.png".format(method))
    plt.show()

def sigmoid(x):
    # TODO: Implement sigmoid function
    return 1/(1 + np.exp(-x))

if __name__ == "__main__":
    methods = ['PCA', 'AE', 'GAN', 'GANY', 'GANS']
    for method in methods:
        # 读取文件
        # gt_labels = np.loadtxt(resultpath.format('gt_label.txt'), dtype=int, delimiter='\n')
        # an_scores = np.loadtxt(resultpath.format('pred.txt'), dtype=float, delimiter='\n')
        # gt_class = np.loadtxt(resultpath.format('gt_class.txt'), dtype=int, delimiter='\n')
        gt_labels = np.loadtxt('output_data/{}_gt_labels.txt'.format(method), dtype=float, delimiter='\n')
        an_scores = np.loadtxt('output_data/{}_an_scores.txt'.format(method), dtype=float, delimiter='\n')
        gt_class = np.loadtxt('output_data/{}_filename.txt'.format(method), dtype=str, delimiter='\n')

        an_score_mean = np.mean(an_scores)
        # 线性归一化
        # self.an_scores = (an_scores - np.min(an_scores)) / (np.max(an_scores) - np.min(an_scores))
        # Sigmoid归一化
        an_scores = 1 * (an_scores - an_score_mean)
        an_scores = sigmoid(an_scores)

        # 计算Accuracy, Precision, Recall
        

        # 计算F1-score, AUC
        # f1 = evaluate(gt_labels, an_scores, metric='f1_score')
        auc_score = evaluate(gt_labels, an_scores, metric='roc')
        # print(f1)
        # print(auc_score)
        acc, pre, rec = ACC(gt_labels, an_scores, gt_class, method)
        with open('save_data/{}_class_precision.txt'.format(method), 'a') as f:
            f.write('\nACC: {}\nPRE: {}\nREC: {}\nAUC: {}'.format(acc, pre, rec, auc_score))
        # print(acc)
        # print(pre)
        # print(rec)






