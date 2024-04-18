'''
ocsvm

author: lizhijian
date: 2019-10-30
'''

import sys
import os
import glob
from keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn import svm
import joblib
import cv2
import numpy as np
from tqdm import tqdm

import tensorflow as tf
import keras.backend.tensorflow_backend as K

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

gpuconfig = tf.ConfigProto()
gpuconfig.gpu_options.allow_growth=True
gpuconfig.gpu_options.per_process_gpu_memory_fraction=0.35

sess = tf.Session(config=gpuconfig)
K.set_session(sess)

class OCSVM(object):
    def __init__(self):
        self.model = ResNet50(input_shape=(224, 224,3),weights=None,include_top=False)
        # the weights below downloaded from ('https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
        self.model.load_weights('/home/wangyl/data_wyl/MAGAD/Comparison/OCSVM/weight/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
        self.ss = StandardScaler()
        self.ocsvmclf = svm.OneClassSVM(gamma=0.001,
                               kernel='rbf',
                               nu=0.08)
        self.ifclf = IsolationForest(contamination=0.08,
                            max_features=1.0,
                            max_samples=1.0,
                            n_estimators=40)
        self.pca = None

    def extractResnet(self, X):
        # X numpy array
        fe_array = self.model.predict(X)
        return fe_array

    def prepareData(self, path):
        datalist = glob.glob(path+'/*.png')
        # print(datalist)
        # felist = []
        # for p in tqdm(datalist):
        #     img = cv2.imread(p)
        #     img = cv2.resize(img, (224, 224))
        #     #img = preprocess_input(img, mode='tf')
        #     img = np.expand_dims(img, axis=0)
        #     fe = self.extractResnet(img)
        #     felist.append(fe.reshape(1,-1))
        
        # X_t = felist[0]
        # for i in range(len(felist)):
        #     if i == 0:
        #         continue
        #     X_t = np.r_[X_t, felist[i]]

        X_t = np.zeros((len(datalist), 7*7*2048))
        i = 0
        for p in tqdm(datalist):
            img = cv2.imread(p)
            img = cv2.resize(img, (224, 224))
            #img = preprocess_input(img, mode='tf')
            img = np.expand_dims(img, axis=0)
            fe = self.extractResnet(img)
            X_t[i] = fe.reshape(1,-1)
        
        return X_t

    def initPCA(self, X_train):
        self.pca = PCA(n_components=X_train.shape[0], whiten=True)

    def doSSFit(self, Xs):
        self.ss.fit(Xs)

    def doPCAFit(self,Xs):
        self.pca = self.pca.fit(Xs)
        return Xs
    
    def doSSTransform(self, Xs):
        Xs = self.ss.transform(Xs)
        return Xs
    
    def doPCATransform(self, Xs):
        Xs = self.pca.transform(Xs)
        return Xs

    def train(self, Xs):
        self.ocsvmclf.fit(Xs)
        self.ifclf.fit(Xs)

    def predict(self, Xs):
        pred = self.ocsvmclf.predict(Xs)
        return pred


def trainSVM(datapath, savepath):
    f = OCSVM()
    trainpath = os.path.join(datapath, 'train/0.normal')
    # print(trainpath)
    X_train = f.prepareData(trainpath)
    # do StandardScaler
    f.doSSFit(X_train)
    X_train = f.doSSTransform(X_train)
    # do pca
    f.initPCA(X_train)
    f.doPCAFit(X_train)
    X_train = f.doPCATransform(X_train)
    # train svm
    f.train(X_train)
    
    # save our models
    joblib.dump(f.ocsvmclf, os.path.join(savepath, 'ocsvmclf.model'))
    joblib.dump(f.pca, os.path.join(savepath, 'pca.model'))
    joblib.dump(f.ss,os.path.join(savepath, 'ss.model'))

def loadSVMAndPredict(datapath, savepath):
    f = OCSVM()
    # load models
    f.ocsvmclf = joblib.load(os.path.join(savepath, 'ocsvmclf.model'))
    f.pca = joblib.load(os.path.join(savepath, 'pca.model'))
    f.ss = joblib.load(os.path.join(savepath, 'ss.model'))

    X_test = f.prepareData(os.path.join(datapath, 'test/1.abnormal'))
    # do test data ss
    X_test = f.doSSTransform(X_test)
    # do test data pca
    X_test = f.doPCATransform(X_test)

    # predict
    preds = f.predict(X_test)
    # labels = np.ones()
    np.savetxt("preds.txt", preds, fmt='%f', delimiter=',')
    print(f'{preds}')


if __name__ == '__main__':
    datapath = "/home/wangyl/Public_Dataset/Anomaly_Detection256"
    savepath = "/home/wangyl/data_wyl/MAGAD/Comparison/OCSVM/output"
    trainSVM(datapath, savepath)
    loadSVMAndPredict(datapath, savepath)
    pass
