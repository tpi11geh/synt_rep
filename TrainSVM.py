#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 18:33:05 2017

@author: hugowerner
"""
import numpy as np
from sklearn import svm
import scipy
import pywt
##dwt by matlab
#mat = scipy.io.loadmat('DataForSVM.mat', squeeze_me=True)
#data=mat['X_trans_train']
#X_train = data[:,0:40]
#y_train = data[:,40]
#data=mat['X_trans_test']
#X_test = data[:,0:40]
#y_test = data[:,40]
#print(X_train[:,-1])

##
mat = scipy.io.loadmat('DataForSVM.mat', squeeze_me=True)
data_train=mat['X_norm_train']
data_test=mat['X_norm_test']
X_train = data_train[:,0:500]
Y_train = data_train[:,500]
X_test = data_test[:,0:500]
Y_test = data_test[:,500]

# Wavelet transform
nbr_train_sampls=np.size(X_train,0)
nbr_test_sampls=np.size(X_train,0)

data_wt_sm_train1=np.zeros([nbr_train_sampls,25])
data_wt_sm_train2=np.zeros([nbr_train_sampls,25])
data_wt_sm_test1=np.zeros([nbr_test_sampls,25])
data_wt_sm_test2=np.zeros([nbr_test_sampls,25])
##normal
sm_nbr=10

for i in range(0,nbr_train_sampls):
    data_wt=pywt.downcoef('d', X_train[i,:], 'db1', mode='symmetric', level=1)
    #Smoothish
    for j in range(25+1):
        data_wt_sm_train1[i,j-1]=np.sum(np.abs(data_wt[(j-1)*sm_nbr:(j-1)*sm_nbr+sm_nbr]))
for i in range(0,nbr_train_sampls):
    data_wt=pywt.downcoef('d', X_train[i,:], 'db3', mode='symmetric', level=1)
    #Smoothish
    for j in range(25+1):
        data_wt_sm_train2[i,j-1]=np.sum(np.abs(data_wt[(j-1)*sm_nbr:(j-1)*sm_nbr+sm_nbr]))

for i in range(0,nbr_test_sampls):
    data_wt=pywt.downcoef('d', X_test[i,:], 'db1', mode='symmetric', level=1)
    #Smoothish
    for j in range(25+1):
        data_wt_sm_test1[i,j-1]=np.sum(np.abs(data_wt[(j-1)*sm_nbr:(j-1)*sm_nbr+sm_nbr]))
for i in range(0,nbr_test_sampls):
    data_wt=pywt.downcoef('d', X_test[i,:], 'db3', mode='symmetric', level=1)
    #Smoothish
    for j in range(25+1):
        data_wt_sm_test2[i,j-1]=np.sum(np.abs(data_wt[(j-1)*sm_nbr:(j-1)*sm_nbr+sm_nbr]))

X_train_pros = np.concatenate((data_wt_sm_train1, data_wt_sm_train2),axis=1)
X_test_pros = np.concatenate((data_wt_sm_test1, data_wt_sm_test2),axis=1)


clf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovo', degree=3, gamma='auto', kernel='linear',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
#clf=svm.LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
#     intercept_scaling=1, loss='squared_hinge', max_iter=10000,
#     multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
#     verbose=0)
clf.fit(X_train_pros, y_train) 
#
plt.plot(np.transpose(X_train_pros[500:510,:]))
plt.show()
print(sum(clf.predict(X_test_pros)==y_test)/1950*100)
#print(clf.predict(X_test))