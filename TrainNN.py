#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 18:15:02 2017

@author: hugowerner
"""
import numpy as np
import pywt
import matplotlib.pyplot as plt
import scipy
# Create your first MLP in Keras
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import imdb

mat = scipy.io.loadmat('DataForNN.mat', squeeze_me=True)
data_train=mat['X_norm_train']
data_test=mat['X_norm_test']
X_train = data_train[:,0:500]
Y_train = data_train[:,500:]
X_test = data_test[:,0:500]
Y_test = data_test[:,500:] 

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

# create model
model = Sequential()
model.add(Dense(50, input_dim=50, activation='sigmoid'))
model.add(Dense(50, input_dim=50, activation='sigmoid'))
model.add(Dense(26, activation='sigmoid'))


# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X_train_pros, Y_train, epochs=200, batch_size=25)

# evaluate the model
scores = model.evaluate(X_test_pros, Y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))  

