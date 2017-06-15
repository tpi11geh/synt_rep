#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 10:19:18 2017

@author: hugowerner
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy
from keras.models import load_model
from CreatClassifiers import useDWT
import json
from Encrypt import Encryptor

########################################################################
########################################################################
#### This Script does the following:
####    1. Reads: Data, and a saved NN model
####    2. Tests NN on traing set and cv set
####    3. Encrypts a letter, plots encrypstion and 
####            uses the NN to classify the letter
########################################################################
########################################################################

#####  Part 1 ######

#Load Saved neural network
NNdecrypter = load_model('NN_1')

#Load data set
ifile = open('DataSet50x500', "r")
inputData = json.load(ifile)
ifile.close()
data=np.array(inputData)

#Split data set
X_train=np.array(data[0:910,0:500])
X_cv=np.array(data[910:1105,0:500])
X_test=np.array(data[1105:,0:500])

y_train=np.array(data[0:910,500])
y_cv=np.array(data[910:1105,500])
y_test=np.array(data[1105:,500])

Y_train=np.array(data[0:910,501:])
Y_cv=np.array(data[910:1105,501:])
Y_test=np.array(data[1105:,501:])

#####  Part 2 ######

wnames=['db1', 'db3']

scores_train = NNdecrypter.evaluate(useDWT(X_train,wnames), Y_train)
scores_cv = NNdecrypter.evaluate(useDWT(X_cv,wnames), Y_cv)

print('\n')
print('Training set\n')
print(scores_train[1]*100)
print('\n')
print('Covalidation set\n')
print(scores_cv[1]*100)

###3##  Part 3  ######
#Encrypt a letter and use NNdecrytper to try and decrypt
Enc=Encryptor()
x=Enc.encrypt('a')
plt.plot(x)
plt.show()
print('Neural Network predicts it is:', chr(NNdecrypter.predict_classes(useDWT(x,wnames))+97))

