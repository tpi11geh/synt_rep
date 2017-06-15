#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 11:13:28 2017

@author: hugowerner
"""

import numpy as np
import pywt
import json
from sklearn.externals import joblib
from keras.models import Sequential
from keras.layers import Dense
from sklearn import svm

def useDWT(X,wnames):
    """
    Uses descrete wavelet transform to preprocess data.
    
    Args:
        X (array) : singals to be transformed
        wnames (Tuple): two names of wavlets to be used in DWT 
    """
    nbr_of_sampls=int(np.size(X)/500)
    X_wt_ds=np.zeros([nbr_of_sampls,50])
    for i in range(0,nbr_of_sampls):
        for j in range(2):
            if nbr_of_sampls==1:
                X_wt=pywt.downcoef('d', X[:],  wnames[j], mode='symmetric', level=1)
            else:
                X_wt=pywt.downcoef('d', X[i,:],  wnames[j], mode='symmetric', level=1)
            for k in range(j*25,(j)*25+25):
                X_wt_ds[i,k]=np.sum(np.abs(X_wt[10*(k-25*j):10*(k-25*j)+10]))
    return X_wt_ds

def NN_1(X,Y):
    """
    NN Model 1
       - Layers: 2
       - Nodes/layer: [50]
       - Train on data prosseced with DWT, with wavelets 'db1' and 'db2'.
    """ 
    model = Sequential()
    
    #Add layers
    model.add(Dense(50, input_dim=50, activation='sigmoid'))
    model.add(Dense(26, activation='sigmoid'))

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    #Preprocces data
    wnames=['db1', 'db3']
    X_proccesed=useDWT(X,wnames)
    
    
    model.fit(X_proccesed, Y, epochs=200, batch_size=25)
    #model.save('NN_2')    
    return model

def NN_2(X,Y):
    """
    NN Model 2
       - Layers: 2
       - Nodes/layer: [50]
       - Train on data prosseced with DWT, with wavelets 'db1' and 'db3'.
    """ 
    model = Sequential()
    
    #Add layers
    model.add(Dense(50, input_dim=50, activation='sigmoid'))
    model.add(Dense(50, input_dim=50, activation='sigmoid'))
    model.add(Dense(26, activation='sigmoid'))
    
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    #Preprocess data
    wnames=['db1', 'db3']
    X_proccesed=useDWT(X,wnames)
    
    model.fit(X_proccesed, Y, epochs=150, batch_size=25)
    model.save('NN_1')
    return model
    


def NN_3(X,Y):
    """
    NN Model 1
       - Layers: 3
       - Nodes/layer: [50 50 50]
       - Train on data prosseced with DWT, with wavelets 'db1' and 'db3'.
    """ 
    model = Sequential()
    
    #Add layers
    model.add(Dense(50, input_dim=50, activation='sigmoid'))
    model.add(Dense(50, input_dim=50, activation='sigmoid'))
    model.add(Dense(50, input_dim=50, activation='sigmoid'))
    model.add(Dense(26, activation='sigmoid'))

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    #Preprocess data
    wnames=['db1', 'db3']
    X_proccesed=useDWT(X,wnames)
    
    
    model.fit(X_proccesed, Y, epochs=150, batch_size=25)
    model.save('NN_2')    
    return model
def SVM_1(X,Y):    
     """
     SVM Model 1
         Kernal: Linear
     """
     #Define model
     model = svm.SVC(C=1.0, cache_size=500,
                                 decision_function_shape='ovo',
                                 kernel='linear', max_iter=-1,shrinking=True,tol=0.001, verbose=False)
     #Preprocess data
     wnames=['db1', 'db3']
     X_proccesed=useDWT(X,wnames)
    
     #Train
     model.fit(X_proccesed,Y)
     
     #Save model
     joblib.dump(model, 'SVM_1.pkl') 
     return model
 
def SVM_2(X,Y):    
     """
     SVM Model 2
      
         Kernal: Ploynomial
         Degree:2
     """
     #Define model
     model= svm.SVC(C=1, cache_size=500, coef0=1.0,
                                 decision_function_shape='ovo', degree=2, gamma=1,
                                 kernel='poly', max_iter=-1, 
                                 shrinking=True,tol=0.001, verbose=False)
     
     #Preprocess data
     wnames=['db1', 'db3']
     X_proccesed=useDWT(X,wnames)
    
     #Train
     model.fit(X_proccesed,Y)
     
     #Save model
     joblib.dump(model, 'SVM_2.pkl') 
     
     return model
 
if __name__ == "__main__":
    # Load data
    ifile = open('DataSet50x500', "r")
    inputData = json.load(ifile)
    ifile.close()
    data=np.array(inputData)
    
    #Select training set, Covalidation set and test set.
    X_train=np.array(data[0:910,0:500])
    X_cv=np.array(data[910:1105,0:500])
    X_test=np.array(data[1105:,0:500])
    
    y_train=np.array(data[0:910,500])
    y_cv=np.array(data[910:1105,500])
    y_test=np.array(data[1105:,500])
    
    Y_train=np.array(data[0:910,501:])
    Y_cv=np.array(data[910:1105,501:])
    Y_test=np.array(data[1105:,501:])

    #Train Neural Networks
    NN_1=NN_1(X_train,Y_train)
    NN_2=NN_2(X_train,Y_train)

    #Train SVMs
    SVM_1=SVM_1(X_train,y_train)
    SVM_2=SVM_2(X_train,y_train)
    
    #Test Neural Netwokrs on CV set
    wnames=['db1', 'db2']
    training_scores_NN_1 = NN_1.evaluate(useDWT(X_cv,wnames), Y_cv)
    training_scores_NN_2 = NN_2.evaluate(useDWT(X_cv,wnames), Y_cv)
   
    #Test SVMs on CV set
    training_score_SVM_1 = sum(SVM_1.predict(useDWT(X_cv,wnames))==y_cv)/np.size(y_cv)*100
    training_score_SVM_2 = sum(SVM_2.predict(useDWT(X_cv,wnames))==y_cv)/np.size(y_cv)*100
    
    #Print results
    print('\n')
    print('Training scores \n')
    print('NN scores \n')
    print(training_scores_NN_1[1]*100)
    print(training_scores_NN_2[1]*100)

    print('SVM scores \n')
    print(training_score_SVM_1)
    print(training_score_SVM_2)
    
