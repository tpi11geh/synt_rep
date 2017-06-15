#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 13:00:38 2017

@author: hugowerner
"""

import numpy as np
import scipy.sp
import json

class DataSet(object):
    """
    Creates shuffled normalized samples with labels
    """
    def __init__(self, n=10, M=26, L=500, normalize=True):
        self.normalize=normalize #Will normalize data set if true
        self.n=n
        self.M=M
        self.L=L
        self.K=10*self.L
        self.t_pos=np.linspace(0,1,self.L)
        self.t_neg=np.linspace(-10,0,10*self.L)
        self.t=np.union1d(self.t_neg,self.t_pos)
        self.data=None
        self.var=None
        self.generateX()    
    
    
    def Hurst(self,letter):
        """
        Hurst functions which changes according to letter.
        """
        tn_pos=self.t_pos
        case=letter%4
        if case==0:
            return 0.1*np.sin(2*np.pi*tn_pos*letter/4)+.5
        elif case==1:
            return 0.6-(.2/(1+np.exp(-600*(tn_pos-((letter+1)/28)))))
        elif case==2:
            return 0.4+(.2/(1+np.exp(-600*(tn_pos-(letter/28)))))
        else:
            return -0.1*np.sin(2*np.pi*tn_pos*((letter+3)/6))+.5
       
    def genDMFBM(self,letter):
        """
        Generates descrete multifractional browian motion with the Husrt function assigned to the given letter.
        """
        L=self.L
        K=self.K
        t=self.t
        t_pos=self.t_pos
        t_neg=self.t_neg
        
        B=np.zeros(self.L)
        randVar=np.random.rand(K+L)-0.5
        H_tn=self.Hurst(letter)-0.5
        for i in range(len(B)-1):
            sum1=0
            sum2=0
            tn=t[K+i]
            sum1=np.sum((np.power(tn-t_neg[:-1],H_tn[i])-np.power(-t_neg[:-1],H_tn[i]))*randVar[:K-1])
            sum2=np.sum(np.power(tn-t_pos[:i+1],H_tn[i])*randVar[K:K+i+1])
            B[i+1]=1/(sps.gamma(H_tn[i]+.5))*(1/np.sqrt(L))*(sum1+sum2)
        return B
    
    
    def getVar(self,n,X):
        """
        Approxiamates the variance a each point for each letter
        
        Args:
            n (int) : Number of samples per letter
            X       : Set of all samples
            
        """
        M=self.M
        L=self.L
        var=np.zeros([M,L])
        for m in range(M):
            for l in range(0,L):
                var[m,l]=np.var(X[m*n:(m*n+n),l])
        return var
    
    def normalizeX(self,n,X,var):
        """
        Normalizes data using the simulated variance
        """
        M=self.M
        L=self.L
    
        X_normed=np.zeros([n*M,L])
        for m in range(M):
            X_normed[m*n:m*n+n,1:]=np.multiply(X[m*n:m*n+n,1:],1/np.sqrt(var[m,1:]))
        return X_normed
    
    def generateX(self):
        """
        Generates a data set, with n samples of each letter
            
        Args:
           n (int) : Number of samples per letter
        """
        M=self.M
        n=self.n
        
        X=np.zeros([n*M,self.L])
        for m in range(M):
            for j in range(n):
                X[m*n+j,:]=self.genDMFBM(m+1)
            print(m)
                
        if self.normalize==1:
            #Approximate the variance and normalize samples      
            self.var=self.getVar(n,X)
            X=self.normalizeX(n,X,self.var)
            
         
        #Labels for SVM
        y=np.zeros([1,n*M]);
        for i in range(M):
            y[0,n*i:n*(i)+n]=i+1
        
        #Labels for NN
        Y=np.zeros([n*M,M]);
        for i in range(M):
            Y[i*n:(i*n+n),i]=1;
            
        #Concatenate samples and labels and shuffle order
        data=np.concatenate((X, y.T,Y),axis=1)
        np.random.shuffle(data)
        
        self.data=data
        
        return True
    
    def saveData(self,filename):
        """
        Save shuffled data set including labels as json file.
        
        Args:
            filename (str) : Name which data will be saved as.
        """
        ofile = open(filename, "w")
        json.dump(self.data.tolist(), ofile, sort_keys = True, indent = 4)
        ofile.close()
    
    def saveVar(self,filename):
        """
        Save simulated variances to json file.
        
        Args:
            filename (str) : Name which variance will be saved as.
        """
        if self.normalize:   
            ofile = open(filename, "w")
            json.dump(self.var.tolist(), ofile, sort_keys = True, indent = 4)
            ofile.close()
        else:
             raise NameError('Variance was not computed')
        
    
if __name__=="__main__":
    DS=DataSet(L=500,n=50)
    DS.saveVar('Variances50x500')
    DS.saveData('DataSet50x500')

