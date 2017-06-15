#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sat May 13 11:24:49 2017

@author: Gustaf Ehn & Hugo Werner
"""

import numpy as np
import scipy.special as sps
import matplotlib.pyplot as plt
import json

class Encrypter(object):
    
    def __init__(self):
        self.M=26
        self.L=500
        self.K=10*self.L
        self.t_pos=np.linspace(0,1,self.L)
        self.t_neg=np.linspace(-10,0,10*self.L)
        self.t=np.union1d(self.t_neg,self.t_pos)

    def Hurst(self,letter):
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
    

    def encryptWithOutNorm(self,letter):
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
 
    def encrypt(self,letter,var):
        L=self.L
        K=self.K
        t=self.t
        t_pos=self.t_pos
        t_neg=self.t_neg
        
        B=np.zeros(L)
        randVar=np.random.rand(K+L)-0.5
        H_tn=self.Hurst(letter)-0.5
        for i in range(len(B)-1):
            sum1=0
            sum2=0
            tn=t[K+i]
            sum1=np.sum((np.power(tn-t_neg[:-1],H_tn[i])-np.power(-t_neg[:-1],H_tn[i]))*randVar[:K-1])
            sum2=np.sum(np.power(tn-t_pos[:i+1],H_tn[i])*randVar[K:K+i+1])
            B[i+1]=(1/(sps.gamma(H_tn[i]+.5))*(1/np.sqrt(L))*(sum1+sum2)/np.sqrt(var[letter-1,i+1]))
        return B
    
    def plotEncryptedLetter(self,samplePath):
        plt.plot(samplePath)
        plt.show()
        
    def plotEncryptedWord(self,word):
        for i in range(np.size(word,0)):
            plt.plot(word[i,:])
            plt.show()
        
class SampleSet(object):
    
    def __init__(self):
        self.M=26
        self.L=500
        self.K=10*self.L
        self.t_pos=np.linspace(0,1,self.L)
        self.t_neg=np.linspace(-10,0,10*self.L)
        self.t=np.union1d(self.t_neg,self.t_pos)
    
    def genSamplesWithOutNorm(self,n):
        M=self.M
        L=self.L

        samples=np.zeros([n*M,L])
        s=Encrypter()
        for m in range(M):
            for j in range(n):
                samples[m*n+j,:]=s.encryptWithOutNorm(m+1)
        return samples

    def genSamples(self,n,var):
        M=self.M
        L=self.L
        
        samples=np.zeros([n*M,L])
        s=Encrypter()
        for m in range(M):
            for j in range(n):
                samples[m*n+j,:]=s.encrypter(m+1,var)
        return samples
    
    def save(self,saveData,filename):
        ofile = open(filename, "w")
        json.dump(saveData.tolist(), ofile, sort_keys = True, indent = 4)
        ofile.close()
        
    def samplePlot(self,nbrOfsam,samples):
        M=self.M
        
        for m in range(M):
            print('-------Sample path of one letter in a sample batch--------',m+1)
            plt.plot(samples[m*nbrOfsam,:])
            plt.show()

class SimulateVar(object):
    
    def __init__(self):
        self.M=26
        self.L=500
        self.K=10*self.L
        self.t_pos=np.linspace(0,1,self.L)
        self.t_neg=np.linspace(-10,0,10*self.L)
        self.t=np.union1d(self.t_neg,self.t_pos)
    
    def getVar(self,nbrOfSam,samples):
        M=self.M
        L=self.L
        simVar=np.zeros([M,L])
        for m in range(M):
            for l in range(0,L):
                simVar[m,l]=np.var(samples[m*nbrOfSam:(m*nbrOfSam+nbrOfSam),l])
        return simVar
    
    def normSet(self,n,sampleSet):
        M=self.M
        L=self.L
        
        norm=np.ones([M,L])
        for m in range(M):
            for l in range(1,L):
                norm[m,l]=np.var(sampleSet[m*n:m*n+n,l])
        
        norm=1/np.sqrt(norm)        
        normedSampleSet=np.zeros([n*M,L])
        for m in range(M):
            normedSampleSet[m*n:m*n+n,:]=np.multiply(sampleSet[m*n:m*n+n,:],norm[m,:])
        return normedSampleSet
    
    def save(self,saveData,filename):
        ofile = open(filename, "w")
        json.dump(saveData.tolist(), ofile, sort_keys = True, indent = 4)
        ofile.close()

       