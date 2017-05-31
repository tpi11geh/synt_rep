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
import time
#
start = time.time()
M=26
L=500
K=10*L
t_pos=np.linspace(0,1,L)
t_neg=np.linspace(-10,0,10*L)
t=np.union1d(t_neg,t_pos)

def Hurst(x,letter):
    case=letter%4
    if case==0:
        return 0.1*np.sin(2*np.pi*x*letter/4)+.5
    elif case==1:
        return 0.6-(.2/(1+np.exp(-600*(x-((letter+1)/28)))))
    elif case==2:
         return 0.4+(.2/(1+np.exp(-600*(x-(letter/28)))))
    else:
        return -0.1*np.sin(2*np.pi*x*((letter+3)/6))+.5
    

def getMbm(letter):
    B=np.zeros(L)
    randVar=np.random.rand(K+L)-0.5
    H_tn=Hurst(t_pos,letter)-0.5
    for i in range(len(B)-1):
        sum1=0
        sum2=0
        tn=t[K+i]
        sum1=np.sum((np.power(tn-t_neg[:-1],H_tn[i])-np.power(-t_neg[:-1],H_tn[i]))*randVar[:K-1])
        sum2=np.sum(np.power(tn-t_pos[:i+1],H_tn[i])*randVar[K:K+i+1])
        B[i+1]=1/(sps.gamma(H_tn[i]+.5))*(1/np.sqrt(L))*(sum1+sum2)
    return B
 
def getMbmVar(letter,var):
    B=np.zeros(L)
    randVar=np.random.rand(K+L)-0.5
    H_tn=Hurst(t_pos,letter)-0.5
    for i in range(len(B)-1):
        sum1=0
        sum2=0
        tn=t[K+i]
        sum1=np.sum((np.power(tn-t_neg[:-1],H_tn[i])-np.power(-t_neg[:-1],H_tn[i]))*randVar[:K-1])
        sum2=np.sum(np.power(tn-t_pos[:i+1],H_tn[i])*randVar[K:K+i+1])
        B[i+1]=(1/(sps.gamma(H_tn[i]+.5))*(1/np.sqrt(L))*(sum1+sum2)/np.sqrt(var[letter-1,i+1]))
    return B

def genSamples(n):
    samples=np.zeros([n*M,L])
    for i in range(M):
        for j in range(n):
            samples[i*n+j,:]=getMbm(i+1)
        print(i)
    return samples

def genSamplesVar(n,var):
    samples=np.zeros([n*M,L])
    for i in range(M):
        for j in range(n):
            samples[i*n+j,:]=getMbmVar(i+1,var)
        print(i)
    return samples

def getVar(nbrOfSam,samples):
    simVar=np.zeros([M,L])
    for i in range(M):
        for j in range(0,L):
            simVar[i,j]=np.var(samples[i*nbrOfSam:(i*nbrOfSam+nbrOfSam),j])
    return simVar

        
def save(saveData,filename):
    ofile = open(filename, "w")
    json.dump(saveData.tolist(), ofile, sort_keys = True, indent = 4)
    ofile.close()

def samplePlot(nbrOfsam,samples):
    for i in range(M):
        print('-------Sample path of one letter in a sample batch--------',i+1)
        plt.plot(samples[i*nbrOfsam,:])
        plt.show()
     
nbrOfSam=30
Xsam=genSamples(nbrOfSam)
sim_var=getVar(nbrOfSam,Xsam)

Xsam_Uvar=genSamplesVar(nbrOfSam,sim_var)
#sim_varU=getVar(nbrOfSam,Xsam_Uvar)
#samplePlot(1,sim_varU)
save(Xsam,'DATA')
end = time.time()
print(end - start)
       