#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 13 11:24:49 2017

@author: Gustaf Ehn & Hugo Werner
"""
import numpy as np
import scipy.special as sps
import matplotlib.pyplot as plt
from numba import jit
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
    case=letter%6
    if case==0:
        return 0.1*np.sin(2*np.pi*x*letter/6)+.5
    elif case==1:
        return .2/(1+np.exp(-550.*(x-.1667*(letter+5)/6)))+.4
    elif case==2:
        return 0.6-(.2/(1+np.exp(-550.*(x-(.1667*(letter+4)/6)))))
    elif case==3:
        return -0.1*np.sin(2*np.pi*x*((letter+3)/6))+.5
    elif case==4:
        c=int(letter/M*L)
        x_out=np.zeros(np.size(x))
        x_out[:c]=-0.2*x[:c]/(letter/M)+.6
        x_out[c:]=0.2*(x[c:]-1)/(1-letter/M)+.6
        return x_out
    else:
        c=int(letter/M*L)
        x_out=np.zeros(np.size(x))
        x_out[:c]=0.2*x[:c]/(letter/M)+.4
        x_out[c:]=-(-0.2*(1-x[c:])/(1-letter/M))+.4
        return x_out
   
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
        B[i+1]=1/(sps.gamma(H_tn[i]+.5))*(1/np.sqrt(L))*(sum1+sum2);
    return B

def genSamples(n):
    samples=np.zeros([n*M,L])
    for i in range(M):
        for j in range(n):
            samples[i,:]=getMbm(i+1)
        print(i)
    return samples        

X=genSamples(1)
plt.plot(np.transpose(X[5,:]))
plt.show()

end = time.time()
print(end - start)
       