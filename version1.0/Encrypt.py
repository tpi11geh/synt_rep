#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 18:08:03 2017

@author: hugowerner
"""
import numpy as np
import scipy.special as sps
import json

class Encryptor(object):
    def __init__(self,normalize=1):
        
        ifile = open('Variances50x500', "r")
        inputData = json.load(ifile)
        ifile.close()
        self.var=np.array(inputData)
        self.L=np.size(self.var,1)
        
        self.normalize=normalize
        self.K=10*self.L
        self.t_pos=np.linspace(0,1,self.L)
        self.t_neg=np.linspace(-10,0,10*self.L)
        self.t=np.union1d(self.t_neg,self.t_pos)
       
        

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
    
    def encrypt(self,letter):
        """
        Encrypt one letter
        
        Args:
            Letter (str): Letter to be encrypted.
        """
        letter=ord(letter)-96 
        x=self.genDMFBM(letter)
        if self.normalize==1:
            x_normed=np.zeros([self.L])
            x_normed[1:]=np.multiply(x[1:],1/np.sqrt(self.var[letter-1,1:]))
            return x_normed
        else:
            return x
    
if __name__=="__main__":
    Enc=Encryptor(normalize=1)
    x=Enc.encrypt('z')