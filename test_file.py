#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 13 11:24:49 2017

@author: Gustaf Ehn & Hugo Werner
"""
import numpy as np

#
M=26
L=1000
K=10*L
t_pos=np.linspace(0,1,L)
t_neg=np.linspace(-10,0,10*L)
t=np.union1d(t_neg,t_pos)
B=np.zeros(L+1)
randVar=np.random.rand(1,K+L)-0.5
                      
def Hurst1(x,letter):
    if x<letter/26:
        return -0.2*x/(letter/26)+.6
    else:
        return .2*(x-1)/(1-letter/26)+.6   
          
def Hurst2(x,letter):
    if x<var/26:
        return  0.2*x/(letter/26)+0.4
    else:
        return 0.4+0.2*(1-x)/(1-(letter/26)) 
    
def Hurst3(x,letter):
    return -0.1*np.sin(2*np.pi*x*((letter+7)/8))+.5

def Hurst4(x,letter):
    return 0.1*np.sin(2*np.pi*x*((letter+7)/8))+.5  
 
def Hurst5(x,letter):
    return .2/(1+np.exp(-550.*(x-.2*(letter+4)/8)))+.4

def Hurst6(x,letter):
    return 0.6-(.2/(1+np.exp(-550.*(x-(.2*(letter+5)/8)))))  

def Hurst7(x,letter):
    if x<var/26:
        return  0.2*x/(letter/26)+0.4
    else:
        return 0.6
    
def Hurst8(x,letter):
    if x<var/26:
        return  -0.2*x/(var/26)+0.6
    else:
        return .4
                      
                                    
                      
def Hurst(x,var):
    switcher = {0 : Hurst1(tn,letter),
           1 : Hurst2(tn,letter),
           2 : Hurst3(tn,letter),
           3 : Hurst4(tn,letter),
           4 : Hurst5(tn,letter),
           5 : Hurst6(tn,letter),
           6 : Hurst7(tn,letter),
           7 : Hurst8(tn,letter),
    }
    return switcher.get(x%8)
def getMbm(letter)
for i in range(len(B)):
    sum1=0;
    sum2=0;
    H_tn=Hurst(t_pos[i],letter)

