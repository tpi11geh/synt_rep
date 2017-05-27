#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 13 11:24:49 2017

@author: Gustaf Ehn & Hugo Werner
"""
import numpy as np

M=26
L=1000
K=10*L
t_pos=np.linspace(0,1,L)
t_neg=np.linspace(-10,0,10*L)
t=np.union1d(t_neg,t_pos)
B=np.zeros(L+1)
print(np.size(B))

print('dysseFan')