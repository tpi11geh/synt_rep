#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 16:22:26 2017

@author: gustafehn
"""
import numpy as np
import scipy.special as sps
import matplotlib.pyplot as plt
import json

ifile = open('DATA', "r")
inputData = json.load(ifile)
ifile.close()
samples=np.array(inputData)
for i in range(26):
    print('-------Sample path of one letter in a sample batch--------',i+1)
    plt.plot(samples[i*30,:])
    plt.show()
