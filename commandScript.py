#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 11:18:45 2017

@author: gustafehn
"""
import samplesGenerator as sg
import numpy as np
import sphinx

nbr=10
#singlesample=sg.Sample()
#sample=singlesample.getMbm(4)
#print(sample)
ss=sg.SampleSet()
Xsam=ss.genSamplesWithOutNorm(nbr)
#
sv=sg.SimulateVar()
sim_var=sv.getVar(nbr,Xsam)
filename='variance'
sv.save(sim_var,filename)
Xsam_normed=sv.normSet(nbr,Xsam)
#
#
#Xsam_Uvar=ss.genSamples(nbr,sim_var)
#s.samplePlot(nbr,Xsam_Uvar)
#sim_varU=sv.getVar(nbr,Xsam_Uvar)
#ss.samplePlot(1,sim_varU)
#ss.samplePlot(nbr,Xsam_normed)
#filename='DATA_Normed'
#ss.save(Xsam_normed,filename)

#e=sg.Encrypt()
#word=np.zeros([3,500])
#count=0
#w=np.array([8,5,10])
#for i in range(3):
#    word[i,:]=e.getEncWithOutNorm(w[i])
#
#e.plotEncryptedWord(word)
