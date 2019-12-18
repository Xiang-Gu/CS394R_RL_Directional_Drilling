# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 14:54:36 2019

@author: Mathew Keller

Evaluate Performance
"""

from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
import numpy as np

#Load R from folder 
#R = np.load('F:\School\Halliburton Research Project\Reinforcement Learning\Experiments\Experiment 2 Terminal Distance Reward (3)\R2.npy')
#R = R[0:100000]

#Load R from parent
R = np.load('R3.npy')

#Average R for plotting
Ravg= R.reshape((10000,10))
Ravg= np.average(Ravg,axis = 1)

#def eval_performance(Q,R):

# Load Q and R from files

#Q = np.load('F:\School\Halliburton Research Project\Reinforcement Learning\Python Code\Experiment Terminal Distance Reward\Qarray1.npy')
R = np.load('R2.npy')
#
#plt.plot(R)
#
#Rstar = R

#np.sum(Rstar<-100)/Rstar.shape[0]

fig = plt.figure(figsize=(10,5))
x = np.array([0,2000,4000,6000,8000,10000])
my_xticks = ['0', '20000', '40000', '60000','80000','100000']
plt.xticks(x,my_xticks)
plt.plot(Ravg[0:10000],LineWidth=.1)
plt.xlabel('Episode Number')
plt.ylabel('Reward')
plt.grid()
fig.show()
