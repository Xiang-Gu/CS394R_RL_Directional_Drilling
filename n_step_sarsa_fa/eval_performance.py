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
#R = np.load('R5_eval.npy')

#Average R for plotting


#Ravg= R[:100000].reshape((1000,100))
#Ravg= np.average(Ravg,axis = 1)

#def eval_performance(Q,R):

# Load Q and R from files

#Q = np.load('F:\School\Halliburton Research Project\Reinforcement Learning\Python Code\Experiment Terminal Distance Reward\Qarray1.npy')
#R = np.load('R2.npy')
#
#plt.plot(R)
#
#Rstar = R

#np.sum(Rstar<-100)/Rstar.shape[0]

fig = plt.figure(figsize=(10,5))
x = np.array([0,50,100,150,200,250,300])
my_xticks = ['0', '50000', '100000', '150000','200000','250000','300000']
#x = np.array([0,20,40,60,80,100])
#my_xticks = ['0', '20000', '40000', '60000','80000','100000']
plt.xticks(x,my_xticks)
plt.plot(R_greedy,LineWidth=1)
#plt.plot(R,LineWidth=1)
plt.ylim(-70,0)
plt.xlabel('Episode Number')
plt.ylabel('Reward')
#plt.title('Tile Coding with Semi-Gradient SARSA')
plt.title('ONLINE Finer Tile Coding with Monte Carlo Updates (alpha = .006)')
plt.grid()
fig.show()