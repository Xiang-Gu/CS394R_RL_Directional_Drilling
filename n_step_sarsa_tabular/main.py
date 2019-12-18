# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 05:34:48 2019

@author: Mathew
"""

import numpy as np
from matplotlib import pyplot as plt

dist=0
tau=15
ku=10

#Simulation parameters
deltad=.1 #ODE solution resolution
stepsize=30 #Feet drilled for each step
epso=np.arange(0,stepsize,deltad)

#Retrieve states
u=6/(11-1) #Calculate action as a percentage of input force
[xo,yo,thetao,wo]= [0,0,0,0] 

alpha=(ku*u+dist)/100;
beta=tau*wo+thetao;
exp1=np.exp(-epso/tau);
theta = (alpha-wo)*tau*exp1 + (epso-tau)*alpha + thetao + tau*wo;

plt.plot(epso,theta)