# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 05:37:15 2019

@author: Mathew
"""

# -*- coding: utf-8 -*-
"""
Created on Thursday Oct 4 12PM 2019
By Mathew Keller

Online on-policy n-step SARSA
"""
import numpy as np
from typing import Iterable, Tuple
from env import EnvSpec
import time

def episodic_semi_gradient_sarsa(
    env_spec:EnvSpec,
    traj:Iterable[Tuple[int,int,int,int]],
    n:int,
    alpha:float,
    V
):
    """
    input:
        env_spec: environment spec
        traj: 1 trajectory generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        n: how many steps?
        alpha: learning rate
        initQ: initial Q values; np array shape of [nS,nA]
    ret:
        w: weight vector; numpy array shape of [nS,nA]
    """

#takes ones trajectory traj        
    T=np.inf
    t=0
    tau=-1
    s = traj[t][0]
    R=np.zeros(len(traj)+1) #zero added since R_0 doesn't exist
    A=np.zeros(len(traj), dtype=int) #There is exactly one action per truple
    S=np.zeros((len(traj)+1,len(s)),dtype=int)

    while tau<(T-1): #Now we iterate each transition in this episode
        if t<T:
            #Extract transition information
            s,a,r,s_prime = traj[t]
            S[t,:]=s
            A[t]=a
            R[t+1]=r
            S[t+1,:]=s_prime
            #Check if S_t+1 is terminal
            if t == (len(traj)-1):
                T=t+1
            else:
                A[t+1]=traj[t+1][1] #Store next action
        tau = int(t-n+1)
        if tau >= 0:
            i=tau+1
            G = 0
            while i<=min(tau+n,T): #Compute sum of discounted rewards nsteps ahead
                G+=env_spec.gamma**(i-tau-1)*R[i]
                i+=1
            #Print sum of Rewards
#                print('G: ',G)
            #Add estimated Q value n steps ahead if we dont hit termination after n steps
            if tau+n<T: 
                G+=(env_spec.gamma**n)*V.__call__(np.append(S[tau+n,:],A[tau+n]))
#                    print('Q(S,A) est:',Q[S[tau+n],A[tau+n]])
#            print('Updating State/Action: ' + str(np.append(S[tau,:],A[tau])))
            V.update(alpha,G,np.append(S[tau,:],A[tau]))
        t+=1
    return None