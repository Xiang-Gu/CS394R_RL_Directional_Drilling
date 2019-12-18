from typing import Iterable, Tuple

import numpy as np
from env import EnvSpec
from policy import Policy

def off_policy_mc_prediction_ordinary_importance_sampling(
    env_spec:EnvSpec,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    bpi:Policy,
    pi:Policy,
    initQ:np.array
) -> np.array:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        bpi: behavior policy used to generate trajectories
        pi: evaluation target policy
        initQ: initial Q values; np array shape of [nS,nA]
    ret:
        Q: $q_pi$ function; numpy array shape of [nS,nA]
    """
    
    #####################
    # TODO: Implement Off Policy Monte-Carlo prediction algorithm using ordinary importance
    # sampling (Hint: Sutton Book p. 109, every-visit implementation is fine)
    #####################
    
    Q = initQ.copy()
    N = np.zeros((env_spec.nS,env_spec.nA))
    for e in trajs:
        rho = 1
        G = 0
        
        for t in range(len(e)-1,-1,-1): #This for loop moves backwards from from the final step to 0 (s_T-1,a_T-1,r_T,s_T)
            #rho importance ratio
            e_t = e[t] #current time step t of current episode e
            s,a,r,s_prime = e_t 
            G=r+env_spec.gamma*G
            N[s,a] += 1 
            Q[s,a] += (rho*G-Q[s,a])/N[s,a] # Dont have to handle 0 in denominator N will always be >=1 
            rho = rho*pi.action_prob(s,a)/bpi.action_prob(s,a)
    return Q

def off_policy_mc_prediction_weighted_importance_sampling(
    env_spec:EnvSpec,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    bpi:Policy,
    pi:Policy,
    initQ:np.array
) -> np.array:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using behavior policy bpi
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        bpi: behavior policy used to generate trajectories
        pi: evaluation target policy
        initQ: initial Q values; np array shape of [nS,nA]
    ret:
        Q: $q_pi$ function; numpy array shape of [nS,nA]
    """

    #####################
    # TODO: Implement Off Policy Monte-Carlo prediction algorithm using weighted importance
    # sampling (Hint: Sutton Book p. 110, every-visit implementation is fine)
    #####################
    
    C = np.zeros((env_spec.nS,env_spec.nA))
    Q = initQ.copy()
    for e in trajs:
        G = 0
        W = 1
        for t in range(len(e)-1,-1,-1): #This for loop moves backwards from the final step to 0
            if W == 0 : #If W is zero then Q is no longer updated
                break
            e_t = e[t] #current time step t of current episode e
            s,a,r,s_prime = e_t 
            G=r+env_spec.gamma*G
            C[s,a] += W #C sums the weights used to update Q[s,a]
            Q[s,a] += (W/C[s,a])*(G-Q[s,a]) 
            W = W*pi.action_prob(s,a)/bpi.action_prob(s,a)
    return Q
