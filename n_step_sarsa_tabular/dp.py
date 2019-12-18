from typing import Tuple

import numpy as np
from env import EnvWithModel
from policy import Policy




def value_prediction(env:EnvWithModel, pi:Policy, initV:np.array, theta:float) -> Tuple[np.array,np.array]:
    """
    inputs:
        env: environment with model information, i.e. you know transition dynamics and reward function
        pi: policy
        initV: initial V(s); numpy array shape of [nS,]
        theta: exit criteria
    return:
        V: $v_\pi$ function; numpy array shape of [nS]
        Q: $q_\pi$ function; numpy array shape of [nS,nA]
    """

    #####################
    # TODO: Implement Value Prediction Algorithm (Hint: Sutton Book p.75)
    #####################
    V=initV.copy() #Arbitrary except terminal states must be 0
    
    delta = theta
    while delta >= theta:
        delta = 0
        for s in range(env.spec.nS):
            v = V[s]
            #action probabilities: pi.action_prob(s) is an array (size nA)
            #transition dynamics: env.TD[state,action,state_t+1] is an array of probabilities (size nS)
            #rewards: env.R[state,action,state_t+1]
            update = 0
            for a in range(env.spec.nA):
                #Note below after action_prob the extra zero is because the array is wrapped in another array since array[None] 
                #puts the array in another array. Though this should always happen since action_prob shouldn't be passed an action
                update += pi.action_prob(s,a)*np.sum(env.TD[s,a,:]*(env.R[s,a,:]+env.spec.gamma*V))
            V[s] = update
            delta = max(delta,np.abs(v-V[s]))
    Q = np.zeros((env.spec.nS,env.spec.nA))
    for s in range(env.spec.nS):
        for a in range(env.spec.nA):
            Q[s,a]= np.sum(env.TD[s,a,:]*(env.R[s,a,:]+env.spec.gamma*V))
            #Note: summing over the actions in pi(a|s)*Q(s,a) gives V(s)
    return V, Q

def value_iteration(env:EnvWithModel, initV:np.array, theta:float) -> Tuple[np.array,Policy]:
    """
    inp:
        env: environment with model information, i.e. you know transition dynamics and reward function
        initV: initial V(s); numpy array shape of [nS,]
        theta: exit criteria
    return:
        value: optimal value function; numpy array shape of [nS]
        policy: optimal deterministic policy; instance of Policy class
    """
    
    class PiStar(Policy):
        def __init__(self, optActionProb, optPolicy):
            self.optActionProb = optActionProb
            self.optPolicy = optPolicy
    
        def action_prob(self, state, action):
            return self.optActionProb[state,action]
    
        def action(self, state):
            return self.optPolicy[state]


    V=initV.copy()
#    print('initV: \n',initV.reshape((4,4)))
#    print('V: \n',V.reshape((4,4)))
    delta = theta
    i=0
    while delta>=theta:
        i+=1
        delta = 0
        for s in range(env.spec.nS):
            v=V[s]
            update = np.zeros(env.spec.nA)
            for a in range(env.spec.nA):
                update[a] = np.sum(env.TD[s,a,:]*(env.R[s,a,:]+env.spec.gamma*V))
            V[s]=update.max()
            delta = max(delta,np.abs(v-V[s]))
#        print('V: \n',V.reshape((4,4)))
#        print('V: \n',V)
#        print('Delta:',delta)
#    print('iterations: ',i)
    Q = np.zeros((env.spec.nS,env.spec.nA))
    optActionProb = np.zeros((env.spec.nS,env.spec.nA))
    optPolicy = np.zeros((env.spec.nS))

    for s in range(env.spec.nS):
        for a in range(env.spec.nA):
            Q[s,a]= np.sum(env.TD[s,a,:]*(env.R[s,a,:]+env.spec.gamma*V))
        aStar = np.argmax(Q[s,:])
        optActionProb[s,aStar] = 1
        optPolicy[s]=aStar
#    print('Q: \n',Q.reshape((4,16)))
    
#    print('Q: \n',Q)

    pi = PiStar(optActionProb,optPolicy)
    #####################
    # TODO: Implement Value Iteration Algorithm (Hint: Sutton Book p.83)
    #####################

    return V, pi