from typing import Iterable, Tuple

import numpy as np
from env import EnvSpec
from policy import Policy

def on_policy_n_step_td(
    env_spec:EnvSpec,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    n:int,
    alpha:float,
    initV:np.array
) -> Tuple[np.array]:
    """
    input:
        env_spec: environment spec (gamma, nA, nS ...)
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        n: how many steps?
        alpha: learning rate
        initV: initial V values; np array shape of [nS]
    ret:
        V: $v_pi$ function; numpy array shape of [nS]
    """
    
    #####################
    # TODO: Implement On Policy n-Step TD algorithm
    # sampling (Hint: Sutton Book p. 144)
    #####################
    
    #Note: the commented lines were all used to visualize the learning process

    V = initV.copy()
#    Vi = np.zeros((len(trajs),2)) #%%%%%%%not for submission
#    ei = 0
    for e in trajs: #Now we're inside the episode, a list of transitions
#        Vi[ei,0] = V[0] #%%%%%%%not for submission
#        ei+=1 #%%%%%%%not for submission
#        print(e)
        T=np.inf
        tau=-1
        R=np.zeros(len(e)+1) #zero added since R_0 doesn't exist
        S=np.zeros(len(e)+1,dtype=int)
        t=0
        while tau<(T-1): #Now we iterate each transition in this episode
            if t<T:
                #Extract transition information
                s,a,r,s_prime = e[t]
                S[t]=int(s)
                R[t+1]=r
                S[t+1]=int(s_prime)
                #Check if S_t+1 is terminal
                if t == (len(e)-1):
                    T=t+1
            tau = int(t-n+1)
            if tau >= 0:
                #Calculate G
                i=tau+1
                G = 0
                #Visualize V
#                print('V:\n',V.reshape((4, 4)))
                while i<=min(tau+n,T): #Compute sum of discounted rewards nsteps ahead
                    G+=env_spec.gamma**(i-tau-1)*R[i]
                    i+=1
                #Print sum of Rewards
#                print('G: ',G)
                #Add estimated V value n steps ahead if we dont hit termination after n steps
                if tau+n<T: 
                    G+=(env_spec.gamma**n)*V[S[tau+n]] 
#                    print('V(S_tau+n) est:',V[S[tau+n]])
                V[S[tau]]+=alpha*(G-V[S[tau]])
                #Visualize V
#                print('V_update:\n',V.reshape((4, 4)))
            t+=1
    return V #, Vi #%%%%%%% Vi is not for submission

def off_policy_n_step_sarsa(
    env_spec:EnvSpec,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    bpi:Policy,
    n:int,
    alpha:float,
    initQ:np.array
) -> Tuple[np.array,Policy]:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        bpi: behavior policy used to generate trajectories
        pi: evaluation target policy
        n: how many steps?
        alpha: learning rate
        initQ: initial Q values; np array shape of [nS,nA]
    ret:
        Q: $q_star$ function; numpy array shape of [nS,nA]
        policy: $pi_star$; instance of policy class
    """

    #####################
    # TODO: Implement Off Policy n-Step SARSA algorithm
    # sampling (Hint: Sutton Book p. 149)
    #####################
    
    class PiStar(Policy):
        def __init__(self, optActionProb, optPolicy):
            self.optActionProb = optActionProb
            self.optPolicy = optPolicy
    
        def action_prob(self, state, action):
            return self.optActionProb[state,action]
    
        def action(self, state):
            return self.optPolicy[state]
    
    
    Q = initQ.copy()
    for e in trajs: #Now we're inside the episode, a list of transitions (list of tuples)
        
        T=np.inf
        tau=-1
        R=np.zeros(len(e)+1) #zero added since R_0 doesn't exist
        A=np.zeros(len(e), dtype=int) #There is exactly one action per truple
        S=np.zeros(len(e)+1,dtype=int)
        t=0
        while tau<(T-1): #Now we iterate each transition in this episode
            if t<T:
                #Extract transition information
                s,a,r,s_prime = e[t]
                S[t]=int(s)
                A[t]=int(a)
                R[t+1]=r
                S[t+1]=int(s_prime)
                #Check if S_t+1 is terminal
                if t == (len(e)-1):
                    T=t+1
                else:
                    A[t+1]=e[t+1][1] #Store next action
            tau = int(t-n+1)
            if tau >= 0:
                #Calculate rho (importance ratio) and G (return estimate)
                i=tau+1
                rho = 1
                while i<=min(tau+n,T-1):
                    #Calculate target policy pi probability (greedy policy = 1 if Q[S_i,] is max)
                    piprob = int(Q[S[i],A[i]]==np.max(Q[S[i],:])) 
                    rho=rho*piprob/bpi.action_prob(S[i],A[i])
                    i+=1
                i=tau+1
                G = 0
                while i<=min(tau+n,T): #Compute sum of discounted rewards nsteps ahead
                    G+=env_spec.gamma**(i-tau-1)*R[i]
                    i+=1
                #Print sum of Rewards
#                print('G: ',G)
                #Add estimated Q value n steps ahead if we dont hit termination after n steps
                if tau+n<T: 
                    G+=(env_spec.gamma**n)*Q[S[tau+n],A[tau+n]] 
#                    print('Q(S,A) est:',Q[S[tau+n],A[tau+n]])
                Q[S[tau],A[tau]]+=alpha*rho*(G-Q[S[tau],A[tau]])
                #Visualize Q_S
#                print('Q_update:\n',Q[S[tau],:].reshape((2, 2)))
            t+=1
    optActionProb = np.zeros((env_spec.nS,env_spec.nA))
    optPolicy = np.zeros(env_spec.nS)
    
    for s in range(env_spec.nS):
        a = np.argmax(Q[s,:])
        optActionProb[s,a] = 1
        optPolicy[s] = a

    pi = PiStar(optActionProb,optPolicy)

    return Q, pi
