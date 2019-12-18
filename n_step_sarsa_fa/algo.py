import numpy as np
from policy import Policy
from tqdm import tqdm
import pickle ############ Remove before submission

class ValueFunctionWithApproximation(object):
    def __call__(self,s) -> float:
        """
        return the value of given state; \hat{v}(s)

        input:
            state
        output:
            value of the given state
        """
        #No need to implement here- they are implemented in the child class that inherets this
        raise NotImplementedError()

    def update(self,alpha,G,s_tau):
        """
        Implement the update rule;
        w <- w + \alpha[G- \hat{v}(s_tau;w)] \nabla\hat{v}(s_tau;w)

        input:
            alpha: learning rate
            G: TD-target
            s_tau: target state for updating (yet, update will affect the other states)
        ouptut:
            None
        """
        #No need to implement here- This is a base class inhereted by other classes
        raise NotImplementedError()

def semi_gradient_n_step_td(
    env, #open-ai environment
    gamma:float,
    pi:Policy,
    n:int,
    alpha:float,
    V:ValueFunctionWithApproximation,
    num_episode:int,
):
    """
    implement n-step semi gradient TD for estimating v

    input:
        env: target environment
        gamma: discounting factor
        pi: target evaluation policy
        n: n-step
        alpha: learning rate
        V: value function
        num_episode: #episodes to iterate
    output:
        None
    """
    #Generate Tuples:
    print("Generating episodes based on instructor provided policy")
    
    ############## Remove before submission
    with open('trajs.pickle','rb') as f:
        trajs = pickle.load(f)
    ############## Uncomment before submission
#    trajs = []
#    for _ in range(num_episode): #Its nice to add tdqm to this loop to track progress
#        s = env.reset()
#        traj = []
#        done = False
#        while done is not True:
#            a = pi.action(s)
#            next_s, r, done, info = env.step(a)
#            traj.append((s, a, r, next_s))
#            s = next_s
#        trajs.append(traj)
    ############## Remove before submission
#    with open('trajs.pickle','wb') as f:
#        pickle.dump(trajs,f)
    #############
        
        
    for e in tqdm(trajs): 
        t=0
#        env.render()
        s = e[t][0]
        R=np.zeros(len(e)+1) #zero added since R_0 doesn't exist
        S=np.zeros((len(e)+1,len(s)))
        T = np.inf
        tau = -1
        while tau<(T-1): #Now we iterate each transition in this episode
            if t<T:
                s,a,r,s_prime = e[t]
                S[t,:]=s
                R[t+1]=r
                S[t+1,:]=s_prime
                if t == (len(e)-1):
                    T = t+1
            tau = t-n+1
            if tau >= 0:
                i=tau+1
                G = 0
                while i<=min(tau+n,T): #Compute sum of discounted rewards nsteps ahead
                    G+=gamma**(i-tau-1)*R[i]
                    i+=1
                if tau+n<T:
                    G+=(gamma**n)*V.__call__(S[tau+n,:]) #S[index] returns the continuous state we recorded at time step "index"
                V.update(alpha,G,S[tau,:]) #Weighting update is performed internal to the FunctionApproximation class - this way we can use different types of function approx.
            t+=1