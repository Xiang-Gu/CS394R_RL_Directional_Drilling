import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt

class NoisyNextStateEnv(gym.Env):
    
    simstates = np.zeros((1,4)) #Continuous state variables
    
    def __init__(self):
        super(NoisyNextStateEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = spaces.Discrete(21)
        
        #Drilling state bounds:
        self.stateLow = np.array([0., 0., -10., -25.])
        self.stateHigh = np.array([1000, 500, 100, 25])
        
        self.observation_space = spaces.Box(low=self.stateLow, high=self.stateHigh, dtype=np.float64)

        self.X = 1000 #Size in feet of position grid x perimeter
        self.Y = 500 #Size in feet of position grid y perimeter

        self.stateDimension = 4
        self.numActions = 21
        self.gamma = 1.
        
        
    def step(self, action):
        global simstates
        # assert action in range(self.spec.nA), "Invalid Action"
        
        #Model Parameters
        dist=0
        tau=self.tau
        ku=self.ku

        #Soft Rock Layer
#        layer_size = 200
#        layer_depth = 200
#        if simstates[-1,0] > layer_depth and simstates[-1,0] < layer_depth+layer_size:
#            tau=25
#            ku=10

        #Simulation parameters
        deltad=1 #ODE solution resolution
        stepsize=30 #Feet drilled for each step 
        epso=np.arange(0,stepsize,deltad)
        
        #Calculate action as a percentage of input force
        u=(action - (self.numActions - 1) / 2) / (self.numActions - 1) * 2 
        #Set initial values
        [xo,yo,thetao,thetadoto]= simstates[-1,:]
        
        #Calculate theta and thetadot
        alpha=(ku*u+dist)/100;
        exp1=np.exp(-epso/tau);
        theta = (alpha-thetadoto)*tau*exp1 + (epso-tau)*alpha + thetao + tau*thetadoto;
        thetadot = alpha*(-exp1+1)+thetadoto*exp1;
        
        #Calculate x and y
        x = xo + deltad*np.cumsum(np.cos(np.deg2rad(theta)))
        y = yo + deltad*np.cumsum(np.sin(np.deg2rad(theta)))
        
        #Check if bounds have been exceeded
        x_check = ((x<self.stateLow[0]) | (x>self.stateHigh[0]))
        y_check = ((y<self.stateLow[1]) | (y>self.stateHigh[1]))
        trim = (x_check | y_check)
        out_of_bounds = np.any(trim)
        
        #init reward
        r = 0
        
        #Penalize obstacle collision
#        self.c_x = 520
#        self.c_y = 400
#        self.c_r = 100
#        if np.any((x-self.c_x)**2 + (y - self.c_y)**2 < self.c_r**2):
#            r=-250
        
        #Truncate states where bounds have been exceeded
        #if all states are out of bounds, keep the first one to avoid empty state arrays
        if trim[0]==True:         
            trim[0]=False
        x = x[~trim].reshape(-1,)
        y = y[~trim].reshape(-1,)
        theta = theta[~trim].reshape(-1,)
        thetadot = thetadot[~trim].reshape(-1,)

        # Add noise to next state for each step taken.
        # Within each step, there is actually many mini steps being
        # calculated and we only add noise to the very last step.
        x[-1] += np.random.randn() * 1.5
        y[-1] += np.random.randn() * 1.5
        theta[-1] += np.random.randn() * 0.5
        thetadot[-1] += np.random.randn() * 0.0001
        
        simstates = np.vstack((simstates,np.transpose(np.array([x,y,theta,thetadot]))))
        self.state = np.array((x[-1],y[-1],theta[-1],thetadot[-1]*100))
        
        if out_of_bounds:
            self.e_pos = -np.sqrt((x[-1]-self.X/2)**2+(y[-1]-self.Y)**2)
            self.r_pos = self.e_pos
            self.r_ang = -2*(abs(90-theta[-1]))
            self.r_cur = -np.max(abs(simstates[:,3]*100))
            r+= self.r_pos + self.r_ang + self.r_cur
            return self.state, r, True, {}
        else:
            r+=0
            return self.state, r, False, {}

    def reset(self):
        global simstates
        #Random Model Parameters
        import random
        self.tau = 15#random.uniform(15.-5,15.+5)
        self.ku = 18#random.uniform(18.-4,18+4)
        
        #Initialize states at zero
#        self.state = np.zeros(4,dtype=int) #Start at home
#        simstates = np.zeros((1,4))
        
        #Initialize states from random distribution
        yo = 0#random.uniform(90-20,90+20)
        xo = 0#random.uniform(0,20)
        
        #Initialize states from array of possible values
#        a= np.load('stacked_stoch_trained_states.npy')
#        so = a[np.random.choice(np.arange(0,a.shape[0],1))]
#        self.state = so
#        simstates = so.reshape(1,-1)

        self.state = np.array([xo,yo,0,0])
        simstates = np.array([[xo,yo,0,0]])
        return self.state
        
    def render(self, mode='human', close=False, show=True, save=False, r_info = True):
        fig, ax = plt.subplots()
#        fig.figsize = (5,16)

        plt.plot(simstates[:,1],-simstates[:,0])
        plt.xlim([0,self.Y])
        plt.ylim([-(np.max(simstates[:,0])+50),0])
        plt.xlabel('Cross Section ft')
        plt.ylabel('TVD ft')
        plt.grid()
        ax.set_axisbelow(True)
        #Plot region to avoid
#        circle = plt.Circle((self.c_y,-self.c_x),self.c_r,color='C2')
#        ax.add_artist(circle)

        reward_info = 'Total Reward:'+str(round(self.r_pos+self.r_ang+self.r_cur,2))+'\nPosition Error:'+str(round(self.r_pos,2))+'\nAngle Error:'+str(round(self.r_ang/2,2))+'\nMax Curvature: '+str(round(self.r_cur,2))
        
        if save is True:
            fig.savefig('otrajectory'+str(round(self.r_pos,3))+'.png')
            np.save('osimstates'+str(round(self.r_pos,3))+'.npy',simstates)
            with open('orewards'+str(round(self.r_pos,3))+'.txt', 'w') as text_file:
                text_file.write(reward_info)
        if show is True:
            plt.show()
        if r_info is True:
            print(reward_info)
        return simstates