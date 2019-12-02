import numpy as np
from matplotlib import pyplot as plt
# Simulator for the directional drilling process

class AutoDrill():     

    simstates = np.zeros((1,4)) #Continuous state variables

    def __init__(self):
        # 100*90*50 states: Position grid of 100 blocks ; angles -180 to 180 at 4deg precision ; build rate -25 to 25 at 1deg/ft precision
        # 21 actions: -1 to 1 duty cycle in .1 resolution (Where duty cycle is the force of input)
        # self.sD = [1000,500,180,50] #Dimensions of state space
        # self.tile_width = np.array((50,5,5,10))
        self.X = 1000 #Size in feet of position grid x perimeter
        self.Y = 500 #Size in feet of position grid y perimeter
        # nS=np.product(self.sD)
        # env_spec = EnvSpec(nS, 21, 1.) #nS,nA,gamma
        # super().__init__(env_spec)

        self.stateDimension = 4
        self.numActions = 21
        self.gamma = 0.99
        
    # def awhere(self,ndloc,ndsize):
    #     pos = 0
    #     for p in range(ndloc.shape[0]-1):
    #         pos += ndloc[p]*np.product(ndsize[p+1:])
    #     pos += ndloc[-1]
    #     return pos
    
    def reset(self):
        global simstates
        
        #Zero initialization at x=0 y=0 theta=0 thetadot=0 initialization
        self.state = np.zeros(4,dtype=int) #Start at home
        simstates = np.zeros((1,4))
        
        return self.state
    
        
    def disp_traj(self, info, save_flag, show_flag=0):
        fig = plt.figure(1)
        plt.plot(simstates[:,1],-simstates[:,0])
        plt.xlim([0,self.Y])
        plt.ylim([-self.X,0])
        plt.xlabel('Cross Section ft')
        plt.ylabel('TVD ft')
        fig = plt.figure(1)
        if save_flag == 1:
            fig.savefig(str(info) + '.png')
        if show_flag == 1:
            plt.show()
#        plt.show()

    def step(self, action):
        global simstates
        # assert action in range(self.spec.nA), "Invalid Action"
        
        #Model Parameters
        dist=0
        tau=15
        ku=18
        
        #Simulation parameters
        deltad=1 #ODE solution resolution
        stepsize=60 #Feet drilled for each step 
        epso=np.arange(0,stepsize,deltad)
        
        #Retrieve states
        #This needs to be fixed- we want to keep track of the actual trajectory, not the coarse grid locations
        u=(action - (self.numActions - 1) / 2) / (self.numActions - 1) * 2 #Calculate action as a percentage of input force
        #Set initial values
        [xo,yo,thetao,thetadoto]= simstates[-1,:]
        
        #Calculate theta and thetadot
        alpha=(ku*u+dist)/100;
        exp1=np.exp(-epso/tau);
        theta = (alpha-thetadoto)*tau*exp1 + (epso-tau)*alpha + thetao + tau*thetadoto;
        thetadot = alpha*(-exp1+1)+thetadoto*exp1;
        
#        plt.plot(epso,theta)
        
        #Calculate x and y
        x = xo + deltad*np.cumsum(np.cos(np.deg2rad(theta)))
        y = yo + deltad*np.cumsum(np.sin(np.deg2rad(theta)))
        
        simstates = np.concatenate((simstates,np.array([[x[-1],y[-1],theta[-1],thetadot[-1]]])))
        
        self.state = np.array((x[-1],y[-1],theta[-1],thetadot[-1]*100))
        
        
        if x[-1] < 0 or x[-1] > self.X or y[-1] > self.Y or y[-1] < 0:
            termination = True
        else:
            termination = False
        
        #Scale states for tile coding - divide states by tile width
        # TODO:implement scaling outside of environment class   
        # self.state=self.state/self.tile_width
        
        if termination is True:
            #For positive rewards try:
            r= -np.sqrt((x[-1]-self.X/2)**2+(y[-1]-self.Y)**2)
            ##For positive rewards try:
#            r=np.sqrt(self.X**2+self.Y**2) - np.sqrt((x[-1]-self.X/2)**2+(y[-1]-self.Y)**2)
            return self.state, r, termination
        else:
            r=0
            return self.state, r, termination
