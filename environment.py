import numpy as np
from matplotlib import pyplot as plt
'''
Simulator for the directional drilling process
'''

class AutoDrill():     

	simstates = np.zeros((1,4)) #Continuous state variables

	def __init__(self):
		self.stateLow = np.array([0., 0., -10., -25.])
		self.stateHigh = np.array([1000, 500, 100, 25])
		self.X = 1000 # Size in feet of position grid x perimeter
		self.Y = 500 # Size in feet of position grid y perimeter

		self.stateDimension = 4
		self.numActions = 21
		self.gamma = 1.
		
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
		
		#Calculate x and y
		x = xo + deltad*np.cumsum(np.cos(np.deg2rad(theta)))
		y = yo + deltad*np.cumsum(np.sin(np.deg2rad(theta)))
		
		simstates = np.concatenate((simstates,np.array([[x[-1],y[-1],theta[-1],thetadot[-1]]])))
		
		self.state = np.array((x[-1],y[-1],theta[-1],thetadot[-1]*100))
		
		
		if x[-1] < 0 or x[-1] > self.X or y[-1] > self.Y or y[-1] < 0:
			termination = True
		else:
			termination = False

		
		if termination is True:
			#For positive rewards try:
			r= -np.sqrt((x[-1]-self.X/2)**2+(y[-1]-self.Y)**2)
			##For positive rewards try:
#            r=np.sqrt(self.X**2+self.Y**2) - np.sqrt((x[-1]-self.X/2)**2+(y[-1]-self.Y)**2)
			return self.state, r, termination
		else:
			r=0
			return self.state, r, termination
