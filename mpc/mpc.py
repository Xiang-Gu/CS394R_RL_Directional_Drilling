"""
@author: Mathew Keller and Xiang Gu
This code frames the control problem as either a non-linear program or a mixed integer program
and solves it to determine an optimal action. The optimization problem is resolved every step
(ie. a model predictive control method).
Notes about optimization method:
    This optimization problem is very hairy, and we are limited by the constraints of available solvers
    as well. In order to make the problem tractable, a few approximations have been made which reduce
    the optimality of the solution and produces mismatch between it's prediction and the environment's
    response (even when transition dynamics are deterministic and match):
        
    1 The environment's transition dynamics apply each action for 30 feet (control increment = 30ft). 
      The solver requires the control increment to be described in terms of solver steps, which depend 
      on the final distance of the solution (since the problem is free final time). To compensate,
      we approximate what the trajectory distance is expected to be based on distance to goal, and use
      the approximation for the control step. This produces a mismatch between the environments response
      and the MPC's expected trajectory.
    2 The environment's actions are discretized such that the control input is actually an integer.
      This makes the problem a mixed integer problem and much more difficult to solve. The IPOPT solver
      
    
    We utilized several reformulations to get the problem to work with open-source solver (IPOPT):
    1 This is a free final time method- but IPOPT expects discretization to occur over a predetermined 
      time interval. To get around this, we introduce an extra variable (tf_scale) which contracts and 
      expands time in the dynamics, but keeps the discretization length the same.
    2 The constraints on final states are imposed in the objective function as final state penalties
      rather than imposed as strict boundary conditions. 
    3 The optimization seeks to minimize the maximimum of thetadot (minimax), this is implemented by
      introducing a new variable (thetadotmax). Thetadotmax is constrained to be >= thetadot. The obj
      then seeks to minimize thetadotmax
    
Reference: this code utilizes the open-source optimization suite from the following paper:
    Beal, L.D.R., Hill, D., Martin, R.A., and Hedengren, J. D., 
    GEKKO Optimization Suite, Processes, Volume 6, Number 8, 2018, doi: 10.3390/pr6080106.
    URL: https://gekko.readthedocs.io/en/latest/index.html#
"""
import numpy as np
import matplotlib.pyplot as plt
#Import gekko package, if not available- install it with pip
try:
    from gekko import GEKKO
except:
    option = input('The gekko solver is not available, would you like pip to install?\n[y/n]:')
    if option == 'y' or option == 'Y':
        import subprocess
        import sys
        #Install gekko package
        subprocess.check_call([sys.executable, "-m", "pip", "install", 'gekko'])


class solveMPC:
    def __init__(self,env):        
        self.theta_goal=90
        self.thetadot_goal=0
        self.x_goal=env.X/2
        self.y_goal=env.Y
        self.last_u = 0 #Keep track of last input control
  
    def action(self, state, plot = False,integer_control=False):
        #Return a discrete action 0-21 based on the dynamic optimization solution.
        """
        State Variables:
            These are the variables included in the dynamics
        Manipulated Variables:
            These are the control inputs
        """        
        
#       Initialize the GEKKO solver and set options:
        # The remote argument allows for the solver to be run on Brigham Young Universitie's public server
        self.m = GEKKO(remote = True)
        init_state = state
        
        self.m.options.IMODE = 6 # control
        if integer_control:
            self.m.options.SOLVER = 1 #Use APOPT Solver (only solver able to handle mixed integer)
        else:
            self.m.options.SOLVER = 3 #Use IPOPT Solver
            
        #Setup the solvers discretization array:
        #   disc_length is the total number of discretization steps for the solver to discretize the problem
        disc_len = 501
        self.disc_array = np.linspace(0,1,disc_len) #Solver time steps (For variable final time, this will be scaled)
        self.m.time = self.disc_array
        
        #Fixed Varaibles: These are fixed over the horizon but able to change for each iteration
        #tf_scale is the final time variable. This is a trick to get the solver to solve a free final time problem
        self.tf_scale = self.m.FV(value=150,lb=1,ub=2000)
        self.tf_scale.STATUS = 1
        
        # Parameters of environment (state space model)
        tau = self.m.Param(value=15)
        ku = self.m.Param(value=18)
        dist = self.m.Param(value=0)
        
        # Manipulated variables (Control Input)
        u = self.m.MV(value=self.last_u, lb=-10, ub=10, integer=integer_control)
        u_step_len = 30 #feet; the length overwhich control must stay constant
        u.MV_STEP_HOR = self.get_control_step(u_step_len,ku.value,state[2],self.theta_goal,disc_len)        
        u.STATUS = 1  # allow optimizer to change u
        
        #Handle Units
        #States have the following units below: x:feet y:feet theta:deg thetadot:deg/ft
        #Convert units of states (1,1, deg->rad *Acutally ode is in deg, deg/100ft -> deg/ft)
        #State Input: x y thetao thetadoto
        convert_units = np.array([1,1,1,1/100])
        init_state = init_state*convert_units
        d2r = np.pi/180
        
        # State Variables
        x = self.m.SV(value=init_state[0], lb=-10000, ub=10000)
        y = self.m.SV(value=init_state[1], lb=-10000, ub=10000)
        theta = self.m.SV(value=init_state[2], lb=-20, ub=100)
        thetadot = self.m.SV(value=init_state[3], lb=-.3, ub=.3)
        thetadotmax = self.m.SV(value=ku.value, lb=-.3, ub=.3)
                
        # Initialize binary vector used to retreive the final state ( vector of all zeros except last element = 1 )
        final = self.m.Param(np.zeros(self.m.time.shape[0]))
        final.value[-1] = 1
        
        # Dynamics
        self.m.Equation(theta.dt() == thetadot*self.tf_scale)
        self.m.Equation(thetadot.dt() == -(1/tau)*thetadot*self.tf_scale + (.01*ku/tau)*u/10*self.tf_scale + .01*dist/tau*self.tf_scale)
        self.m.Equation(x.dt() == self.m.cos(theta*d2r)*self.tf_scale)
        self.m.Equation(y.dt() == self.m.sin(theta*d2r)*self.tf_scale)
        self.m.Equation(thetadotmax >= thetadot) #This variable acts as max(thetadot)
        
        # Objective Functions
        self.m.Obj((thetadotmax)**2)
        self.m.Obj(10*(x*final-self.x_goal)**2)
        self.m.Obj(10*(y*final-self.y_goal)**2)
        self.m.Obj((theta*final-self.theta_goal)**2)
        self.m.Obj(.1*(thetadot*final-self.thetadot_goal)**2)
        
        try:
            self.m.solve(disp=False)
        except:
            print('Enter Debug Mode')
            
        if plot:
            self.plot(u,x,y,theta,thetadot)
        self.last_u = u.value[1]
        action = int(round(u.value[1]/10,1)*10+10)
        return action
   
    def get_control_step(self,desired_u_step,ku,thetao,theta_goal,disc_len):
        """
        This function estimates the number of solver discretization steps that the
        control input should stay constant. We have to estimate what this value should
        be since the final time is free and the solver doesn't allow us to vary the
        control step size with each solver iteration.
        First guess the length of the optimal trajectory, then convert this from
        feet to discretization steps"""
        delta_theta = np.abs(thetao-theta_goal)
        curvature_max = ku
        #Calculate the average arc length if we were to drill with full curvature
        #and half curvature capabilities
        avg_arc_length = 3/2*delta_theta/curvature_max*100
        u_solver_step = int(desired_u_step*disc_len/avg_arc_length)
        u_solver_step = max(u_solver_step,1)
        return u_solver_step

    def plot(self,u,x,y,theta,thetadot):
        theta=np.array(theta.value)
        thetadot=np.array(thetadot.value)
        x=np.array(x.value)
        y=np.array(y.value)
        sim_t = self.disc_array*self.tf_scale.value[0]
        plt.subplot(3,1,1)
        plt.plot(sim_t,u.value,'b-',label='Control Optimized')
        plt.legend()
        plt.ylabel('Input')
        plt.subplot(3,1,2)
        plt.plot(sim_t,theta,'r--',label='Theta Response')
        plt.ylabel('Deg')
        plt.legend(loc='best')
        plt.subplot(3,1,3)
        plt.plot(sim_t,thetadot,'r--',label='Theta_dot Response')
        plt.ylabel('Units')
        plt.xlabel('Measured Depth (ft)')
        plt.legend(loc='best')
        plt.show()
        plt.plot(y,-x,'r--',label='Trajectory')
        plt.ylabel('TVD')
        plt.xlabel('Cross Section')
        plt.legend(loc='best')
        plt.grid()
        plt.show()