import gym
import numpy as np
import matplotlib.pyplot as plt

#This function registers out environment 
def env_fn():
    try:
        import env.tightWellEnvironment
        return gym.make('TightWellEnv-v0')
    except:
        print('gym environment already registered. Removing...')
        env_dict = gym.envs.registration.registry.env_specs.copy()
        for env in env_dict:
             if 'TightWellEnv-v0' in env:
                  print('Remove {} from registry'.format(env))
                  del gym.envs.registration.registry.env_specs[env]
        import env.tightWellEnvironment
        return gym.make('TightWellEnv-v0')
        print('gym environment is now registered.')
        
from mpc import solveMPC
env = env_fn()
mpcPolicy = solveMPC(env)
for i in range(1):
    s = env.reset()
    done = False
    while not done:
    #    Call mpc to solve optimal control problem at current step
        a = mpcPolicy.action(s,plot=True,integer_control=False)
        print('action taken:',a)
        s,r,done,_ = env.step(a)
    simstates = env.render(save=False, show=True)
np.save('mpc_trajectory.npy',simstates)