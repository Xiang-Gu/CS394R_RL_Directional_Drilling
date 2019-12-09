import numpy as np
import gym
from env.environment import CustomEnv
from policy import ParameterizedPiWithNN
from functionApproximator import NonLinearApproximatorOfStateValuesWithNN
from utility import evaluation
from matplotlib import pyplot as plt
import random
import torch

def ActorCritic(seed):
    random.seed(seed) # Get consistent hashing for tile coding
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Initialize environment, state-value approximator, parameterized policy
    env = gym.make('CustomEnv-v0')
    V = NonLinearApproximatorOfStateValuesWithNN(alpha=1e-5, stateLow=env.stateLow, stateHigh=env.stateHigh)
    pi = ParameterizedPiWithNN(alpha=1e-5, state_dims=env.stateDimension, num_actions=env.numActions)

    # Initialize training parameters
    INF = 999
    epochSteps = 600
    numEpochs = 500
    maxSteps = epochSteps * numEpochs
    # max epoch returns (epochSteps steps is one epoch). Each number represents
    # the max return episode return (from starting state) over all episodes in that epoch.
    # E.g. maxEpRets = [-44, -21, -7.5] means we have run 3 epochs and -44 is the maximal
    # episode return among all, say, 5.7 episodes executed during the first epoch. (.7 because)
    # agent might be in the middle of an episode when the current epoch is finished.
    maxEpRets = [] 

    # Initialize counting apparatus
    t = 0             # number of steps agent has executed so far
    localT = 0        # number of steps within current epoch
    maxReward = -INF  # maximal non-zero reward encoutered during each step within current epoch 


    while True:
        state = env.reset()
        done = False

        while not done:
            # Choose A from parameterized policy
            action = pi.action(state)
            # Take A and observe R and S'.
            nextState, reward, done, info = env.step(action)
            # Compute one-step return target (R + gamma * V(S'; w))
            target = reward + env.gamma * V(nextState)
            # Update our function approximator
            V.update(s=state, G=target)
            # Update our parameterized policy
            pi.update(s=state, a=action, gamma_t=env.gamma ** t, delta=target - V(state))

            state = nextState

            # increment local time step within current epoch
            localT += 1
            if reward != 0: # end of one episode
                maxReward = max(maxReward, reward)
            if localT == epochSteps: # one epoch is full
                # under our speical reward design, maximal non-zero reward is also
                # the highest episode return among all episodes wihtin this epoch.
                maxEpRets.append(maxReward)
                print(str(len(maxEpRets)) + "th epoch is finished. localT = " + str(localT))
                print("global t = " + str(t+1)) 
                maxReward = -INF
                localT = 0

            # increment global time step
            t += 1
            if t == maxSteps:
                break

        if t == maxSteps:
            break

    np.savetxt("ActorCritic_maxEpRet_seed_" + str(seed) + ".npy", maxEpRets)


if __name__ == "__main__":
    for seed in range(5):
        print("initialize ActorCritic with seed = " + str(seed))
        ActorCritic(seed)