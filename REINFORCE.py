import numpy as np
np.random.seed(1)
import gym
from env.environment import CustomEnv
from policy import ParameterizedPiWithNN
from functionApproximator import NonLinearApproximatorOfStateValuesWithNN
from utility import evaluation
from matplotlib import pyplot as plt

def REINFORCE():
    env = gym.make('CustomEnv-v0')
    V = NonLinearApproximatorOfStateValuesWithNN(alpha=1e-3, stateLow=env.stateLow, stateHigh=env.stateHigh)
    pi = ParameterizedPiWithNN(alpha=5e-5, state_dims=env.stateDimension, num_actions=env.numActions)
    numEpisodes = 22000
    rewards = []

    for episode in range(numEpisodes):
        traj = [] # used to hold transitions of one complete episode
        I = 1. # I = gamma**t
        state = env.reset()
        action = pi.action(state)
        done = False

        # generate one complete episode
        while not done:
            nextState, reward, done, info = env.step(action)
            traj.append((state, action, reward, nextState))
            state = nextState
            action = pi.action(state)

        print(str(episode) + "th epside. Reward = " + str(reward))

        # learn from this just completed episode
        T = len(traj)
        for t in range(T):
            S_t = traj[t][0]
            A_t = traj[t][1]

            # compute G_t
            G_t = 0.
            for k in range(t + 1, T + 1):
                R_k = traj[k - 1][2] # e.G_t. R_{t+1} is stored in traj[t][2]
                G_t += env.gamma ** (k - t - 1) * R_k   

            # compute TD error
            delta = G_t - V(S_t)
            if np.isnan(delta):
                print("error! delta is NaN")
            if delta >= 1000 or delta <= -1000:
                print("delta is too extreme:", delta)
            # update value function network
            V.update(s=S_t, G=G_t)
            # update policy network
            pi.update(s=S_t, a=A_t, gamma_t=I, delta=delta)
            # update gamma**t
            I = env.gamma * I

        # Evaluate our current policy every 20 episodes
        if episode % 20 == 0:
            print('Value function estimate: ', V.__call__(np.array([0,0,0,0])))
            rewards.append(reward)
            print(pi.action_prob(np.array([0,0,0,0])))
            # evaluation(env, pi, str(episode))

    np.savetxt("rewards", rewards)

if __name__ == "__main__":
    REINFORCE()