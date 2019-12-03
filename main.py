import numpy as np
np.random.seed(1)
from environment import AutoDrill
from policy import EpsilonGreedy
from functionApproximator import LinearApproximatorOfActionValuesWithTile
from utility import evaluation

if __name__ == "__main__":
    env = AutoDrill()
    Q = LinearApproximatorOfActionValuesWithTile()
    epsilonGreedyPolicy = EpsilonGreedy(Q, 0.05)
    greedyPolicy = EpsilonGreedy(Q, 0.)

    numEpisodes = 50000

    print("Initiating Learning with Q-learning with function approximation")

    for episode in range(numEpisodes):
        # Initialize S, A, and done=False
        state = env.reset()
        reward = 0.
        done = False

        while not done:
            # Choose A from Q epsilon-greedily
            action = epsilonGreedyPolicy.action(state)
            # Take A and observe R and S'.
            nextState, reward, done = env.step(action)
            # Compute Q-learning target (R + gamma * max_a Q(s', a))
            target = reward + env.gamma * np.max(Q[nextState])
            # Update our function approximator
            Q.update(0.006, target, state, action)

            state = nextState

        print(str(episode) + "th episode: reward = " + str(reward))

        if episode % 1000 == 0:
            evaluation(env, greedyPolicy, Q, str(episode))