import numpy as np

class RandomPolicy():
    def __init__(self, numActions):
        self.numActions = numActions

    def action_prob(self, state, action):
        return 1 / self.numActions

    def action(self, state):
        return np.random.choice(self.numActions)


class EpsilonGreedy():
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def action_prob(self, state, action, Q):
        numActions = Q.numActions

        if Q[state][action] == np.max(Q[state]): # prob. of choosing greedy action
            return 1. - self.epsilon + self.epsilon / numActions
        else: # prob. of choosing any other action
            return self.epsilon / numActions

    def action(self, state, Q):
        numActions = Q.numActions
        if np.random.rand() < self.epsilon: # random action
            return np.random.choice(numActions)
        else:
            maxActionValue = np.max(Q[state]) # greedy action (random if there are multiple greedy actions)
            return np.random.choice([a for a in range(numActions) if Q[state][a] == maxActionValue])


