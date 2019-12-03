import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim

class RandomPolicy():
    def __init__(self, numActions):
        self.numActions = numActions

    def action_prob(self, state, action):
        return 1 / self.numActions

    def action(self, state):
        return np.random.choice(self.numActions)


class EpsilonGreedy():
    def __init__(self, Q, epsilon):
        self.Q = Q
        self.numActions = self.Q.numActions
        self.epsilon = epsilon

    def action_prob(self, state, action):
        # an ndarray of Q(state, action) for all actions.
        actionValues = self.Q[state] 
        if actionValues[action] == np.max(actionValues): # prob. of choosing greedy action
            return 1. - self.epsilon + self.epsilon / self.numActions
        else: # prob. of choosing any other action
            return self.epsilon / self.numActions

    def action(self, state):
        if np.random.rand() < self.epsilon: 
            # random action
            return np.random.choice(self.numActions)
        else:
            # greedy action (break ties randomly)
            actionValues = self.Q[state]
            maxActionValue = np.max(actionValues)
            return np.random.choice([a for a in range(self.numActions) if actionValues[a] == maxActionValue])



''' Parameterized policy with neural network. '''
class PiApproximationWithNN():
    # A nested class that specifies the strucutre of the network.
    # Input layer receives (raw) state representation of s
    # and output probability of selecting each action in s.
    class PiNet(nn.Module):
        def __init__(self, stateDims, numActions):
            super(PiNet, self).__init__()
            self._stateDims = stateDims
            # Three affine operations and one softmax operation
            self.fc1 = nn.Linear(stateDims, 32)
            self.fc2 = nn.Linear(32, 32)
            self.fc3 = nn.Linear(32, numActions)
            self.softmax1 = nn.Softmax()

        def forward(self, x):
            assert len(x) == self._stateDims

            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            x = self.softmax1(x)

            return x

    def __init__(self, state_dims, num_actions, alpha):
        """
        state_dims: the number of dimensions of state space
        action_dims: the number of possible actions
        alpha: learning rate
        """
        self.piNet = self.PiNet(state_dims, num_actions)
        self.piNetOptimizer = optim.Adam(self.piNet.parameters(), lr=alpha)
        self.numActions = num_actions

    def __call__(self,s) -> int:
        # Convert input state s (a numpy array) to a torch tensor
        s = torch.tensor(s, dtype=torch.float32)
        # Add a fake additional dimension for the batch_size dimension.
        s.unsqueeze(0)

        # Compute pi(.|s)
        actionProbabilities = self.piNet.forward(s).detach().numpy()
        # Get rid of the first fake dimension (i.e. the #_batches dimension)
        np.squeeze(actionProbabilities)

        # Return an action sampled from pi(.|s)
        return np.random.choice([a for a in range(self.numActions)], p=actionProbabilities)

    def update(self, s, a, gamma_t, delta):
        """
        s: state S_t
        a: action A_t
        gamma_t: gamma^t
        delta: G-v(S_t,w)
        """
        self.piNetOptimizer.zero_grad()

        # Compute pi(.|s) (a torch.Tensor)
        s = torch.tensor(s, dtype=torch.float32)
        s.unsqueeze(0)
        actionProbabilities = self.piNet.forward(s)

        # Define a loss whose gradient is equal to -gamma_t*delta*grad_ln_pi(a|s)
        # so when we minimize this loss, we update parameters we have
        # theta = theta - (-gamma_t * delta * grad_ln_pi(a|s)) = theta + gamma_t * delta * grad_ln_pi(a|s)
        loss  = -gamma_t * delta / actionProbabilities[a].item() * actionProbabilities[a]

        loss.backward()
        self.piNetOptimizer.step()

