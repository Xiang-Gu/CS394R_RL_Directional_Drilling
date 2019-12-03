import random
random.seed(1) # Get consistent hashing for tile coding
from tiles3 import tiles, IHT
import numpy as np
from utility import findProperNumberOfTilings
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim

''' A linear function approximator using tile coding '''
class LinearApproximatorOfActionValuesWithTile():
	def __init__(self, alpha, stateLow, stateHigh, numActions):
		assert(len(stateLow) == len(stateHigh))

		self.alpha = alpha
		self.stateLow = stateLow
		self.stateHigh = stateHigh
		self.numActions = numActions

		self.numTilings = findProperNumberOfTilings(len(stateLow)) # 16
		self.tileWidth = np.array([3, 3, 1, 1]) * self.numTilings # tileWidth = ultimate resolution * numTilings
		# Used to re-scale the range of each dimension in state to use Sutton's 
		# tile coding software interface.
		self.scalingFactor = 1 / self.tileWidth

		# One advantage of Sutton's tilecoder is the use of hashing. In our problem,
		# the state range is too large so that the size of the resulting weight vector
		# is also prohibitively large. However, we can use his tilecoder by specifying
		# a upper range of returned indices regardless of the state space and tiling numbers.
		# Since the tilecoder is guaranteed to return different active tiles for different
		# input state (or state-action pair) as long as there is unused indices left.
		# This way, we implicitly achieve the desired property of "dynamic tile coding" where
		# the total number of tiles stays unchanged but give more resolution to state spaces
		# that are visited more often.
		maxSize = np.prod(np.ceil((self.stateHigh - self.stateLow) / self.tileWidth), dtype=int) * self.numTilings * self.numActions
		self.iht = IHT(maxSize)
		self.w = np.zeros(maxSize)

	def mytiles(self, s, a):
		'''
		Wrapper method to produce binary feature of state-action pair (s, a). It
		returns a list of self.numTilings numbers that denote the indices of active tile.
		'''
		assert(len(s) == len(self.stateLow) and 0 <= a < self.numActions)

		return tiles(self.iht, self.numTilings, list(self.scalingFactor * s), ints=[a])

	def __call__(self, s, a=None):
		'''
		Compute estimated Q(s) (which returns an ndarray of Q(s,a)'s) or Q(s, a).
		'''
		if a == None:
			result = np.zeros(self.numActions)
			for a in range(self.numActions):
				result[a] = self.__call__(s, a)
			return result
		else:
			activeIndices = self.mytiles(s, a)
			return np.sum(self.w[activeIndices])

	def update(self, s, a, G):
		activeIndices = self.mytiles(s, a)
		estimatedStateActionValue = self.__call__(s, a)

		self.w[activeIndices] += self.alpha * (G - estimatedStateActionValue)


''' A non-linear function approximator using neural networks with Pytorch '''
class NonLinearApproximatorOfStateValuesWithNN():
	# A Nested class that specifies the structure of the network.
	# Input layer receives (raw) state representation of s
	# and output estimated state value of s.
	class ValueNet(nn.Module):
		def __init__(self, stateDimension):
			super(ValueNet, self).__init__()
			self._stateDims = stateDimension

			# Three affine operations and one softmax operation
			self.fc1 = nn.Linear(stateDimension, 32)
			self.fc2 = nn.Linear(32, 32)
			self.fc3 = nn.Linear(32, 1)

		def forward(self, x):
			assert len(x) == self._stateDims

			x = F.relu(self.fc1(x))
			x = F.relu(self.fc2(x))
			x = self.fc3(x)

			return x

	def __init__(self, alpha, stateLow, stateHigh):
		assert(len(stateLow) == len(stateHigh))

		self.valueNet = self.ValueNet(len(stateLow))
		self.valueNetOptimizer = optim.Adam(self.valueNet.parameters(), lr=alpha)

	def __call__(self, s) -> float:
		# Convert input state s (a numpy array) to a torch tensor
		s = torch.tensor(s, dtype=torch.float32)
		# Add a fake additional dimension for the #_samples dimension.
		s.unsqueeze(0)

		return self.valueNet.forward(s).item()

	def update(self, s, G):
		self.valueNetOptimizer.zero_grad()
		
		# forward pass to compute estimated state value of s
		s = torch.tensor(s, dtype=torch.float32)
		s.unsqueeze(0)
		estimatedStateValue = self.valueNet.forward(s)

		# define loss whose gradient is -(semi_grad) so that 
		# w = w - alpha * (-(semi_grad)) = w + alpha * semi_grad
		criterion = nn.MSELoss()
		G = torch.tensor([G], dtype=torch.float32)
		G.unsqueeze(0)
		loss = 0.5 * criterion(estimatedStateValue, G)

		# update parameter
		loss.backward()
		self.valueNetOptimizer.step()




