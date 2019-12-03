import random
random.seed(1) # Get consistent hashing for tile coding
from tiles3 import tiles
import numpy as np

''' A linear function approximator using tile coding '''
class ValueFunctionWithLinearApproximationUsingTiles():
	def __init__(self):
		self.stateLow = np.array([0., 0., -10., -25.])
		self.stateHigh = np.array([1000, 500, 100, 25])
		self.numTilings = 16
		self.numActions = 21

		# resolution means the fine-ness we want in each dimension. E.g., 3
		# in the first dimension means we want two x-coordinates x_a and x_b 
		# such that |x_a - x_b| > 3 (feet) to have different features (and thus 
		# different estimated values).
		resolution = np.array([3, 3, 1, 1])
		# Used to re-scale the range of each dimension in state to use Sutton's 
		# tile coding software interface.
		self.scalingFactor = 1 / (self.numTilings * resolution)

		# One advantage of Sutton's tilecoder is the use of hashing. In our problem,
		# the state range is too large so that the size of the resulting weight vector
		# is also prohibitively large. However, we can use his tilecoder by specifying
		# a upper range of returned indices regardless of the state space and tiling numbers.
		# Since the tilecoder is guaranteed to return different active tiles for different
		# input state (or state-action pair) as long as there is unused indices left.
		# This way, we implicitly achieve the desired property of "dynamic tile coding" where
		# the total number of tiles stays unchanged but give more resolution to state spaces
		# that are visited more often.
		self.maxSize = 100000000
		self.w = np.zeros(self.maxSize)

	def mytiles(self, s, a):
		'''
		Wrapper method to produce binary feature of state-action pair (s, a). It
		returns a list of self.numTilings numbers that denote the indices of active tile.
		'''
		assert(len(s) == 4)

		return tiles(self.maxSize, self.numTilings, list(self.scalingFactor * s), ints=[a])

	def __call__(self, s, a):
		activeIndices = self.mytiles(s, a)
		return np.sum(self.w[activeIndices])

	def __getitem__(self, s):
		'''
		Overload the indexing operator [] so that users of this class
		can use something like Q[state] to get a list of estimated
		state-action values.
		'''
		result = np.zeros(self.numActions)
		for a in range(self.numActions):
			result[a] = self.__call__(s, a)
		return result

	def update(self, alpha, G, s, a):
		activeIndices = self.mytiles(s, a)
		estimatedStateActionValue = self.__call__(s, a)

		self.w[activeIndices] += alpha * (G - estimatedStateActionValue)


''' A non-linear function approximator using neural networks with Pytorch '''



