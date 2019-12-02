import random
random.seed(0) # Get consistent hashing for tile coding
from tiles3 import tiles, IHT
import numpy as np

''' Base class for function approximator'''
class ValueFunctionWithApproximation(object):
	def __call__(self,s) -> float:
		"""
		return the value of given state; \hat{v}(s)

		input:
			state
		output:
			value of the given state
		"""
		#No need to implement here- they are implemented in the child class that inherets this
		raise NotImplementedError()

	def update(self,alpha,G,s_tau):
		"""
		Implement the update rule;
		w <- w + \alpha[G- \hat{v}(s_tau;w)] \nabla\hat{v}(s_tau;w)

		input:
			alpha: learning rate
			G: TD-target. It can be formed by any valid target like TD_target or Monte-Carlo target.
			s_tau: target state for updating (yet, update will affect the other states)
		ouptut:
			None
		"""
		#No need to implement here- This is a base class inhereted by other classes
		raise NotImplementedError()

''' A linear function approximator using tile coding '''
class ValueFunctionWithLinearApproximationUsingTiles(ValueFunctionWithApproximation):
	def __init__(self):
		"""
		state_low: possible minimum value for each dimension in state
		state_high: possible maimum value for each dimension in state
		num_actions: the number of possible actions
		num_tilings: # tilings
		tile_width: tile width for each dimension
		"""
		self.stateLow = np.array([0., 0., -10., -25.])
		self.stateHigh = np.array([1000, 500, 100, 25])
		self.numTilings = 16
		self.numActions = 21
		# resolution means the fine-ness we want in each dimension. E.g., 3
		# in the first dimension means we want two x-coordinates x_a and x_b that is
		# such that |x_a - x_b| > 3 (feet) to have different features (and thus 
		# different estimated values).
		resolution = np.array([3, 3, 1, 1])
		# Used to re-scale the range of each dimension in state to use Sutton's 
		# tile coding interface.
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



