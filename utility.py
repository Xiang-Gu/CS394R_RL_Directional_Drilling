import numpy as np

def evaluation(env, evalPolicy, episodeNum):
    s = env.reset()
    done = False
    while not done:
        a = evalPolicy.action(s)
        next_s, r, done, info = env.step(a)
        if done is True:
        	env.render(mode='human', close=False, show=False, save=True, r_info = True)
        s = next_s

def computeMaxEpRetFromRewards(rewards, episodeLengths, epochSteps, gamma):
	assert len(rewards) == len(epochSteps)
	assert len(rewards) == np.sum(episodeLengths)

	result = -999999

	# idx should point to the start of each episode
	idx = 0
	for length in episodeLengths:
		# Compute episode return for current episode (an episode of length length).
		episodeRet = 0.
		for jdx in range(idx, idx + length):
			episodeRet += gamma**(jdx-idx) * rewards[jdx]
		
		result = max(result, episodeRet)
		idx += length

	return result



'''
Heuristic says, for best performance of tiling coding,
choose the number of tilings to be some power of 2 and be
larger than or equal to four times state dimension.
'''
def findProperNumberOfTilings(stateDimension):
	for power in range(20):
		if 2**power >= 4 * stateDimension:
			return 2**power