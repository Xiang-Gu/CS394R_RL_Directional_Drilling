import tensorflow as tf
from spinup import trpo, ppo, vpg
import spinup.utils.logx
import numpy as np
import matplotlib.pyplot as plt
import gym

def run_algo(output_dir, seed):
	#Note it's best to run the algorithm in a function to allow for multiple runs
	#in the same Ipython console:
	#I believe what it's doing is avoiding contamination of tensor flow objects 
	#which can cause the reloaded networks to lose their trained parameters

	def env_fn():
		try:
			import env.tightWellEnvironment
			return gym.make('TightWellEnv-v0')
		except:
			env_dict = gym.envs.registration.registry.env_specs.copy()
			for env in env_dict:
				if 'CustomEnv-v0' in env:
					print('Remove {} from registry'.format(env))
					del gym.envs.registration.registry.env_specs[env]
			import env.tightWellEnvironment
			return gym.make('TightWellEnv-v0')
			print('gym environment already registered.')
            
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	
	logger_kwargs = dict(output_dir = output_dir)
	
	ac_kwargs = dict(hidden_sizes=(32,))
	with tf.Session(graph=tf.Graph()):
		#ppo or trpo or vpg
		trpo(env_fn,gamma=1., steps_per_epoch=600, epochs=500, seed=seed, ac_kwargs=ac_kwargs, logger_kwargs = logger_kwargs)
	return

"""Run"""
#   Run algorithm
Rs = []
for seed in range(1):
	print("running trpo with seed " + str(seed))
	np.random.seed(seed)
	output_dir = '/mnt/f/School/Reinforcement Learning Theory and Practice/Project/directional_drilling-master/Experiment0Data/model_output_trpo'
	run_algo(output_dir, seed)


	"""Load Algorithm Outputs"""
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	sd = spinup.utils.logx.restore_tf_graph(sess, output_dir + '/simple_save')
	pi = sd['pi']
	logp = sd['logp']
	x_ph = sd['x']
	a_ph = sd['a']
	v = sd['v']

	print("Deploying learned policy to determinstic environment to compare with MPC trajectory\n.")

	import env.largeVaringGoalRep0Environment 
	env = gym.make('TightWellEnv-v0')

	# Test the learned policy over 1000 episode on stochastic environment.
	for episode in range(1):
		s = env.reset()
		done = False
		while not done:
			# Use determinstic policy
			a_all = np.arange(0, 21, 1, dtype='int32')
			p_a = sess.run(logp, feed_dict={x_ph : s.reshape(1,-1), a_ph : a_all})
			p_a = np.exp(p_a)
			a = np.argmax(p_a)

			# Use stochastic policy
			# a = sess.run(pi,feed_dict={x_ph:s.reshape((-1,4))})
			
			s, r, done, _ = env.step(a)
		simstates = env.render(save=False, show=True)
np.save('trpo_trajectory.npy',simstates)