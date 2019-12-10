# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 01:08:16 2019

@author: Mathew
"""
import csv
import numpy as np 
import matplotlib.pyplot as plt

def getSpinningUpLearningReturns(directory):
	with open(str(directory) + 'progress.txt', newline = '') as progress:                                                                                         
	    progress_reader = csv.reader(progress, delimiter='\t')
	    i=0
	    a=[]
	    for line in progress_reader:
	        i+=1
	        if i == 1:
	            continue
	        a.append(line[3])

	returns = np.array(a, dtype=float)
	return returns

if __name__ == "__main__":
	trpo_data = []
	ppo_data = []
	vpg_data = []
	actorcritic_data = []
	sarsa_FA_data = []
	sarsa_tabular_data = []
	for seed in range(5):
		trpo_data.append(getSpinningUpLearningReturns("/home/xiang/Desktop/School/UT_grad_school/First_Year/CS394R_RL/myProject/Experiment1Data/trpo_model_output_seed" + str(seed) + "/"))
		ppo_data.append(getSpinningUpLearningReturns("/home/xiang/Desktop/School/UT_grad_school/First_Year/CS394R_RL/myProject/Experiment1Data/ppo_model_output_seed" + str(seed) + "/"))
		vpg_data.append(getSpinningUpLearningReturns("/home/xiang/Desktop/School/UT_grad_school/First_Year/CS394R_RL/myProject/Experiment1Data/vpg_model_output_seed" + str(seed) + "/"))
		actorcritic_data.append(np.loadtxt("Experiment1Data/ActorCritic_maxEpRet_seed_" + str(seed) + ".npy"))
		sarsa_FA_data.append(np.loadtxt("Experiment1Data/FA_on_policy_n_step_sarsa_seed" + str(seed) + ".npy"))
		sarsa_tabular_data.append(np.loadtxt("Experiment1Data/tabular_on_policy_n_step_sarsa_seed" + str(seed) + ".npy"))

	trpo_mean = np.mean(trpo_data, axis=0)
	trpo_std = np.std(trpo_data, axis=0)
	ppo_mean = np.mean(ppo_data, axis=0)
	ppo_std = np.std(ppo_data, axis=0)
	vpg_mean = np.mean(vpg_data, axis=0)
	vpg_std = np.std(vpg_data, axis=0)
	actorcritic_mean = np.mean(actorcritic_data, axis=0)
	actorcritic_std = np.std(actorcritic_data, axis=0)
	sarsa_FA_mean = np.mean(sarsa_FA_data, axis=0)
	sarsa_FA_std = np.std(sarsa_FA_data, axis=0)
	sarsa_tabular_mean = np.mean(sarsa_tabular_data, axis=0)
	sarsa_tabular_std = np.std(sarsa_tabular_data, axis=0)

	plt.plot(range(500), trpo_mean, label="TRPO")
	plt.fill_between(range(500), trpo_mean - trpo_std, trpo_mean + trpo_std, alpha = 0.25)
	plt.plot(range(500), ppo_mean, label="PPO")
	plt.fill_between(range(500), ppo_mean - ppo_std, ppo_mean + ppo_std, alpha = 0.25)
	plt.plot(range(500), vpg_mean, label="REINFORCE w/ baseline")
	plt.fill_between(range(500), vpg_mean - vpg_std, vpg_mean + vpg_std, alpha = 0.25)
	plt.plot(range(500), actorcritic_mean, label="Actor-Critic")
	plt.fill_between(range(500), actorcritic_mean - actorcritic_std, actorcritic_mean + actorcritic_std, alpha = 0.25)
	plt.plot(range(500), sarsa_FA_mean, label="n-step Sarsa with tiles")
	plt.fill_between(range(500), sarsa_FA_mean - sarsa_FA_std, sarsa_FA_mean + sarsa_FA_std, alpha = 0.25)
	plt.plot(range(500), sarsa_tabular_mean, label="tabular n-step Sarsa")
	plt.fill_between(range(500), sarsa_tabular_mean - sarsa_tabular_std, sarsa_tabular_mean + sarsa_tabular_std, alpha = 0.25)


	plt.legend(loc=0, fontsize = 'xx-large')
	plt.xlabel("# of Epochs (600 steps / epoch)", size=30)
	plt.ylabel("Max Epoch Retrun", size=30)
	plt.xticks(fontsize=15)
	plt.yticks(fontsize=15)
	plt.title("Experiment 1: Performance of Various Algorithms", fontsize=35)

	plt.show()