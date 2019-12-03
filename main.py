import sys
import QLearning
# import ActorCritic

if __name__ == "__main__":
	d = {0 : "One-step Q learning",
		 1 : "Actor-Critic"}

	algo = int(input("Choose an algorithm to run (type in a number from 0 to " + str(len(d)-1) + "). \n" + str(d) + "\n\n"))

	if algo == 0:
		QLearning.QLearningAlgorithm()




