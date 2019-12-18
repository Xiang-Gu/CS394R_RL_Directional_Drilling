This is the course project for CS394R Reinforcement Learning -- Theory and Practice

Team Member: Xiang Gu, Alexander Matthew Keller

In this project, we explored the possibility of using reinforcement learning techniques to directional drilling in the oil&gas industry. Briefly, this is what we did:

- We formulated the (simplified) directional drilling process as a Markov Decision Process (MDP). 

- We tried to apply several classical RL algorithms to solve this problem including Tabular n-step Sarsa, n-step Sarsa with Tile Coding, One-step Actor-Critic, Q-learning with Tile Coding, Vanilla Policy Gradient (VPG), Trust Region Policy Optimization (TRPO), and Proximal Policy Optimization (PPO). We also implemented a conventional solver for this problem using Model Predictive Control (MPC) and compared the performance with those RL algorithm. (Experiment 1)

- We studied the robustness/generalization of the policy learned from RL algorithms. Namely, we looked at the generalization performance when there are stochasticity in the test environment and what is the best practical way to train a robust policy. (Experiment 2). In addition, we also investigated the generalization performance w.r.t. different goal position during test time using different state representation. (Experiment 3)

To run the learning algorithms to get data (already done):
	1). (Tabular n-step Sarsa): python3 n_step_sarsa_tabular/main_code.py
	2). (n-step Sarsa with Tile Coding): python3 n_step_sarsa_fa/main_code.py
	3). (One-step Actor-Critic): python3 ActorCritic.py
	4). (Q-learning with Tile Coding): python3 QLearning.py
	5). (VPG, TRPO, PPO): python3 VPG_TRPO_PPO.py   # change one line of code to vpg, trpo, or ppo to run different algorithms

To plot figures of Experiment 1:
	python3 Experiment1Plot.py

To plot figures of Experiment 2:
	python3 Experiment2Plot.py
	
To plot figures of Experiment 3:
	python3 Experiment3Plot.py



