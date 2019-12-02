import numpy as np
from environment import AutoDrill
from policy import EpsilonGreedy
from functionApproximator import ValueFunctionWithLinearApproximationUsingTiles


def evaluation(env, evalPolicy, Q, episodeNum):
    s = env.reset()
    done = False
    while not done:
        a = evalPolicy.action(s, Q) # Online policy (epsilon-greedy)
        next_s, r, done = env.step(a)
        if done is True:
            env.disp_traj("episode " + str(episodeNum) + " reward = " + str(round(r,2)), save_flag=1, show_flag=0)
        s = next_s


if __name__ == "__main__":
    env = AutoDrill()
    Q = ValueFunctionWithLinearApproximationUsingTiles()
    epsilonGreedyPolicy = EpsilonGreedy(0.05)
    greedyPolicy = EpsilonGreedy(0.)

    numEpisodes = 50000

    print("Initiating Learning with Q-learning with function approximation")

    for episode in range(numEpisodes):
        # Initialize S, A, and done=False
        state = env.reset()
        reward = 0.
        done = False

        while not done:
            # Choose A from Q
            action = epsilonGreedyPolicy.action(state, Q)
            # Take A and observe R and S'.
            nextState, reward, done = env.step(action)
            # Compute Q-learning target (R + gamma * max_a Q(s', a))
            target = reward + env.gamma * np.max(Q[nextState])
            # Update our function approximator
            Q.update(0.006, target, state, action)

            state = nextState

        print(str(episode) + "th episode: reward = " + str(reward))

        if episode % 1000 == 0:
            evaluation(env, greedyPolicy, Q, str(episode))


#     R=np.zeros(numEpisodes)
#     R_greedy=[]
#     start_estimate = []
#     for i in tqdm(range(numEpisodes)): 
#         s = drill_world.reset()
#         traj = []
#         done = False
#         while done is False:
#             a = episodic_policy.action(V,s) # Online policy (epsilon-greedy)
#             next_s, r, done = drill_world.step(a)
            
# #            print('State')
# #            print(s)
# #            print('State2Feature:')
# #            print(V.s2f(np.append(s,a)))
#             traj.append((s, a, r, next_s))
#             s = next_s
            
#             if done is True:
# #                print('Reward at T: '+str(r))
# #                print('Length of Traj: '+str(len(traj)))
# #                print('Initial action taken: '+str(traj[0][1]))
#                 if i%1000 == 0: 
#                     print('Reward: ',r)
# #                    drill_world.disp_traj(i,0)
#                     print('Performing Evaluation')
#                     r_greedy = evaluation(drill_world,eval_policy,i)
#                     R_greedy.append(r_greedy)
                    
#         R[i]=r
#         a_initial_opt = eval_policy.action(V,[0,0,0,0])
#         start_estimate.append(V.__call__(np.append(np.array((0,0,0,0)),0)))
#         #n_steps is set to a large value of 100 to implement monte carlo updates
#         episodic_semi_gradient_sarsa(drill_world.spec, traj,100, .006, V)
   
#     #Save the arrays to the .npy binary data type
#     w=V.getWeight()
#     print('Size of w in GB: ' + str(w.shape[0]*64/10**9/8))
#     np.save('w8.npy',w)
#     np.save('R8.npy',R)
#     np.save('R8_eval.npy',R_greedy)
