def evaluation(env, evalPolicy, Q, episodeNum):
    s = env.reset()
    done = False
    while not done:
        a = evalPolicy.action(s) # Online policy (epsilon-greedy)
        next_s, r, done = env.step(a)
        if done is True:
            env.disp_traj("episode " + str(episodeNum) + " reward = " + str(round(r,2)), save_flag=1, show_flag=0)
        s = next_s
