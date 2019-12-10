import numpy as np 
import matplotlib.pyplot as plt

trpo_traj = np.load('trpo_trajectory.npy')
mpc_traj = np.load('mpc_trajectory.npy')

#Plot trajectories of RL Policy vs MPC Policy
legend = ['TRPO Policy','MPC']
fig = plt.figure(figsize=(8,5))
LW = 3
a=1
FS = 20
plt.plot(trpo_traj[:,1],-trpo_traj[:,0],color='C0',LineWidth=LW , alpha=a)
plt.plot(mpc_traj[:,1],-mpc_traj[:,0],color='C1',LineWidth=LW,alpha = a)
plt.legend(legend, fontsize=15)
plt.xlim([55,420])
plt.xticks(fontsize=FS/1.5)
plt.yticks(fontsize=FS/1.5)

#plt.ylim([-(1000/2+50),0])
plt.title('TRPO Policy and MPC Comparison', fontsize = FS)
plt.xlabel('Lateral Distance (ft)',fontsize=FS)
plt.ylabel('Vertical Depth (ft)',fontsize=FS)
plt.grid()
plt.show()

