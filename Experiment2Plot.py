import matplotlib.pyplot as plt 
import numpy as np
# Load data
average_parameter_data = np.load("Experiment2Data/trpo_trained_on_averageParameterEnv.npy")
noisy_next_state_data = np.load("Experiment2Data/trpo_trained_on_noisyNextStateEnv.npy")
random_starting_state_data = np.load("Experiment2Data/trpo_trained_on_randomStartingStateEnv.npy")
stochastic_environment_data = np.load("Experiment2Data/trpo_trained_on_stochasticEnv.npy")

# Prepare plotting apparatus
x = ["", "Average Parameter", "Noisy Next State", "Random Starting State", "Stochastic Environment (Unreal)"]
x_pos = [i for i in range(5)]
y = [np.mean(average_parameter_data), np.mean(noisy_next_state_data), np.mean(random_starting_state_data), np.mean(stochastic_environment_data)]
yerr = [np.std(average_parameter_data), np.std(noisy_next_state_data), np.std(random_starting_state_data), np.std(stochastic_environment_data)]


# plot error bars
# plt.bar(x_pos, y, yerr=yerr)
fig = plt.figure(1, figsize=(6, 4))
# Create an axes instance
ax = fig.add_subplot(111)
bp = ax.boxplot([average_parameter_data, noisy_next_state_data, random_starting_state_data, stochastic_environment_data], showfliers=True, showcaps = False,patch_artist=True)
for box in bp['boxes']:
    # change outline color
    box.set( color='tab:blue', linewidth=2)
    # change fill color
    box.set( facecolor = 'tab:blue' )
## Change median color 
for median in bp['medians']:
    median.set(color='#A9D0F5', linewidth=2)
## change the style of fliers and their fill
for flier in bp['fliers']:
    flier.set(marker='o', color='#e7298a', alpha=0.5,linewidth=3)
plt.grid()

plt.xlabel("Trained on Different Environment", size=20)
plt.ylabel("Return / Episode", size=20)
plt.title("Robustness Test of Policy Trained on Various Environments", size=30)
plt.xticks(x_pos, x)
plt.show()
