import matplotlib.pyplot as plt 
import numpy as np

# Load data
representation0_data = np.load("trpo_representaion0.npy")
representation1_data = np.load("trpo_representaion1.npy")
representation2_data = np.load("trpo_representaion2.npy")


# Prepare plotting apparatus
x = ["", r"Representation 0 (x,y,$\theta$,$\theta$')", r"Representation 1 ($\delta$x, $\delta$y, $\delta \theta$, $\theta$')", r"Representation 2 (x,y,$\theta$,$\theta$', $G_x$, $G_y$)"]
x_pos = [i for i in range(4)]


# plot error bars
# plt.bar(x_pos, y, yerr=yerr)
fig = plt.figure(1, figsize=(6, 4))
# Create an axes instance
ax = fig.add_subplot(111)
bp = ax.boxplot([representation0_data, representation1_data, representation2_data], showfliers=True, showcaps = False,patch_artist=True)
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

plt.xlabel("Trained with Different State Representations", size=20)
plt.ylabel("Return / Episode", size=20)
plt.title("Effectiveness of Various State Representations", size=30)
plt.xticks(x_pos, x)
plt.show()
