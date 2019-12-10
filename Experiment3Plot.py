import matplotlib.pyplot as plt 
import numpy as np

# Load data
representation0_data = np.load("trpo_representation0.npy")
representation1_data = np.load("trpo_representation1.npy")
representation2_data = np.load("trpo_representation2.npy")


# Prepare plotting apparatus
x = ["", "Representation 0 (Default)\n" + r"(x, y, $\theta$, $\theta$')", "Representation 1\n" + r"($\Delta$x, $\Delta$y, $\Delta\theta$, $\theta$')", "Representation 2\n" + r"(x, y, $\theta$, $\theta$', $G_x$, $G_y$)"]
x_pos = [i for i in range(4)]


# plot error bars
# plt.bar(x_pos, y, yerr=yerr)
fig = plt.figure(1, figsize=(6, 4))
# Create an axes instance
ax = fig.add_subplot(111)
bp = ax.boxplot([representation0_data, representation1_data, representation2_data], showfliers=True, showcaps = False,patch_artist=True)
for box in bp['boxes']:
    # change outline color
    box.set( color='tab:orange', linewidth=2)
    # change fill color
    box.set( facecolor = 'tab:orange' )
## Change median color 
for median in bp['medians']:
    median.set(color='#F0B47B', linewidth=2)
## change the style of fliers and their fill
for flier in bp['fliers']:
    flier.set(marker='o', color='#e7298a', alpha=0.5,linewidth=3)
plt.grid()

plt.xlabel("Trained with Different State Representations", size=25)
plt.ylabel("Return / Episode", size=25)
plt.title("Effectiveness of Various State Representations", size=35)
plt.xticks(x_pos, x, fontsize=19)
plt.yticks(fontsize=15)
plt.show()
