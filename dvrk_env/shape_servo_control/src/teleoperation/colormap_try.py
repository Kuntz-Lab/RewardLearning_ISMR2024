
import numpy as np
import pickle
import timeit
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

import os

import pandas as pd 

from matplotlib import cm
from matplotlib.colors import ListedColormap,LinearSegmentedColormap

# colors = plt.cm.colors.CSS4_COLORS
# # print(colors)

# data = np.random.uniform(size=(100,100))
# new_inferno = cm.get_cmap('inferno', 5)# visualize with the new_inferno colormaps
# top = cm.get_cmap('Oranges_r', 128) # r means reversed version
# bottom = cm.get_cmap('Blues', 128)# combine it all
# newcolors = np.vstack((top(np.linspace(0, 1, 128)),
#                        bottom(np.linspace(0, 1, 128))))# create a new colormaps with a name of OrangeBlue
# orange_blue = ListedColormap(newcolors, name='OrangeBlue')
# plt.pcolormesh(data, cmap = orange_blue)
# plt.colorbar()
# plt.show()



# Sample data
# categories = ['Category 1', 'Category 2', 'Category 3']
# data1 = [10, 15, 20]
# data2 = [8, 12, 18]
# data3 = [5, 10, 15]

# # Bar width
# bar_width = 0.2

# # Set up positions for bars
# bar_positions1 = np.arange(len(categories))
# bar_positions2 = bar_positions1 + bar_width
# bar_positions3 = bar_positions1 + 2 * bar_width

# # Set up colormap
# cmap = plt.cm.get_cmap('viridis')

# # Plotting the bars with colormap
# plt.bar(bar_positions1, data1, width=bar_width, color=cmap(0.2), label='Data 1')
# plt.bar(bar_positions2, data2, width=bar_width, color=cmap(0.5), label='Data 2')
# plt.bar(bar_positions3, data3, width=bar_width, color=cmap(0.8), label='Data 3')

# # Set up x-axis ticks and labels
# plt.xticks(bar_positions2, categories)

# # Add legend below the x-axis
# plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=3)

# # Set axis labels and title
# plt.xlabel('Categories')
# plt.ylabel('Values')
# plt.title('Multiple Bar Charts with Colormap and Legend Below X-axis')

# plt.show()





# Sample data
categories = ['Category A', 'Category B', 'Category C']
data1 = [10, 15, 20]
data2 = [8, 12, 15]
data3 = [5, 10, 12]

# Create subplots with a common legend
fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

# Bar width
bar_width = 0.2

# Positions for each bar group
r = np.arange(len(categories))

# Plot data1 on the first subplot
axs[0].bar(r, data1, width=bar_width, edgecolor='grey', label='Data 1')
axs[0].set_title('Plot 1')
axs[0].set_xticks(r)
axs[0].set_xticklabels(categories)
axs[0].legend()

# Plot data2 on the second subplot
axs[1].bar(r, data2, width=bar_width, edgecolor='grey', label='Data 2')
axs[1].set_title('Plot 2')
axs[1].set_xticks(r)
axs[1].set_xticklabels(categories)
axs[1].legend()

# Plot data3 on the third subplot
axs[2].bar(r, data3, width=bar_width, edgecolor='grey', label='Data 3')
axs[2].set_title('Plot 3')
axs[2].set_xticks(r)
axs[2].set_xticklabels(categories)
axs[2].legend()

# Adjust layout to prevent clipping of titles
plt.tight_layout()

# Show the plot
plt.show()