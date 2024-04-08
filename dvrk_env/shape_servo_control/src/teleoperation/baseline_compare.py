
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
import matplotlib.colors as mcolors


'''
create baseline figures for trace curve and cut
'''

def get_episode_data(data_recording_path, episode_idx):
    file = os.path.join(data_recording_path, f"episode {episode_idx}.pickle")
    with open(file, 'rb') as handle:
        data = pickle.load(handle)
    
    return data


def plot_curves(xs, ys_list, title, path, x_label, y_label, curve_labels):
    fig, ax = plt.subplots()
    assert(len(curve_labels)==len(ys_list))
    if len(ys_list)==2:
        ax.plot(xs, ys_list[0], label=curve_labels[0])
        ax.plot(xs, ys_list[1], label=curve_labels[1], linestyle='dashed')
    else:
        for idx in range(len(ys_list)):
            ax.plot(xs, ys_list[idx], label=curve_labels[idx])
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    os.makedirs(path, exist_ok=True)
    fig.savefig(f"{path}/{title}.png")
    plt.cla()
    plt.close(fig) 


def plot_bar_chart(xs, ys, title, path, x_label, y_label, tick_labels, colors=['red', 'green', 'brown', 'orange', 'blue']):
    fig, ax = plt.subplots()
    bars = ax.bar(xs, ys, width=8 ,align='center', tick_label=tick_labels)
    for i, _ in enumerate(bars):
        bars[i].set_color(colors[i])
    # r = patches.Rectangle((xs[0]-6, 0), 12, max(ys),
    #                   lw=3, ls='--',
    #                   edgecolor='gray', facecolor='none',
    #                   clip_on=False
    #                  )
    # ax.add_patch(r)
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    os.makedirs(f"{path}", exist_ok=True)
    fig.savefig(f"{path}/{title}.png")
    plt.cla()
    plt.close(fig)

def plot_bar_chart_2col(categories, bar_labels_per_cat, bar_values_per_cat, x_label, y_label, title, path):
    cmap = plt.cm.get_cmap('tab20c')
    colors = [17,0,1,2,4]
    bar_positions_list = [np.arange(len(categories))*1.8]
    num_bar_per_cat = len(bar_labels_per_cat)
    bar_width = 0.2
    for i in range(1, num_bar_per_cat, 1):
        bar_positions = bar_positions_list[i-1] + bar_width
        bar_positions_list.append(bar_positions)

    fig, ax = plt.subplots(figsize=(8,7))
    # plotting the bars
    for i, bar_label in enumerate(bar_labels_per_cat):
        ax.bar(bar_positions_list[i], bar_values_per_cat[i], width=bar_width, label=bar_label, color=cmap(colors[i]))

    # Set up x-axis ticks and labels
    ax.set_xticks(bar_positions_list[2], categories, fontsize=12)

    # Add legend below the x-axis
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=len(bar_labels_per_cat), fontsize=12)

    #ax.set_xlabel(x_label, fontdict={"fontsize":11})
    ax.set_ylabel(y_label, fontdict={"fontsize":12})
    ax.set_title(title, fontdict={"fontsize":18})
    fig.savefig(f"{path}/{title}.png")


    # dict_df = {}
    # dict_df[x_label] = categories
    # for i, bar_label in enumerate(bar_labels_per_cat):
    #     dict_df[bar_label] = bar_values_per_cat[i]
    # df = pd.DataFrame(dict_df)
    # ax = df.plot(x=x_label, y=bar_labels_per_cat, kind="bar", colormap=cmap, rot=0, width=0.6)
    # fig = ax.get_figure()  
    # ax.legend(loc="upper left", mode = "expand", ncol = len(bar_labels_per_cat)) 
    # ax.set_xlabel(x_label, fontdict={"fontsize":11})
    # ax.set_ylabel(y_label, fontdict={"fontsize":11})
    # ax.set_title(title, fontdict={"fontsize":12})

    # fig.savefig(f"{path}/{title}.png")

def plot_bar_chart_multiple(categories, bar_labels_per_cat, bar_values_per_cat, x_label, y_label, title, path):
    bar_values_per_cat = bar_values_per_cat.transpose((1,0))
    cmap = plt.cm.get_cmap('tab20c')
    colors = [17,0,1,2,4]
    bar_width = 0.2
    bar_positions = np.arange(0,bar_width*len(bar_labels_per_cat), bar_width)
    num_bar_per_cat = len(bar_labels_per_cat)
    #print(bar_positions.shape)

    fig, axs = plt.subplots(1, len(categories), figsize=(12, 5))
    for k in range(len(categories)):
        # plotting the bars
        for i, bar_label in enumerate(bar_labels_per_cat):
            axs[k].bar(bar_positions[i], bar_values_per_cat[k][i], width=bar_width, label=bar_labels_per_cat[i], color=cmap(colors[i]))
        axs[k].set_title(categories[k], fontdict={"fontsize":18})
        axs[k].set_ylabel(y_label, fontdict={"fontsize":18})
        axs[k].set_xticks([])

    # Add legend below the x-axis
    axs[0].legend(loc='lower center', bbox_to_anchor=(1, -0.15), fancybox=True, shadow=True, ncol=len(bar_labels_per_cat), fontsize=15)
    
    #fig.set_title(title, fontdict={"fontsize":18})
    fig.savefig(f"{path}/{title}.png")
  


if __name__ == "__main__":
    plot_path = "/home/dvrk/RL_data/baselineFigures_dummy"
    os.makedirs(plot_path, exist_ok=True) 

    max_num_balls_list = []
    root_paths_list = []
    labels_list = []
    data_recording_paths_list = []
    max_episodes_list = []
    min_episodes_list = []
    step = 10
    labels = ["GT reward", "BC perfect", "BC total", "BC pair", "our method"]

    ### Sphere ###
    max_num_balls = 2
    root_paths = ["/home/dvrk/RL_data/traceCurve/gt_reward_eef_pos_2ball", "/home/dvrk/RL_data/traceCurve/BC_perfect", "/home/dvrk/RL_data/traceCurve/BC_total", "/home/dvrk/RL_data/traceCurve/BC_pair", "/home/dvrk/RL_data/traceCurve/learned_reward_eef_pos_2ball"]
    data_recording_paths = [f"{root_paths[i]}/data" for i in range(len(root_paths))]
    max_episodes = [741, 91, 81, 81, 601] # last episode idx + 1 for each plot
    min_episodes = [640, 10, 10, 10, 500] # the minimum episode idx for each plot
    

    max_num_balls_list.append(max_num_balls)
    root_paths_list.append(root_paths)
    labels_list.append(labels)
    data_recording_paths_list.append(data_recording_paths)
    max_episodes_list.append(max_episodes)
    min_episodes_list.append(min_episodes)

    ### cut ####
    max_num_balls = 1
    root_paths = ["/home/dvrk/RL_data/cut/gt_reward_eef_pos", "/home/dvrk/RL_data/cut/BC_perfect", "/home/dvrk/RL_data/cut/BC_total", "/home/dvrk/RL_data/cut/BC_pair", "/home/dvrk/RL_data/cut/learned_reward_eef_pos"]
    data_recording_paths = [f"{root_paths[i]}/data" for i in range(len(root_paths))]
    max_episodes = [491, 111, 91, 291, 801] # last episode idx + 1 for each plot
    min_episodes = [400, 10, 10, 10, 400] # the minimum episode idx for each plot

    max_num_balls_list.append(max_num_balls)
    root_paths_list.append(root_paths)
    labels_list.append(labels)
    data_recording_paths_list.append(data_recording_paths)
    max_episodes_list.append(max_episodes)
    min_episodes_list.append(min_episodes)
   
    tasks_list = ["Sphere", "Cutting"]
    num_categories = len(max_num_balls_list)
    avg_rewards_all_cat = []
    avg_num_reaches_all_cat = []
    percent_num_env_success_all_cat = []
    for k in range(num_categories):
        root_paths = root_paths_list[k]
        min_episodes = min_episodes_list[k]
        max_episodes = max_episodes_list[k]
        data_recording_paths = data_recording_paths_list[k]
        max_num_balls = max_num_balls_list[k]
    ######### plot rewards and success rate #############
        num_plots = len(root_paths)
        avg_rewards_all_plots = []
        avg_num_reaches_all_plots = []
        percent_num_env_success_all_plots = []
        for i in range(num_plots):
            avg_rewards = []
            avg_num_reaches = []
            percent_num_env_success = []
            for episode_idx in range(min_episodes[i], max_episodes[i], step):
                
                data = get_episode_data(data_recording_paths[i], episode_idx=episode_idx)
                
                rewards = data["gt_cum_reward"]
                avg_reward = np.sum(rewards)/len(rewards)
                avg_rewards.append(avg_reward)
            
                num_balls_reached = data["num_balls_reached"]
                avg_num_reach = np.sum(num_balls_reached)/len(num_balls_reached)
                avg_num_reaches.append(avg_num_reach)
                
                num_env_success = np.count_nonzero(num_balls_reached==max_num_balls)
                num_env_success = num_env_success / len(num_balls_reached)
                percent_num_env_success.append(num_env_success)

            avg_reward_over_time = sum(avg_rewards)/len(avg_rewards)
            avg_num_reach_over_time = sum(avg_num_reaches)/len(avg_num_reaches)
            percent_num_env_success_over_time = sum(percent_num_env_success)/len(percent_num_env_success)

            avg_rewards_all_plots.append(avg_reward_over_time)
            avg_num_reaches_all_plots.append(avg_num_reach_over_time)
            percent_num_env_success_all_plots.append(percent_num_env_success_over_time)

        
        avg_rewards_all_cat.append(avg_rewards_all_plots)
        avg_num_reaches_all_cat.append(avg_num_reaches_all_plots)
        percent_num_env_success_all_cat.append(percent_num_env_success_all_plots)
   
        print(f"- {tasks_list[k]}: {percent_num_env_success_all_plots}")
        print(f"- BC vs our: {percent_num_env_success_all_plots[-1] - percent_num_env_success_all_plots[1]}")

    avg_rewards_all_cat = np.array(avg_rewards_all_cat).transpose((1,0))
    avg_num_reaches_all_cat = np.array(avg_num_reaches_all_cat).transpose((1,0))
    percent_num_env_success_all_cat = np.array(percent_num_env_success_all_cat).transpose((1,0))

    # top = cm.get_cmap('Oranges_r', 128) # r means reversed version
    # bottom = cm.get_cmap('Blues', 128)# combine it all
    # newcolors = np.vstack((top(np.linspace(0, 1, 128)),
    #                     bottom(np.linspace(0, 1, 128))))# create a new colormaps with a name of OrangeBlue
    # colors = ListedColormap(newcolors, name='OrangeBlue')

        
    

    
    # plot_bar_chart_2col(categories=tasks_list, bar_labels_per_cat=labels, bar_values_per_cat=avg_rewards_all_cat, x_label="Task", y_label="reward", title="average ground truth rewards comparison", path=plot_path)
    # plot_bar_chart_2col(categories=tasks_list, bar_labels_per_cat=labels, bar_values_per_cat=avg_num_reaches_all_cat, x_label="Task", y_label="number of contacts", title="average number of contacts comparison", path=plot_path)
    # plot_bar_chart_2col(categories=tasks_list, bar_labels_per_cat=labels, bar_values_per_cat=percent_num_env_success_all_cat, x_label="Task", y_label="task success rate", title="task success rate comparison", path=plot_path)

    plot_bar_chart_multiple(categories=tasks_list, bar_labels_per_cat=labels, bar_values_per_cat=avg_rewards_all_cat, x_label="Task", y_label="reward", title="average ground truth rewards comparison", path=plot_path)
    plot_bar_chart_multiple(categories=tasks_list, bar_labels_per_cat=labels, bar_values_per_cat=avg_num_reaches_all_cat, x_label="Task", y_label="number of contacts", title="average number of contacts comparison", path=plot_path)
    plot_bar_chart_multiple(categories=tasks_list, bar_labels_per_cat=labels, bar_values_per_cat=percent_num_env_success_all_cat, x_label="Task", y_label="task success rate", title="task success rate comparison", path=plot_path)


    # xs = [16*i for i in range(len(labels))]
    # plot_bar_chart(xs, avg_rewards_all_plots, title="average ground truth rewards comparison", path=plot_path, x_label="methods", y_label="reward", tick_labels=labels)
    # plot_bar_chart(xs, avg_num_reaches_all_plots, title="average number of contacts comparison", path=plot_path, x_label="methods", y_label="number of contacts", tick_labels=labels)
    # plot_bar_chart(xs, percent_num_env_success_all_plots, title=f"task success rate comparison", path=plot_path, x_label="methods", y_label="task success rate", tick_labels=labels)





