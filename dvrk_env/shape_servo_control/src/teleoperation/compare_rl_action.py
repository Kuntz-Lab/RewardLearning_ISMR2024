
import os
import numpy as np
import pickle
import timeit
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

import os

'''
compare data of different rl runs
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
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    os.makedirs(path, exist_ok=True)
    fig.savefig(f"{path}/{title}.png")
    plt.cla()
    plt.close(fig) 


if __name__ == "__main__":
    ### CHANGE ####
    max_num_balls = 2
    plot_path = "/home/dvrk/RL_data/traceCurveActionCompare"
    os.makedirs(plot_path, exist_ok=True)

    #####################################################################################################################################
    ##########################################  learning Curve  ###############################################################
    #####################################################################################################################################
    root_paths = ["/home/dvrk/RL_data/traceCurve/learned_reward_eef_pos_2ball",  "/home/dvrk/RL_data/traceCurve/learned_reward_dof_vel"] #"/home/dvrk/RL_data/traceCurve/learned_reward_eef_vel", 
    labels = ["eef pos","joint vel"] # "eef vel", 
    data_recording_paths = [f"{root_paths[i]}/data" for i in range(len(root_paths))]

    max_episodes = 601
    last_episode_idx = max_episodes - 1
    step = 10
    print(f"num episodes: {max_episodes}")
   
    ######### plot rewards and success rate #############
    num_plots = len(root_paths)
    avg_rewards_all_plots = []
    avg_num_reaches_all_plots = []
    percent_num_env_success_all_plots = []
    for i in range(num_plots):
        avg_rewards = []
        avg_num_reaches = []
        percent_num_env_success = []
        for episode_idx in range(0, max_episodes, step):
            
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

        avg_rewards_all_plots.append(avg_rewards)
        avg_num_reaches_all_plots.append(avg_num_reaches)
        percent_num_env_success_all_plots.append(percent_num_env_success)

    episodes = [i for i in range(0, max_episodes, step)]

    plot_curves(xs=episodes, ys_list=avg_rewards_all_plots, title="average ground truth rewards 2", path=plot_path, x_label="episodes", y_label="reward", curve_labels=labels)
    plot_curves(xs=episodes, ys_list=avg_num_reaches_all_plots, title="average number of contact 2", path=plot_path, x_label="episodes", y_label="number of contacts", curve_labels=labels)
    plot_curves(xs=episodes, ys_list=percent_num_env_success_all_plots, title=f"task success rate 2", path=plot_path, x_label="episodes", y_label="task success rate", curve_labels=labels) 

