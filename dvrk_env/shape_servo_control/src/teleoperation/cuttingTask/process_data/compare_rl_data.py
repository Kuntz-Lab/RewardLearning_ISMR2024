
import os
import numpy as np
import pickle
import timeit
import matplotlib.pyplot as plt
from PIL import Image

import os


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

def plot_curve(xs, ys, title, path, x_label, y_label):
    fig, ax = plt.subplots()
    ax.plot(xs, ys)
    #ax.legend()
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    os.makedirs(path, exist_ok=True)
    fig.savefig(f"{path}/{title}.png")
    plt.cla()
    plt.close(fig)    

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

# def plot_bar_chart(xs, ys, title, path, x_label, y_label, tick_labels):
#     fig, ax = plt.subplots()
#     ax.bar(xs, ys, width=1.5 ,align='center', tick_label=tick_labels)
#     #ax.legend()
#     ax.set_xticklabels(tick_labels, fontsize=8)
#     ax.set_title(title)
#     ax.set_xlabel(x_label)
#     ax.set_ylabel(y_label)
#     os.makedirs(f"{path}", exist_ok=True)
#     fig.savefig(f"{path}/{title}.png")
#     plt.cla()
#     plt.close(fig)

def plot_bar_chart(xs, ys, title, path, x_label, y_label, tick_labels, colors=['red', 'green', 'brown', 'orange', 'blue']):
    fig, ax = plt.subplots()
    bars = ax.bar(xs, ys, width=8 ,align='center', tick_label=tick_labels)
    for i, _ in enumerate(bars):
        bars[i].set_color(colors[i])
    r = patches.Rectangle((xs[0]-6, 0), 12, max(ys),
                      lw=3, ls='--',
                      edgecolor='gray', facecolor='none',
                      clip_on=False
                     )
    ax.add_patch(r)
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    os.makedirs(f"{path}", exist_ok=True)
    fig.savefig(f"{path}/{title}.png")
    plt.cla()
    plt.close(fig)

if __name__ == "__main__":
    ### CHANGE ####
    max_num_balls = 1
    plot_path = "/home/dvrk/RL_data/cutCompare"
    os.makedirs(plot_path, exist_ok=True)

    #####################################################################################################################################
    ##########################################  learning Curve  ###############################################################
    #####################################################################################################################################
    # root_paths = ["/home/dvrk/RL_data/traceCurve/learned_reward_eef_pos_2",  "/home/dvrk/RL_data/traceCurveXYZ/gt_reward_eef_pos_2ball_old2"]
    # #root_paths = ["/home/dvrk/RL_data/pointReacher/learned_reward_eef_pos_rand",  "/home/dvrk/RL_data//pointReacher/gt_reward_eef_pos_rand_pc_obs"]
    # labels = ["policy using learned reward", "policy using ground truth reward"]
    # data_recording_paths = [f"{root_paths[i]}/data" for i in range(len(root_paths))]

    # max_episodes = 241
    # last_episode_idx = max_episodes - 1
    # step = 10
    # print(f"num episodes: {max_episodes}")
   
    # ######### plot rewards and success rate #############
    # num_plots = len(root_paths)
    # avg_rewards_all_plots = []
    # avg_num_reaches_all_plots = []
    # percent_num_env_success_all_plots = []
    # for i in range(num_plots):
    #     avg_rewards = []
    #     avg_num_reaches = []
    #     percent_num_env_success = []
    #     for episode_idx in range(0, max_episodes, step):
            
    #         data = get_episode_data(data_recording_paths[i], episode_idx=episode_idx)
            
    #         rewards = data["gt_cum_reward"]
    #         avg_reward = np.sum(rewards)/len(rewards)
    #         avg_rewards.append(avg_reward)
        
    #         num_balls_reached = data["num_balls_reached"]
    #         avg_num_reach = np.sum(num_balls_reached)/len(num_balls_reached)
    #         avg_num_reaches.append(avg_num_reach)
            
    #         num_env_success = np.count_nonzero(num_balls_reached==max_num_balls)
    #         num_env_success = num_env_success / len(num_balls_reached)
    #         percent_num_env_success.append(num_env_success)

    #     avg_rewards_all_plots.append(avg_rewards)
    #     avg_num_reaches_all_plots.append(avg_num_reaches)
    #     percent_num_env_success_all_plots.append(percent_num_env_success)

    # episodes = [i for i in range(0, max_episodes, step)]

    # plot_curves(xs=episodes, ys_list=avg_rewards_all_plots, title="average ground truth rewards", path=plot_path, x_label="episodes", y_label="reward", curve_labels=labels)
    # plot_curves(xs=episodes, ys_list=avg_num_reaches_all_plots, title="average number of reaches", path=plot_path, x_label="episodes", y_label="number of reaches", curve_labels=labels)
    # plot_curves(xs=episodes, ys_list=percent_num_env_success_all_plots, title=f"fraction of successful environments (task success rate)", path=plot_path, x_label="episodes", y_label="fraction of environments", curve_labels=labels) 

    #####################################################################################################################################
    ##########################################  bar chart ###############################################################
    #####################################################################################################################################

    root_paths = ["/home/dvrk/RL_data/cut/gt_reward_eef_pos", "/home/dvrk/RL_data/cut/BC_perfect", "/home/dvrk/RL_data/cut/BC_total", "/home/dvrk/RL_data/cut/BC_pair", "/home/dvrk/RL_data/cut/learned_reward_eef_pos"]
    labels = ["GT reward", "BC perfect", "BC total", "BC pair", "our method"]
    data_recording_paths = [f"{root_paths[i]}/data" for i in range(len(root_paths))]
    max_episodes = [491, 111, 91, 291, 801] # last episode idx + 1 for each plot
    min_episodes = [400, 10, 10, 10, 400] # the minimum episode idx for each plot
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

    xs = [16*i for i in range(len(labels))]
    plot_bar_chart(xs, avg_rewards_all_plots, title="average ground truth rewards comparison", path=plot_path, x_label="methods", y_label="reward", tick_labels=labels)
    plot_bar_chart(xs, avg_num_reaches_all_plots, title="average number of contacts comparison", path=plot_path, x_label="methods", y_label="number of contacts", tick_labels=labels)
    plot_bar_chart(xs, percent_num_env_success_all_plots, title=f"task success rate comparison", path=plot_path, x_label="methods", y_label="task success rate", tick_labels=labels)





