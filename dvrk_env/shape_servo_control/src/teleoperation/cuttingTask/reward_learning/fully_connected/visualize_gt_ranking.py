import pickle
import os
import numpy as np
import timeit

from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym import gymutil

from reward import RewardNetPointCloudEEF as RewardNet
import matplotlib.pyplot as plt
import torch
import argparse



def get_random_data(data_recording_path, num_processed_samples):
    '''
    '''
    idx = np.random.randint(low=0, high= num_processed_samples)
    file = os.path.join(data_recording_path, f"processed sample {idx}.pickle")
    with open(file, 'rb') as handle:
        data = pickle.load(handle)
    
    return idx, data

def get_data(data_recording_path, idx):
    file = os.path.join(data_recording_path, f"processed sample {idx}.pickle")
    with open(file, 'rb') as handle:
        data = pickle.load(handle)
    
    return idx, data

def success_stat(data_path, max_num_balls=2):
    '''
    for every i, how many demonstrations have reached i balls
    '''
    files = os.listdir(data_path)
    stat = {i:0 for i in range(max_num_balls+1)}
    for i, file in enumerate(files):
        if i%1000==0:
            print("now at sample ", i)
        with open(f"{data_path}/{file}", 'rb') as handle:
            data = pickle.load(handle)
        count = data["num_balls_reached"]
        stat[count] = stat[count]+1
    print("how many samples have i balls reached for each i: ", stat)

def num_reach_each_ball(data_path, max_num_balls=2):
    '''
    for every i, among all demonstrations that reach i ball(s), how many times each ball is reached respectively
    '''
    files = os.listdir(data_path)
    stat = {i:[0 for j in range(max_num_balls)] for i in range(max_num_balls+1)}
    for i, file in enumerate(files):
        if i%1000==0:
            print("now at sample ", i)
        with open(f"{data_path}/{file}", 'rb') as handle:
            data = pickle.load(handle)
        num_reach_each_ball = data["which_balls_reached"]
        num_balls_reached = data["num_balls_reached"]
        for i, num in enumerate(num_reach_each_ball):
            stat[num_balls_reached][i] = stat[num_balls_reached][i]+num
    print("how many contact for each ball categorized by num balls reached: ", stat)


def show_gt_preference_3D(data, only_failure=True):
    label = data["label"]
    rewards_no_bonus_1 = np.array(data["rewards_no_bonus_1"])
    rewards_no_bonus_2 = np.array(data["rewards_no_bonus_2"])
    gt_reward_1 = data["gt_reward_1"]
    gt_reward_2 = data["gt_reward_2"]

    print("==============")
    print(f"red reward: {gt_reward_1}")
    print(f"green reward: {gt_reward_2}")
    print(f"num balls reached red: ", data["num_balls_reached_1"])
    print(f"num balls reached green: ", data["num_balls_reached_2"])
    print(f"group: ", data["group"])
    print(f"traj_1_idx: ", data["indices"][0])
    print(f"traj_2_idx: ", data["indices"][1])
    print("==============")

    if only_failure:
        wrong_1 = (data["num_balls_reached_1"] > data["num_balls_reached_2"]) and (gt_reward_1<=gt_reward_2)
        wrong_2 = (data["num_balls_reached_1"] < data["num_balls_reached_2"]) and (gt_reward_1>=gt_reward_2)
        if not (wrong_1 or wrong_2):
            return 0

    which = ["red better", "green better"]
    traj = [data["eef_traj_1"].cpu().detach().numpy(), data["eef_traj_2"].cpu().detach().numpy()]

    max_reward_1 = np.max(rewards_no_bonus_1)
    max_reward_2 = np.max(rewards_no_bonus_2)
    max_reward = max(max_reward_1, max_reward_2)
    min_reward_1 = np.min(rewards_no_bonus_1)
    min_reward_2 = np.min(rewards_no_bonus_2)
    min_reward = min(min_reward_1, min_reward_2)

    balls_xyz = data["balls_xyz"]

    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")


    for ball_pose in balls_xyz:
        print("ball: ", balls_xyz)
        ax.plot(ball_pose[0], ball_pose[1], ball_pose[2], "o", markersize=35)

    print("~~traj red: ", traj[0])
    print("~~traj green: ", traj[1])
    xs = [traj[0][t,0] for t in range(len(traj[0]))]
    ys = [traj[0][t,1] for t in range(len(traj[0]))]
    zs = [traj[0][t,2] for t in range(len(traj[0]))]
    heats = [[(rewards_no_bonus_1[t].item() - min_reward) / (max_reward - min_reward), 0, 0] for t in range(len(traj[0]))]
    ax.scatter(xs, ys, zs, c=heats, s=[30+2*i for i in range(len(traj[0]))]) 
    ax.plot(xs, ys, zs, color='red', label=f"num balls reached: "+ str(data["num_balls_reached_1"])) 

    xs = [traj[1][t,0] for t in range(len(traj[1]))]
    ys = [traj[1][t,1] for t in range(len(traj[1]))]
    zs = [traj[1][t,2] for t in range(len(traj[1]))]
    heats = [[0, (rewards_no_bonus_2[t].item() - min_reward) / (max_reward - min_reward), 0] for t in range(len(traj[1]))]
    ax.scatter(xs, ys, zs, c=heats, s=[30+2*i for i in range(len(traj[1]))]) 
    ax.plot(xs, ys, zs, color='green', label=f"num balls reached: "+ str(data["num_balls_reached_2"])) 

    ax.legend()

    plt.title(f"GT: {which[int(label)]}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    # ax.set_xticks([-0.1+0.02*i for i in range(11)])
    # ax.set_yticks([-0.6+0.02*i for i in range(11)])
    # ax.set_zticks([0+0.02*i for i in range(11)])
    plt.show()

    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--demo_path', type=str, help="path to demos")
    parser.add_argument('--data_processed_path', type=str, help="path to processed data")
    parser.add_argument('--num_data_pt', default=14000, type=int, help="num data points in the processed data to loop through")
    parser.add_argument('--max_num_balls', default=1, type=int, help="max num ball in the demo")


    args = parser.parse_args()
    num_data_pt = args.num_data_pt
    data_recording_path = args.demo_path
    data_processed_path = args.data_processed_path
    max_num_balls = args.max_num_balls

    start_time = timeit.default_timer()

    success_stat(data_recording_path, max_num_balls=max_num_balls) 
    num_reach_each_ball(data_recording_path, max_num_balls=max_num_balls)

    num_fail = 0
    for i in range(num_data_pt):

        if i % 1 == 0:
            print("========================================")
            print("current count:", i, " , time passed:", timeit.default_timer() - start_time)
        
        #idx, data = get_random_data(data_recording_path, num_data_pt)  
        _, data = get_data(data_processed_path, i)

        ## for only showing the group
        # group = data["group"]
        # if (group!=0):
        #     continue

        num_fail += show_gt_preference_3D(data, only_failure=False)

    print(f"################### num_wrong_ranking: {num_fail}")

        



    