import sys

import torch
import math
import argparse
import os
import numpy as np
import pickle
import timeit
import matplotlib.pyplot as plt


def get_trajectory(data_recording_path, group, sample_idx):
    '''
    get a trajectory from a group and a specific sample index.
    Returns data
    '''
    file = os.path.join(data_recording_path, f"group {group} sample {sample_idx}.pickle")
    with open(file, 'rb') as handle:
        data = pickle.load(handle)
    
    return data

def get_processed_sample(data_recording_path, sample_idx):
    '''
    get a processed sample from training/testing data.
    Returns data
    '''
    file = os.path.join(data_recording_path, f"processed sample {sample_idx}.pickle")
    with open(file, 'rb') as handle:
        data = pickle.load(handle)
    
    return data

def show_3D_traj(data):
    traj = data["eef_traj"].cpu().detach().numpy()

    balls_xyz = data["balls_xyz"]

    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")


    for ball_pose in balls_xyz:
        print("ball: ", balls_xyz)
        ax.plot(ball_pose[0], ball_pose[1], ball_pose[2], "o", markersize=35)

    xs = [traj[t,0] for t in range(len(traj))]
    ys = [traj[t,1] for t in range(len(traj))]
    zs = [traj[t,2] for t in range(len(traj))]
    ax.scatter(xs, ys, zs, s=[30+2*i for i in range(len(traj))]) 
    ax.plot(xs, ys, zs, color='red', label=f"num balls reached: "+ str(data["num_balls_reached"])) 

    ax.legend()

    plt.title(f"BC traj")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xticks([-0.1+0.02*i for i in range(11)])
    ax.set_yticks([-0.6+0.02*i for i in range(11)])
    ax.set_zticks([0+0.02*i for i in range(11)])
    plt.show()

    return 1



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--BC_demo_path', type=str, help="path to recorded demo")
    parser.add_argument('--BC_data_path', type=str, help="path to recorded BC state-action pairs")
   
    args = parser.parse_args()

   
    BC_data_path = args.BC_data_path
    demo_path = args.BC_demo_path

    start_time = timeit.default_timer() 

    device = torch.device("cuda")
    
    print("---- process trajectories into total ranking")
    num_data = len(os.listdir(BC_data_path))
    visited_traj = set()
    for i in range(num_data):
        if i%1000 ==0:
            print("current sample: ", i)
        data = get_processed_sample(BC_data_path, sample_idx=i)
        group = data["group"]
        sample_idx = data["sample_idx"]
        traj = get_trajectory(demo_path, group, sample_idx)
        if (group, sample_idx) not in visited_traj:
            visited_traj.add((group, sample_idx))
            show_3D_traj(traj)
        


    