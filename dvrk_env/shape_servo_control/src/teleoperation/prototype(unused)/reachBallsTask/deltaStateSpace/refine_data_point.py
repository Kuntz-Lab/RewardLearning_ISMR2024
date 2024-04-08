#!/usr/bin/env python3
import os
import numpy as np
import pickle
import timeit
from random import sample
from utils import *
from process_data2 import get_states, get_all_pos_states
np.random.seed(2020)

num_data_pt = 10000
num_demos = 200
data_recording_path = "/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_3/demos_test"
data_processed_path = "/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_3/demos_test_refined"
os.makedirs(data_processed_path, exist_ok=True)



def get_states_and_gt_reward(group, perfect_trajectories, data):
    """Return all states of the trajectory with ground truth reward"""
    states = get_states(data)
    all_pose_states = get_all_pos_states(data)
    perfect_traj = perfect_trajectories[group]

    success_count = int(data["success goal"]) + int(data["success mid"])
    deviation = compute_deviation(open3d_ize(all_pose_states), perfect_traj)
    gt_reward = success_count*1 - deviation

    return states, gt_reward


start_time = timeit.default_timer() 
num_samples_per_group = 10#20 #Change
NUM_GROUP = 14#20 #Change
perfect_trajectories = []

for group_idx in range(NUM_GROUP):
    perfect_trajectories.append(compute_perfect_trajectory(group_idx, data_recording_path))

# loop over groups and samples within each group
for group_idx in range(NUM_GROUP):    
    for sample_idx in range(num_samples_per_group): 
        file = os.path.join(data_recording_path, f"group {group_idx} sample {sample_idx}.pickle")
        with open(file, 'rb') as handle:
            data = pickle.load(handle)

        states, gt_reward = get_states_and_gt_reward(group_idx, perfect_trajectories, data)

        processed_data = {"traj": states, "gt_reward": gt_reward, \
                        "mid pose": data["mid pose"],\
                        "success goal": data["success goal"], "success mid": data["success mid"], \
                        "goal pose": data["goal pose"],\
                        "num cp": data["num cp"]}
                        

        with open(os.path.join(data_processed_path, f"group {group_idx} sample {sample_idx}.pickle"), 'wb') as handle:
            pickle.dump(processed_data, handle, protocol=3)     



    print("========================================")
    print("current group:", group_idx, " , time passed:", timeit.default_timer() - start_time)

        

    
