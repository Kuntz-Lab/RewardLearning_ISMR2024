#!/usr/bin/env python3
import argparse
import os
import numpy as np
import pickle
import timeit
from random import sample
from utils import *
from process_data2 import get_states, get_all_pos_states
np.random.seed(2020)

'''
Refine the data for testing
'''

def get_states_and_gt_reward(group, perfect_trajectories, data):
    """Return all states of the trajectory with ground truth reward"""
    states = get_states(data)
    all_pose_states = get_all_pos_states(data)
    perfect_traj = perfect_trajectories[group]

    success_count = int(data["success goal"]) + int(data["success mid"])
    deviation = compute_deviation(open3d_ize(all_pose_states), perfect_traj)
    gt_reward = success_count*5 - deviation

    return states, gt_reward

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Options')
    ### CHANGE ####
    is_train = True
    suffix = "train" if is_train else "test"
    parser.add_argument('--data_recording_path', default= f"/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_indicator_1/demos_{suffix}" , type=str, help="location of existing data")
    parser.add_argument('--data_refined_path', default= f"/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_indicator_1/demos_{suffix}_refined" , type=str, help="location to save the refined data")
    parser.add_argument('--num_samples_per_group', default= 10, type=int, help="number of samples to refine within a group")
    parser.add_argument('--NUM_GROUP', default= 10, type=int, help="number of groups to refine")
    ### CHANGE ####
    args = parser.parse_args()

    data_recording_path = args.data_recording_path
    data_refined_path = args.data_refined_path
    num_samples_per_group = args.num_samples_per_group
    NUM_GROUP = args.NUM_GROUP

    os.makedirs(data_refined_path, exist_ok=True)

    start_time = timeit.default_timer() 
    perfect_trajectories = []

    for group_idx in range(NUM_GROUP):
        perfect_trajectories.append(compute_perfect_trajectory(group_idx, data_recording_path))

    # count2 = 0
    # count4 = 0
    # count6 = 0
    # count3 = 0
    # loop over groups and samples within each group
    for group_idx in range(NUM_GROUP):    
        for sample_idx in range(num_samples_per_group): 
            file = os.path.join(data_recording_path, f"group {group_idx} sample {sample_idx}.pickle")
            with open(file, 'rb') as handle:
                data = pickle.load(handle)

            # if data["traj seg label"]==3:
            #     count3+=1
            # if data["traj seg label"]==2:
            #     count2+=1
            # if data["traj seg label"]==4:
            #     count4+=1
            # if data["traj seg label"]==6:
            #     count6+=1
            states, gt_reward = get_states_and_gt_reward(group_idx, perfect_trajectories, data)

            refined_data = {"traj": states, "gt_reward": gt_reward, \
                            "mid pose": data["mid pose"],\
                            "success goal": data["success goal"], "success mid": data["success mid"], \
                            "goal pose": data["goal pose"],\
                            "num cp": data["num cp"]}

            with open(os.path.join(data_refined_path, f"group {group_idx} sample {sample_idx}.pickle"), 'wb') as handle:
                pickle.dump(refined_data, handle, protocol=3)     



        print("========================================")
        print("current group:", group_idx, " , time passed:", timeit.default_timer() - start_time)

    # print("2:", count2)
    # print("4:", count4)
    # print("6:", count6)
    # print("3:", count3)  
    # print("total:", NUM_GROUP*num_samples_per_group)   
              


        
