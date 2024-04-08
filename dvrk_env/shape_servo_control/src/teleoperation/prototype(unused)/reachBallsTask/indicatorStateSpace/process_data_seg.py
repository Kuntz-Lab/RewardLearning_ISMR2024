#!/usr/bin/env python3
import math
import argparse
import os
import numpy as np
import pickle
import timeit
from random import sample
import open3d
from utils import *
np.random.seed(2020)

'''
Process the data for training
'''

def get_random_trajectory(group, num_samples_per_group):
    '''
    get a trajectory randomly from a group.
    Returns the sample_idx, data and the segmentation label
    '''
    idx = np.random.randint(low=0, high=num_samples_per_group)
    file = os.path.join(data_recording_path, f"group {group} sample {idx}.pickle")
    with open(file, 'rb') as handle:
        data = pickle.load(handle)
    
    return idx, data, data["traj seg label"]

def get_states(data):
    """Return all states of the trajectory"""
    states = []
    obj_pose = np.concatenate((data["mid pose"], data["goal pose"]))
    goal_indicators = data["goal indicators"]

    for i, eef_state in enumerate(data["traj"]):
        eef_pose = np.array(list(eef_state["pose"]["p"]))
        goal_indicator = goal_indicators[i]
        states.append(np.concatenate((np.array([goal_indicator]), eef_pose, obj_pose), axis=None))

    # states: shape (traj_length, 10). First: goal_indicator; Next 3: eef_pose; Next 6: two object poses

    return np.array(states)

def get_all_pos_states(data):
    """Return all states of eef pose and obj poses of the trajectory"""
    states = []
    obj_pose = np.concatenate((data["mid pose"], data["goal pose"]))

    for eef_state in data["traj"]:
        eef_pose = np.array(list(eef_state["pose"]["p"]))
        states.append(np.concatenate((eef_pose, obj_pose), axis=None))

    # states: shape (traj_length, 9). First 3: eef_pose; Next 6: two object poses

    return np.array(states)


def compare_two_trajectories_w_reward(group, perfect_trajectories, data_1, data_2):
    """Return True if traj 1 > traj 2 else False assuming they are both from the same group"""

    datas = [data_1, data_2]
    success_counts = []
    states = []
    all_pos_states = []
    rewards  = []

    perfect_traj = perfect_trajectories[group]
    deviations = []
    for i, data in enumerate(datas):
        success_counts.append(int(data["success goal"]) + int(data["success mid"]))
        states.append(get_states(data))
        all_pos_states.append(get_all_pos_states(data))
        deviations.append(compute_deviation(open3d_ize(all_pos_states[i]), perfect_traj)) 
        rewards.append(success_counts[i]*5 - deviations[i])

    if success_counts[0] > success_counts[1]:
        label = False
    elif success_counts[0] < success_counts[1]:
        label = True
    else:        
        if deviations[0] > deviations[1]:
            label = True
        elif deviations[0] < deviations[1]:
            label = False   
    # print("deviations:", deviations)

    return states, label, rewards

def find_split_traj_idx(goal_indicators):
    '''
    returns the index where it is the first occurence of goal_indicator==1
    '''
    for i, indicator in enumerate(goal_indicators):
        if indicator==1:
            return i
    return None


def segment_data(data_1, data_2, goal_indicator):
    '''
    segment out the trajectories (obtain the part of each trajectory with the
    goal indicators == the specified goal_indicator)

    S is the start position
    2: S-----------
    4: S-----b0----
    6: S-----b0---b1
    3: S----------b1 (all goal indicators here are assumed to be 1 for training purpose)
    '''
    split_idx_1 = find_split_traj_idx(data_1["goal indicators"])
    split_idx_2 = find_split_traj_idx(data_2["goal indicators"])
    
    if goal_indicator == 0:
        data_1["traj"] = data_1["traj"][:split_idx_1]
        data_1["goal indicators"] = data_1["goal indicators"][:split_idx_1]
        data_1["success goal"] = False

        data_2["traj"] = data_2["traj"][:split_idx_2]
        data_2["goal indicators"] = data_2["goal indicators"][:split_idx_2]
        data_2["success goal"] = False    
    elif goal_indicator == 1:
        data_1["traj"] = data_1["traj"][split_idx_1:]
        data_1["goal indicators"] = data_1["goal indicators"][split_idx_1:]
        data_1["success mid"] = False

        data_2["traj"] = data_2["traj"][split_idx_2:]
        data_2["goal indicators"] = data_2["goal indicators"][split_idx_2:]
        data_2["success mid"] = False
    
    return data_1, data_2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Options')
    ### CHANGE ####
    is_train = True
    suffix = "train" if is_train else "test"
    parser.add_argument('--data_recording_path', default= f"/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_indicator_0and1/demos_{suffix}" , type=str, help="location of existing raw data")
    parser.add_argument('--num_samples_per_group', default= 20, type=int, help="maximum sample index to process within a group")
    parser.add_argument('--NUM_GROUP', default= 20, type=int, help="number of groups to process")
    ### CHANGE ####
    args = parser.parse_args()

    data_recording_path = args.data_recording_path
    num_samples_per_group = args.num_samples_per_group
    NUM_GROUP = args.NUM_GROUP

    start_time = timeit.default_timer() 
    num_data_pt = 10000 #Change
    get_gt_rewards = True #Change
    perfect_trajectories = []

    for group_idx in range(NUM_GROUP):
        perfect_trajectories.append(compute_perfect_trajectory(group_idx, data_recording_path))
        #perfect_trajectories.append(compute_perfect_first_half_trajectory(group_idx, data_recording_path))


#######################################
# Training data for reaching b0
#######################################
    # print("+++++++++++++ generate training data for reaching b0 +++++++++++++++++")
    # data_processed_path = f"/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_indicator_0and1/processed_data_{suffix}_0_sub1"
    # os.makedirs(data_processed_path, exist_ok=True)

    # for i in range(num_data_pt):

    #     if i % 50 == 0:
    #         print("========================================")
    #         print("current count:", i, " , time passed:", timeit.default_timer() - start_time)

    #     group_idx = np.random.randint(low=0, high=NUM_GROUP)

    #     idx_1 = 0
    #     idx_2 = 0
    #     while (idx_1 == idx_2) or (seg_label_1 == 3) or (seg_label_2 == 3):
    #         idx_1, data_1, seg_label_1 = get_random_trajectory(group_idx, num_samples_per_group)  
    #         idx_2, data_2, seg_label_2 = get_random_trajectory(group_idx, num_samples_per_group)
    #         #### for only saving data from path type 2 or 4
    #         # if (idx_1 != idx_2) and (((seg_label_1 == 4) and (seg_label_2 == 2)) or ((seg_label_1 == 2) and (seg_label_2 == 4)) or ((seg_label_1 == 2) and (seg_label_2 == 2)) and ((seg_label_1 == 4) and (seg_label_2 == 4))):
    #         #     break

    #     ################## segment here ########################
    #     data_1, data_2 = segment_data(data_1, data_2, goal_indicator=0)
    #     #### for testing quality of data for b0 model
    #     # data_1["goal indicators"] = [1 for i in data_1["goal indicators"]]
    #     # data_2["goal indicators"] = [1 for i in data_2["goal indicators"]]

    #     if get_gt_rewards:
    #         states, label, gt_rewards = compare_two_trajectories_w_reward(group_idx, perfect_trajectories, data_1, data_2)
    #         processed_data = {"traj_1": states[0], "traj_2": states[1], "indices":[idx_1, idx_2], \
    #                         "gt_reward_1": gt_rewards[0], "gt_reward_2": gt_rewards[1], \
    #                         "label": label}
    #     else:
    #         states, label, _ = compare_two_trajectories_w_reward(group_idx, perfect_trajectories, data_1, data_2)    

    #         processed_data = {"traj_1": states[0], "traj_2": states[1], "indices":[idx_1, idx_2], \
    #                         "label": label}
                        
        
    #     with open(os.path.join(data_processed_path, "processed sample " + str(i) + ".pickle"), 'wb') as handle:
    #         pickle.dump(processed_data, handle, protocol=3)

#######################################
# Training data for reaching b1
#######################################
    print("+++++++++++++ generate training data for reaching b1 +++++++++++++++++")
    data_processed_path = f"/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_indicator_0and1/processed_data_{suffix}_1_sub0"
    os.makedirs(data_processed_path, exist_ok=True)

    for i in range(num_data_pt):

        if i % 50 == 0:
            print("========================================")
            print("current count:", i, " , time passed:", timeit.default_timer() - start_time)

        group_idx = np.random.randint(low=0, high=NUM_GROUP)

        idx_1 = 0
        idx_2 = 0
        while (idx_1 == idx_2) or (seg_label_1 == 2) or (seg_label_2 == 2):
            idx_1, data_1, seg_label_1 = get_random_trajectory(group_idx, num_samples_per_group)  
            idx_2, data_2, seg_label_2 = get_random_trajectory(group_idx, num_samples_per_group)

        ################## segment here ########################
        if seg_label_1 == 3:
            data_1["goal indicators"] = [1 for i in data_1["goal indicators"]]
        if seg_label_2 == 3:
            data_2["goal indicators"] = [1 for i in data_2["goal indicators"]]
        data_1, data_2 = segment_data(data_1, data_2, goal_indicator=1)
        data_1["goal indicators"] = [0 for i in data_1["goal indicators"]]
        data_2["goal indicators"] = [0 for i in data_2["goal indicators"]]

        if get_gt_rewards:
            states, label, gt_rewards = compare_two_trajectories_w_reward(group_idx, perfect_trajectories, data_1, data_2)
            processed_data = {"traj_1": states[0], "traj_2": states[1], "indices":[idx_1, idx_2], \
                            "gt_reward_1": gt_rewards[0], "gt_reward_2": gt_rewards[1], \
                            "label": label}
        else:
            states, label, _ = compare_two_trajectories_w_reward(group_idx, perfect_trajectories, data_1, data_2)    

            processed_data = {"traj_1": states[0], "traj_2": states[1], "indices":[idx_1, idx_2], \
                            "label": label}
                        
        
        with open(os.path.join(data_processed_path, "processed sample " + str(i) + ".pickle"), 'wb') as handle:
            pickle.dump(processed_data, handle, protocol=3)     

     




            

        
