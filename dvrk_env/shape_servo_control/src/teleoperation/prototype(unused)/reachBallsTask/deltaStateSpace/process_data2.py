#!/usr/bin/env python3
import os
import numpy as np
import pickle
import timeit
from random import sample
import open3d
from utils import *
np.random.seed(2020)



def get_random_trajectory(group, num_samples_per_group):
    idx = np.random.randint(low=0, high=num_samples_per_group)
    file = os.path.join(data_recording_path, f"group {group} sample {idx}.pickle")
    with open(file, 'rb') as handle:
        data = pickle.load(handle)

    return idx, data

def get_states(data):
    """Return all states of the trajectory"""
    states = []
    mid_pose = np.array(data["mid pose"])
    goal_pose = np.array(data["goal pose"])

    for i, eef_state in enumerate(data["traj"]):
        eef_pose = np.array(list(eef_state["pose"]["p"]))
        if data["cp was reached"][i]:
            state = np.concatenate((eef_pose*0-mid_pose*0, eef_pose-goal_pose), axis=None)
        else:
            state = np.concatenate((eef_pose-mid_pose, eef_pose-goal_pose), axis=None)
        states.append(state)

    # states: shape (traj_length, 6)
    
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



def compare_two_trajectories(group, perfect_trajectories, data_1, data_2):
    """Return True if traj 1 > traj 2 else False"""

    datas = [data_1, data_2]
    success_counts = []
    states = []
    all_pos_states = []

    
    for data in datas:
        success_counts.append(int(data["success goal"]) + int(data["success mid"]))
        states.append(get_states(data))
        all_pos_states.append(get_all_pos_states(data))

    if success_counts[0] > success_counts[1]:
        label = False
    elif success_counts[0] < success_counts[1]:
        label = True
    else:
        perfect_traj = perfect_trajectories[group] #compute_perfect_trajectory(group)
        deviations = []
        for i, data in enumerate(datas):
            deviations.append(compute_deviation(open3d_ize(all_pos_states[i]), perfect_traj)) 
        
        if deviations[0] > deviations[1]:
            label = True
        elif deviations[0] < deviations[1]:
            label = False   
    # print("deviations:", deviations)

    return states, label


def compare_two_trajectories_w_reward(group, perfect_trajectories, data_1, data_2):
    """Return True if traj 1 > traj 2 else False"""

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

# # make sure all zeros after some point
# data_recording_path = "/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_3/demos_train" #Change
# data_processed_path = "/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_3/processed_data_train" #Change
# idx_1, data_1 = get_random_trajectory(0, 10)  
# states = get_states(data_1)
# print(states[:, :3])

if __name__ == "__main__":
    ### Train
    # data_recording_path = "/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_3/demos_train"
    # data_processed_path = "/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_3/processed_data_train"

    ### Test
    data_recording_path = "/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_3/demos_test" #Change
    data_processed_path = "/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_3/processed_data_test" #Change


    os.makedirs(data_processed_path, exist_ok=True)

    start_time = timeit.default_timer() 
    num_data_pt = 10000 #Change
    NUM_GROUP = 20#10 #Change
    get_gt_rewards = True #Change
    perfect_trajectories = []

    for group_idx in range(NUM_GROUP):
        perfect_trajectories.append(compute_perfect_trajectory(group_idx, data_recording_path))

    for i in range(num_data_pt):

        if i % 50 == 0:
            print("========================================")
            print("current count:", i, " , time passed:", timeit.default_timer() - start_time)

        group_idx = np.random.randint(low=0, high=NUM_GROUP)

        idx_1 = 0
        idx_2 = 0
        while (idx_1 == idx_2):
            idx_1, data_1 = get_random_trajectory(group=group_idx, num_samples_per_group=10)  
            idx_2, data_2 = get_random_trajectory(group=group_idx, num_samples_per_group=10)

        if get_gt_rewards:
            states, label, gt_rewards = compare_two_trajectories_w_reward(group_idx, perfect_trajectories, data_1, data_2)
            processed_data = {"traj_1": states[0], "traj_2": states[1], "indices":[idx_1, idx_2], \
                            "gt_reward_1": gt_rewards[0], "gt_reward_2": gt_rewards[1], \
                            "label": label}
        else:
            states, label = compare_two_trajectories(group_idx, perfect_trajectories, data_1, data_2)    

            processed_data = {"traj_1": states[0], "traj_2": states[1], "indices":[idx_1, idx_2], \
                            # "mid pose": np.array(data_1["mid pose"]), "goal pose": np.array(data_1["goal pose"]), \
                            # "target obj poses": np.concatenate((data_1["mid pose"], data_1["goal pose"]), axis=None), \
                            "label": label}
                        
        
        with open(os.path.join(data_processed_path, "processed sample " + str(i) + ".pickle"), 'wb') as handle:
            pickle.dump(processed_data, handle, protocol=3)     




            

        
