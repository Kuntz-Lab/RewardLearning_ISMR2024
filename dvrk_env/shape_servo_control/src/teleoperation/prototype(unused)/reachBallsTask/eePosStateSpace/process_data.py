#!/usr/bin/env python3
import os
import numpy as np
import pickle
import timeit
from random import sample
np.random.seed(2020)

num_data_pt = 10000
num_demos = 200
data_recording_path = "/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_2/demos"
data_processed_path = "/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_2/processed_data_5"

def get_random_trajectory():
    idx = np.random.randint(low=0, high=num_demos)
    file = os.path.join(data_recording_path, "sample " + str(idx) + ".pickle")
    with open(file, 'rb') as handle:
        trajectory = pickle.load(handle)

    return idx, trajectory

def get_states_and_gt_reward(trajectory):
    """Return all states and ground truth rewards of the trajectory"""
    states = []
    gt_reward = 0
    # print(len(trajectory))
    for eef_state in trajectory:
        pos = np.array(list(eef_state["pose"]["p"]))
        # print("pos:", pos)
        states.append(pos)
        # gt_reward += np.linalg.norm(pos - np.array([0.0, -0.44, 0.16+0.015]))
        gt_reward += np.linalg.norm(pos[:2] - np.array([0.0, -0.44])) - 0.5*abs(pos[2]-(0.16+0.015))


    return states, gt_reward*1000

# def sample_trajectory(shorter_traj, longer_traj):
#     new_longer_traj = []
#     new_longer_traj.append(shorter_traj[0])
#     new_longer_traj.extend(sample(longer_traj[1:-1], len(shorter_traj)-2))
#     new_longer_traj.append(shorter_traj[-1])

#     assert len(shorter_traj) == len(new_longer_traj)
    
#     # print("FIX!")
#     return new_longer_traj

# def sample_trajectory(shorter_traj, longer_traj):
#     new_longer_traj = []
#     new_longer_traj.append(longer_traj[0])
    
#     sampled_idxs = sorted(sample(list(range(1,len(longer_traj)-1)), k=len(shorter_traj)-2))
#     sampled_traj = [longer_traj[i] for i in sampled_idxs]
#     new_longer_traj.extend(sampled_traj)
#     new_longer_traj.append(longer_traj[-1])

    
#     assert len(shorter_traj) == len(new_longer_traj)
    
#     # print("FIX!")
#     return new_longer_traj


start_time = timeit.default_timer() 

for i in range(num_data_pt):

    idx_1 = 0
    idx_2 = 0
    while (idx_1 == idx_2):
        idx_1, trajectory_1 = get_random_trajectory()  
        idx_2, trajectory_2 = get_random_trajectory()

    states_1, gt_reward_1 = get_states_and_gt_reward(trajectory_1)
    states_2, gt_reward_2 = get_states_and_gt_reward(trajectory_2)
    # if len(states_1) > len(states_2):
    #     states_1 = sample_trajectory(states_2, states_1)
    # elif len(states_1) < len(states_2):
    #     states_2 = sample_trajectory(states_1, states_2)

    # output_label = int(gt_reward_1 > gt_reward_2)
    output_label = int(gt_reward_1 < gt_reward_2)

    states_1 = np.array(states_1)
    states_2 = np.array(states_2)
    # print("states_1.shape, states_2.shape:", states_1.shape, states_2.shape)

    processed_data = {"traj_1": states_1, "traj_2": states_2, "indices":[idx_1, idx_2], \
                    "gt_reward_1": gt_reward_1, "gt_reward_2": gt_reward_2,
                    "label": output_label}
                    
    
    with open(os.path.join(data_processed_path, "processed sample " + str(i) + ".pickle"), 'wb') as handle:
        pickle.dump(processed_data, handle, protocol=pickle.HIGHEST_PROTOCOL)     


    if i % 50 == 0:
        print("========================================")
        print("current count:", i, " , time passed:", timeit.default_timer() - start_time)
        # print("label", output_label)
        

    
