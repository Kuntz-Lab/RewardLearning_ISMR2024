#!/usr/bin/env python3
import os
import numpy as np
import pickle
import timeit
from random import sample
np.random.seed(2020)

num_data_pt = 10000
num_demos = 200
data_recording_path = "/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_2/test_demos"
data_processed_path = "/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_2/test_refined_demos"



def get_states_and_gt_reward(trajectory):
    """Return all states and ground truth rewards of the trajectory"""
    states = []
    gt_reward = 0
    # print(len(trajectory))
    for eef_state in trajectory:
        pos = np.array(list(eef_state["pose"]["p"]))
        # print("pos:", pos)
        states.append(pos)
        gt_reward += np.linalg.norm(pos - np.array([0.0, -0.44, 0.16+0.015]))
        # gt_reward += np.linalg.norm(pos[:2] - np.array([0.0, -0.44])) - 0.5*abs(pos[2]-(0.16+0.015))


    return states, gt_reward*1000


def sample_trajectory(traj, new_len=7):
    new_traj = []
    new_traj.append(traj[0])
    
    sampled_idxs = sorted(sample(list(range(1,len(traj)-1)), k=new_len-2))
    sampled_traj = [traj[i] for i in sampled_idxs]
    new_traj.extend(sampled_traj)
    new_traj.append(traj[-1])

    # print(len(new_traj), new_len, len(sampled_idxs))
    assert len(new_traj) == new_len
    
    # print("FIX!")
    return new_traj


start_time = timeit.default_timer() 

# for i in range(num_demos):
# for i in range(21,31):
for i in list(range(0,10))+list(range(21,31)):    
    file = os.path.join(data_recording_path, "sample " + str(i) + ".pickle")
    with open(file, 'rb') as handle:
        trajectory = pickle.load(handle)

    states, gt_reward = get_states_and_gt_reward(trajectory)


    # processed_data = {"traj": sample_trajectory(states), "gt_reward": gt_reward}
    processed_data = {"traj": states, "gt_reward": gt_reward}
                    

    with open(os.path.join(data_processed_path, "sample " + str(i) + ".pickle"), 'wb') as handle:
        pickle.dump(processed_data, handle, protocol=pickle.HIGHEST_PROTOCOL)     


    if i % 50 == 0:
        print("========================================")
        print("current count:", i, " , time passed:", timeit.default_timer() - start_time)
        # print("label", output_label)
        

    
