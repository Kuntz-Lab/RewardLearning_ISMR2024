import sys

import torch
import math
import argparse
import os
import numpy as np
import pickle
import timeit

'''
Choose the more preferred trajectory from each preference pair as the expert trajectory to train behavioral cloning
'''

def get_preference_data(data_recording_path, idx):
    '''
    get preference data
    '''
    file = os.path.join(data_recording_path, "processed sample " + str(idx) + ".pickle")
    with open(file, 'rb') as handle:
        data = pickle.load(handle)
    
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--data_recording_path', type=str, help="path to recorded preference data")
    parser.add_argument('--data_processed_path', type=str, help="where you want to save the behavioral cloning data")
    parser.add_argument('--max_num_balls', default=2, type=int, help="number of balls originally in the workspace")
    parser.add_argument('--rand_seed', default=2021, type=int, help="random seed")

    args = parser.parse_args()

    np.random.seed(args.rand_seed)

    data_recording_path = args.data_recording_path
    data_processed_path = args.data_processed_path
    os.makedirs(data_processed_path, exist_ok=True)

    max_num_balls = args.max_num_balls

    print(f"data processed path: {data_processed_path}")

    num_preferences = len(os.listdir(data_recording_path))

    data_idx = 0

    start_time = timeit.default_timer() 

    device = torch.device("cuda")
    
    used_trajs = set()
    count_num_balls_reached = {i:0 for i in range(max_num_balls+1)}

    for i in range(num_preferences):
        if i % 50 == 0:
            print("========================================")
            print("current count:", i, " , time passed:", timeit.default_timer() - start_time)

        preference_data = get_preference_data(data_recording_path, i)
        label = preference_data["label"]
        if label:
            emb_states = preference_data["emb_traj_2"]
            eef_states = preference_data["eef_traj_2"]
            sample_idx = preference_data["indices"][1]
            num_balls_reached = preference_data["num_balls_reached_2"]
        else:
            emb_states = preference_data["emb_traj_1"]
            eef_states = preference_data["eef_traj_1"]
            sample_idx = preference_data["indices"][0]
            num_balls_reached = preference_data["num_balls_reached_1"]
        
        group_idx = preference_data["group"]

        if (group_idx, sample_idx) not in used_trajs:

            used_trajs.add((group_idx, sample_idx))
            count_num_balls_reached[num_balls_reached] += 1

            offset = 2

            for j in range(0, eef_states.shape[0]-offset, 1):
                state = torch.cat((eef_states[j], emb_states[j]), dim=0)
                action = eef_states[j+offset]
                
                processed_data = {"state": state, "action": action, "group": group_idx, "sample_idx": sample_idx}
                with open(os.path.join(data_processed_path, "processed sample " + str(data_idx) + ".pickle"), 'wb') as handle:
                    pickle.dump(processed_data, handle, protocol=3)   

                data_idx += 1


    print(f"--- num_balls_reached statistics in more preferred traj: ")
    print(count_num_balls_reached)

    