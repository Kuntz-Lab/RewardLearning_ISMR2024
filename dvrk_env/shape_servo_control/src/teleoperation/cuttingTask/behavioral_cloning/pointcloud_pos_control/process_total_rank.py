import sys

import torch
import math
import argparse
import os
import numpy as np
import pickle
import timeit

'''
Choose the top k% more preferred trajectory from a total ranking of over trajectories to train behavioral cloning
'''


def get_trajectory(data_recording_path, group, sample_idx):
    '''
    get a trajectory from a group and a specific sample index.
    Returns data
    '''
    file = os.path.join(data_recording_path, f"group {group} sample {sample_idx}.pickle")
    with open(file, 'rb') as handle:
        data = pickle.load(handle)
    
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--data_recording_path', type=str, help="path to recorded data")
    parser.add_argument('--data_processed_path', type=str, help="where you want to save the behavioral cloning data")
    parser.add_argument('--num_groups', default=30, type=int, help="num groups to process")
    parser.add_argument('--num_samples_per_group', default=30, type=int, help="num samples per group")
    parser.add_argument('--percent', type=float, help="k, specifying the top k percent of trajectories you want")
    parser.add_argument('--max_num_balls', default=2, type=int, help="number of balls originally in the workspace")
    parser.add_argument('--rand_seed', default=2021, type=int, help="random seed")

    args = parser.parse_args()

    np.random.seed(args.rand_seed)

    data_recording_path = args.data_recording_path
    data_processed_path = args.data_processed_path
    os.makedirs(data_processed_path, exist_ok=True)

    print(f"data processed path: {data_processed_path}")

    k_percent = args.percent
    assert(k_percent <=100)
    k_fraction = k_percent/100

    max_num_balls = args.max_num_balls
    num_groups = args.num_groups
    num_samples_per_group = args.num_samples_per_group

    start_time = timeit.default_timer() 

    device = torch.device("cuda")
    
    print("---- process trajectories into total ranking")
    total_ranking = []
    for group_idx in range(num_groups):
        if group_idx % 2 == 0:
                print("========================================")
                print("current group:", group_idx, " , time passed:", timeit.default_timer() - start_time)

        for sample_idx in range(num_samples_per_group):
            
            traj = get_trajectory(data_recording_path, group_idx, sample_idx)
            data_for_sort = {"reward": traj["gt_reward"], "group": group_idx, "sample_idx": sample_idx}
            total_ranking.append(data_for_sort)

    total_ranking = sorted(total_ranking, key=lambda traj: traj["reward"]) # low reward to high reward

    print("---- select the most preferred trajectories as training data of behavioral cloning")
    data_idx = 0
    best_traj_idx = len(total_ranking) - 1
    count_num_balls_reached_in_topk_traj = {i:0 for i in range(max_num_balls+1)}

    for i in range(best_traj_idx, best_traj_idx-int(k_fraction*len(total_ranking)), -1):
        traj_in_rank = total_ranking[i]
        group_idx = traj_in_rank["group"]
        sample_idx = traj_in_rank["sample_idx"]
        traj_full_data = get_trajectory(data_recording_path, group_idx, sample_idx)

        emb_states = traj_full_data["emb_traj"]
        eef_states = traj_full_data["eef_traj"]
        num_balls_reached = traj_full_data["num_balls_reached"]
        count_num_balls_reached_in_topk_traj[num_balls_reached] += 1

        offset = 2 #5 #2 #10

        for j in range(0, eef_states.shape[0]-offset, 1):
            state = torch.cat((eef_states[j], emb_states[j]), dim=0)
            action = eef_states[j+offset]
            
            processed_data = {"state": state, "action": action, "group": group_idx, "sample_idx": sample_idx}
            with open(os.path.join(data_processed_path, "processed sample " + str(data_idx) + ".pickle"), 'wb') as handle:
                pickle.dump(processed_data, handle, protocol=3)   

            data_idx += 1

    print(f"--- num_balls_reached statistics of top {k_percent}% trajectory: ")
    print(count_num_balls_reached_in_topk_traj)


    