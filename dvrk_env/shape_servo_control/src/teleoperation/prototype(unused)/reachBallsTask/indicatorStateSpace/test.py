import torch
import torch.nn as nn
import numpy as np

import torch.optim as optim
import torch.nn.functional as F
import os
from reward import RewardNet2 as RewardNet
import pickle
import random
from random import sample
import matplotlib.pyplot as plt 
import argparse

random.seed(1000)

torch.manual_seed(2021)

def get_processed_data(processed_sample_idx, processed_data_path):
    file = os.path.join(processed_data_path, "processed sample " + str(processed_sample_idx) + ".pickle")
    with open(file, 'rb') as handle:
        data = pickle.load(handle)

    return data

def compute_predicted_reward(traj, reward_net):
    # traj = torch.from_numpy(traj).float().to(device)   
    traj = torch.Tensor(traj).float().to(device) 
    reward = reward_net.cum_return(traj)[0]
    return reward 


def get_refined_traj(group_idx, sample_idx, data_refined_path):
    file = os.path.join(data_refined_path, f"group {group_idx} sample {sample_idx}.pickle")
    with open(file, 'rb') as handle:
        data = pickle.load(handle)

    return data["traj"], data["gt_reward"]


def normalize(arr, t_min=0, t_max=1):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)    
    for i in arr:
        temp = (((i - min(arr))*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Options')
    ### CHANGE ####
    is_train = False
    suffix = "train" if is_train else "test"
    parser.add_argument('--data_refined_path', default= f"/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_indicator_1/demos_{suffix}_refined" , type=str, help="location of existing refined data")
    parser.add_argument('--data_processed_path', default= f"/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_indicator_1/processed_data_{suffix}" , type=str, help="location of existing processed data")
    parser.add_argument('--reward_model_path', default= f'/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_indicator_1/weights/weights_1' , type=str, help="location of saved reward net")
    ### CHANGE ####
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    reward_net = RewardNet()
    reward_net.to(device)

    data_refined_path = args.data_refined_path
    data_processed_path = args.data_processed_path
    reward_model_path = args.reward_model_path
    reward_net.load_state_dict(torch.load(os.path.join(reward_model_path, "epoch 80")))


################################# Plot ground truth vs predicted reward ################################
plt.figure(1)
for group_idx in range(0,10):
    gts = []
    preds = []
    lens = []
    for sample_idx in range(10): 
        test_traj, gt_reward = get_refined_traj(group_idx, sample_idx, data_refined_path)
        predicted_rew = compute_predicted_reward(test_traj, reward_net)
        print(f"group{group_idx}, sample {sample_idx}. Predicted: {predicted_rew}. Gt: {gt_reward}")
        # print(type(predicted_rew.cpu().detach().numpy()), type(test_data[1]))
        gts.append(gt_reward)#/len(test_data[0])) 
        preds.append(predicted_rew.cpu().detach().numpy())   
        lens.append(len(test_traj))

    # plt.scatter(np.array(gts), normalize(np.array(preds), t_min=min(gts), t_max=max(gts)), color=np.random.rand(3,))
    plt.scatter(normalize(np.array(gts), t_min=0, t_max=1), normalize(np.array(preds), t_min=0, t_max=1), color=np.random.rand(3,))

plt.title('Predicted vs. Ground Truth Reward')
plt.xlabel("Ground Truth Returns")
plt.ylabel("Predicted Returns (normalized)")

plt.show()



# ####################### Compute accuracy on 1000 pairs ##############################
accuracy = 0
num_test_data = 1000
for i in range(0, num_test_data):
    test_data = get_processed_data(i, data_processed_path)
    predicted_rew_1 = compute_predicted_reward(test_data["traj_1"], reward_net)
    predicted_rew_2 = compute_predicted_reward(test_data["traj_2"], reward_net)
    # assert test_data["traj_1"].shape == test_data["traj_2"].shape
    label = test_data["label"]

    if (predicted_rew_1 - predicted_rew_2 >= 0 and label == False) or \
        (predicted_rew_1 - predicted_rew_2 <= 0 and label == True):
            accuracy += 1

print(f"Accuracy: {accuracy}/1000 OR {accuracy/num_test_data*100}%")