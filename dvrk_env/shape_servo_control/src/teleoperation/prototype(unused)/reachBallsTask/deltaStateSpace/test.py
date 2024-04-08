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

random.seed(1000)

torch.manual_seed(2021)

def get_test_data_point(idx=None, processed_data_path="/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_3/processed_data_test"):
    if idx is None:
        idx = np.random.randint(low=0, high=10000)
    file = os.path.join(processed_data_path, "processed sample " + str(idx) + ".pickle")
    with open(file, 'rb') as handle:
        data = pickle.load(handle)

    return data

def compute_predicted_reward(traj):
    # traj = torch.from_numpy(traj).float().to(device)   
    traj = torch.Tensor(traj).float().to(device) 
    reward = reward_net.cum_return(traj)[0]
    return reward 


def get_demo_traj(group_idx, sample_idx, data_path):
    file = os.path.join(data_path, f"group {group_idx} sample {sample_idx}.pickle")
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

# def get_cmap(n, name='hsv'):
#     '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
#     RGB color; the keyword argument name must be a standard mpl colormap name.'''
#     return plt.cm.get_cmap(name, n)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
reward_net = RewardNet()
reward_net.to(device)

reward_model_path = '/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_3/weights/weights_3'
processed_data_path = "/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_3/processed_data_test"
reward_net.load_state_dict(torch.load(os.path.join(reward_model_path, "epoch 80")))


################################# Plot ground truth vs predicted reward ################################
plt.figure(1)
for group_idx in range(0,10):
    gts = []
    preds = []
    lens = []
    for sample_idx in range(10): 
        test_data_path = "/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_3/demos_test_refined"
        test_traj, gt_reward = get_demo_traj(group_idx, sample_idx, test_data_path)
        predicted_rew = compute_predicted_reward(test_traj)
        # print("+++++++++++++++++++++")
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



# # shuffled_idxs = np.arange(list(range(5001,10000)))
# # np.random.shuffle(shuffled_idxs)

# ####################### Compute accuracy on 1000 pairs ##############################
accuracy = 0
for i in range(0, 1000):
    test_data = get_test_data_point(i, processed_data_path)
    predicted_rew_1 = compute_predicted_reward(test_data["traj_1"])
    predicted_rew_2 = compute_predicted_reward(test_data["traj_2"])
    # assert test_data["traj_1"].shape == test_data["traj_2"].shape
    label = test_data["label"]

    if (predicted_rew_1 - predicted_rew_2 >= 0 and label == False) or \
        (predicted_rew_1 - predicted_rew_2 <= 0 and label == True):
            accuracy += 1

    # if (predicted_rew_1-predicted_rew_2)*(test_data['gt_reward_1']-test_data['gt_reward_2']) > 0:
    #     accuracy += 1
    # else:
    #     rand = np.random.randint(low=1, high=11)
    #     if rand > 8:
    #         print("===================================================")
    #         print(f"Trajectory 1. Predicted: {predicted_rew_1}. Gt: {test_data['gt_reward_1']}")
    #         print(f"Trajectory 2. Predicted: {predicted_rew_2}. Gt: {test_data['gt_reward_2']}")        


print(f"Accuracy: {accuracy}/1000 OR {accuracy/1000*100}%")


# i = 0
# # for i in range(0,10):
# for i in list(range(0,10))+list(range(21,31)):

#     path = "/home/baothach/shape_servo_data/teleoperation/sanity_check_examples/ex_2/test_refined_demos"
#     test_data = get_demo_traj(i, path)
#     predicted_rew = compute_predicted_reward(test_data[0])
#     print("+++++++++++++++++++++")
#     print(f"Trajectory {i}. Predicted: {predicted_rew}. Gt: {test_data[1]}")



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

# path = "/home/baothach/shape_servo_data/teleoperation/sanity_check_examples/ex_2/test_refined_demos"


# accuracy = 0
# count = 0

# for i in list(range(0,10))+list(range(21,31)):
# # for i in list(range(21,31)):
#     test_data_1 = get_demo_traj(i, path)
#     # predicted_rew = compute_predicted_reward(test_data[0])

#     for j in list(range(0,10))+list(range(21,31)):
        
#         if i == j:
#             continue
#         count += 1
        

#         test_data_2 = get_demo_traj(j, path)


#         # print(len(test_data_1[0]), len(test_data_1[0]))

#         # if len(test_data_1[0]) > len(test_data_2[0]):
#         #     traj_1 = sample_trajectory(test_data_2[0], test_data_1[0])
#         #     traj_2 = test_data_2[0]
#         # else:
#         #     traj_2 = sample_trajectory(test_data_1[0], test_data_2[0])  
#         #     traj_1 = test_data_1[0]     
#         traj_1 = test_data_1[0] 
#         traj_2 = test_data_2[0]
        
#         predicted_rew_1 = compute_predicted_reward(traj_1)
#         predicted_rew_2 = compute_predicted_reward(traj_2)



#         if (predicted_rew_1-predicted_rew_2)*(test_data_1[1]-test_data_2[1]) < 0:
#             accuracy += 1
#         else:
#             rand = np.random.randint(low=1, high=11)
#             if rand > 5:
#                 print("===================================================")
#                 print(f"Trajectory 1. Predicted: {predicted_rew_1}. Gt: {test_data_1[1]}")
#                 print(f"Trajectory 2. Predicted: {predicted_rew_2}. Gt: {test_data_2[1]}")   
#         # print("+++++++++++++++++++++")
#         # print(f"Trajectory {i}. Predicted: {predicted_rew}. Gt: {test_data[1]}")


# print(f"Accuracy: {accuracy}/{count} OR {accuracy/count*100}%")        