import os
import numpy as np
import pickle
import timeit
import torch
import sys
#sys.path.append("/home/baothach/dvrk_ws/src/dvrk_env/shape_servo_control/src/teleoperation/learn_reward")
#from reward_model import RewardNetPointCloud as RewardNet
from reward import RewardNet2 as RewardNet
import matplotlib.pyplot as plt



def compute_predicted_reward(state): 
    traj = torch.Tensor(state).unsqueeze(0).float().to(device) 
    reward = reward_net.cum_return(traj)[0]
    # print(reward.cpu().detach().numpy())
    return reward.cpu().detach().numpy() 

def sample_eef_pose(data):
    z = data["mid pose"][2]
 
    # return np.random.uniform(low=[-0.0,-0.55,z], high=[0.1,-0.35,z+0.001], size=3) 
    # return np.random.uniform(low=[-0.04,-0.55,z], high=[0.04,-0.35,z+0.001], size=3)
    return np.random.uniform(low=[-0.1,-0.55,z], high=[0.1,-0.35,z+0.001], size=3)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
reward_net = RewardNet()
reward_net.to(device)

reward_model_path = '/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_3/weights/weights_3'
training_data_path = "/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_3/processed_data_train"
reward_net.load_state_dict(torch.load(os.path.join(reward_model_path, "epoch 100")))

# eef_samples = []
# rewards = []
# num_samples = 1500

# for group_idx in range(2,3):  #range1,2, test  
#     for sample_idx in range(1): 

#         data_path = "/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_3/demos_test_refined"
#         file = os.path.join(data_path, f"group {group_idx} sample {sample_idx}.pickle")
#         with open(file, 'rb') as handle:
#             data = pickle.load(handle)

#         # max_reward = compute_predicted_reward(np.array(data["mid pose"]), obj_embedding)

#         for _ in range(num_samples):
#             eef_pose = sample_eef_pose(data)
#             mid_pose = data["mid pose"]
#             goal_pose = data["goal pose"]
#             state = np.concatenate((eef_pose-mid_pose, eef_pose-goal_pose), axis=None)
#             reward = compute_predicted_reward(state)

#             # print("shapes:", eef_pose.shape, reward)

#             eef_samples.append(eef_pose)
#             rewards.append(reward)

#             # print(reward/max_reward)

#             # plt.plot(eef_pose[0], eef_pose[1], '.', color=(reward/max_reward, 0, 0), markersize=20) 

#         max_reward = max(rewards)
#         min_reward = min(rewards)

#         for i in range(len(eef_samples)):
#             heat = (rewards[i] - min_reward) / (max_reward - min_reward)
#             plt.plot(eef_samples[i][0], eef_samples[i][1], '.', color=(heat, 0, 0), markersize=20) 

#         plt.plot(data["mid pose"][0], data["mid pose"][1], "y*", markersize=30)
#         plt.plot(data["goal pose"][0], data["goal pose"][1], "o", markersize=30)

#     plt.title("Visualization of predicted rewards at multiple states on the scene")
#     plt.xlabel("x")
#     plt.ylabel("y")

#     plt.show()

eef_samples = []
rewards = []
num_samples = 1500


# max_reward = compute_predicted_reward(np.array(data["mid pose"]), obj_embedding)

for _ in range(num_samples):
    eef_pose = np.random.uniform(low=[-0.07,-0.48,0.16+0.015], high=[0.07,-0.42,0.16+0.015+0.001], size=3)
    mid_pose = np.array([0.05, -0.44, 0.16+0.015+0.05])
    goal_pose = np.array([-0.05, -0.44, 0.16+0.015+0.05])
    state = np.concatenate(((eef_pose-mid_pose)*0, (eef_pose-goal_pose)*1), axis=None)
    reward = compute_predicted_reward(state)

    # print("shapes:", eef_pose.shape, reward)

    eef_samples.append(eef_pose)
    rewards.append(reward)

    # print(reward/max_reward)

    # plt.plot(eef_pose[0], eef_pose[1], '.', color=(reward/max_reward, 0, 0), markersize=20) 

max_reward = max(rewards)
min_reward = min(rewards)

for i in range(len(eef_samples)):
    heat = (rewards[i] - min_reward) / (max_reward - min_reward)
    plt.plot(eef_samples[i][0], eef_samples[i][1], '.', color=(heat, 0, 0), markersize=20) 

plt.plot(mid_pose[0], mid_pose[1], "y*", markersize=30)
plt.plot(goal_pose[0], goal_pose[1], "o", markersize=30)

plt.title("Visualization of predicted rewards at multiple states on the scene")
plt.xlabel("x")
plt.ylabel("y")

plt.show()