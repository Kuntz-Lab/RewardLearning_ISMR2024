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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
reward_net = RewardNet()
reward_net.to(device)

reward_model_path = '/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_indicator_0and1/weights/weights_0'
reward_net.load_state_dict(torch.load(os.path.join(reward_model_path, "epoch 100")))


eef_samples = []
rewards = []
num_samples = 1500


# max_reward = compute_predicted_reward(np.array(data["mid pose"]), obj_embedding)
goal_indicator = 0
for _ in range(num_samples):
    eef_pose = np.random.uniform(low=[-0.07,-0.6,0.16+0.015], high=[0.07,-0.3,0.16+0.015+0.001], size=3)
    mid_pose = np.array([0.05, -0.44, 0.16+0.015]) #try swap y-coordinate of mid and goal
    goal_pose = np.array([-0.05, -0.44, 0.16+0.015])
    state = np.concatenate((np.array([goal_indicator]), eef_pose, mid_pose, goal_pose), axis=None)
    reward = compute_predicted_reward(state)

    eef_samples.append(eef_pose)
    rewards.append(reward)

max_reward = max(rewards)
min_reward = min(rewards)

for i in range(len(eef_samples)):
    heat = (rewards[i] - min_reward) / (max_reward - min_reward)
    plt.plot(eef_samples[i][0], eef_samples[i][1], '.', color=(heat, 0, 0), markersize=20) 

plt.plot(mid_pose[0], mid_pose[1], "y*", markersize=30)
plt.plot(goal_pose[0], goal_pose[1], "o", markersize=30)

# plt.title("Visualization of predicted rewards at multiple states on the scene")
plt.xlabel("x", fontsize=15)
plt.ylabel("y", fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.show()