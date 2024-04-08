import os
import numpy as np
import pickle
import timeit
import torch
import sys
sys.path.append("./pointcloud_representation_learning")
#from reward_model import RewardNetPointCloud as RewardNet
from architecture import AutoEncoder
from utils import *
from reward import RewardNetPointCloud as RewardNet
import matplotlib.pyplot as plt

print(sys.path)


def compute_predicted_reward(state, encoder, reward_net, device): 
    traj = torch.Tensor(state.transpose(1,0)).unsqueeze(0).float().to(device) 
    embedding = encoder(traj, get_global_embedding=True)
    reward = reward_net.cum_return(embedding)[0]
    # print(reward.cpu().detach().numpy())
    return reward.cpu().detach().numpy() 




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
reward_net = RewardNet()
reward_net.to(device)

encoder =  AutoEncoder(num_points=256, embedding_size=256).to(device)
encoder.load_state_dict(torch.load("/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_AE_boxConeCloser/weights/weights_1/epoch 140"))

reward_model_path = '/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_push/weights_no_eef/weights_1'
reward_net.load_state_dict(torch.load(os.path.join(reward_model_path, "epoch 100"))) #epoch 60

vis_data_path = "/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_push/vis_reward/sample 1.pickle"
with open(vis_data_path, 'rb') as handle:
    data = pickle.load(handle)
    

partial_pcs = data["partial_pcs"]
box_poses = data["box_poses"]
#print(box_poses[0])
# print(partial_pcs[0][0])
assert(len(partial_pcs)==len(box_poses))
print("num_samples: ", len(partial_pcs))

rewards = []
for state in partial_pcs:
    rew = compute_predicted_reward(state, encoder, reward_net, device)
    rewards.append(rew)

max_reward = max(rewards)
min_reward = min(rewards)
#print(rewards[0])

for i in range(len(partial_pcs)):
    heat = (rewards[i] - min_reward) / (max_reward - min_reward)
    plt.plot(box_poses[i][0], box_poses[i][1], '.', color=(heat, 0, 0), markersize=20) 

plt.plot(data["cone_pose"][0], data["cone_pose"][1], "^", markersize=60)

print("cone z: ", data["cone_pose"][2])
print("box z: ", data["box_poses"][0][2])

# plt.title("Visualization of predicted rewards at multiple states on the scene")
plt.xlabel("x", fontsize=15)
plt.ylabel("y", fontsize=15)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)

plt.show()