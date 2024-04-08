import os
import numpy as np
import pickle
import timeit
import torch
import sys
sys.path.append("../pointcloud_representation_learning")
#from reward_model import RewardNetPointCloud as RewardNet
from architecture import AutoEncoder
from utils import *
from reward import RewardNetPointCloud as RewardNet
import matplotlib.pyplot as plt


def compute_predicted_reward(state, encoder, reward_net, device): 
    traj = torch.Tensor(state.transpose(1,0)).unsqueeze(0).float().to(device) 
    embedding = encoder(traj, get_global_embedding=True).unsqueeze(0)
    reward = reward_net.cum_return(embedding)[0]
    #print(reward.cpu().detach().item())
    return reward.cpu().detach().item()

def plot_reward_heat_map(rewards, box_poses, cone_pose, title = "rewards with different box position and a fixed cone position left"):
    fig, ax = plt.subplots()
    max_reward = max(rewards)
    min_reward = min(rewards)   
    for i in range(len(rewards)):
        heat = (rewards[i] - min_reward) / (max_reward - min_reward)
        ax.plot(box_poses[i][0], box_poses[i][1], '.', color=(heat, 0, 0), markersize=20) 
    ax.plot(cone_pose[0], cone_pose[1], ".", markersize=300)
    print("cone z: ",cone_pose[2])
    print("box z: ",box_poses[0][2])
    #ax.legend()
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    os.makedirs(f"./figures", exist_ok=True)
    fig.savefig(f"./figures/{title}.png")
    plt.cla()
    plt.close(fig)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
reward_net = RewardNet()
reward_net.to(device)

encoder =  AutoEncoder(num_points=256, embedding_size=256).to(device)
encoder.load_state_dict(torch.load("/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_AE_boxCone_corrected/weights/weights_40000/epoch 150"))

reward_model_path = '/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_push_corrected/weights/weights_no_eef'
reward_net.load_state_dict(torch.load(os.path.join(reward_model_path, "epoch 100")))

vis_data_path = "/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_push_corrected/vis_reward/sample 4.pickle"
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

plot_reward_heat_map(rewards, box_poses, data["cone_pose"])

# max_reward = max(rewards)
# min_reward = min(rewards)
# #print(rewards[0])

# for i in range(len(partial_pcs)):
#     heat = (rewards[i] - min_reward) / (max_reward - min_reward)
#     plt.plot(box_poses[i][0], box_poses[i][1], '.', color=(heat, 0, 0), markersize=20) 

# plt.plot(data["cone_pose"][0], data["cone_pose"][1], "^", markersize=60)

# print("cone z: ", data["cone_pose"][2])
# print("box z: ", data["box_poses"][0][2])

# # plt.title("Visualization of predicted rewards at multiple states on the scene")
# plt.xlabel("x", fontsize=15)
# plt.ylabel("y", fontsize=15)
# plt.xticks(fontsize=8)
# plt.yticks(fontsize=8)

# plt.show()