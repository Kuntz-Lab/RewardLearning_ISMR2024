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
from reward import RewardNetPointCloudEEF as RewardNet
import matplotlib.pyplot as plt


def compute_predicted_reward(partial_pc, eef_pose, encoder, reward_net, device): 
    traj = torch.Tensor(partial_pc.transpose(1,0)).unsqueeze(0).float().to(device) 
    embedding = encoder(traj, get_global_embedding=True).unsqueeze(0)
    eef_pose = torch.Tensor(eef_pose).unsqueeze(0).unsqueeze(0).float().to(device)
    reward = reward_net.cum_return(eef_pose, embedding)[0]
    # print(reward.cpu().detach().numpy())
    return reward.cpu().detach().item()




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
reward_net = RewardNet()
reward_net.to(device)

encoder =  AutoEncoder(num_points=256, embedding_size=256).to(device)
encoder.load_state_dict(torch.load("/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_AE_boxCone_corrected/weights/weights_40000/epoch 150"))

reward_model_path = '/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_push_corrected/weights/weights_eef'
reward_net.load_state_dict(torch.load(os.path.join(reward_model_path, "epoch 100")))

vis_data_path = "/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_push_corrected/demos_train/group 1 sample 4.pickle"
with open(vis_data_path, 'rb') as handle:
    data = pickle.load(handle)
    

partial_pcs = data["pcds"]
box_states = data["box_states"]
box_poses = []
for i, box_state in enumerate(data["box_states"]):
    box_pose = np.array(list(box_state["pose"]["p"]))
    box_pose = [box_pose[0][0], box_pose[0][1], box_pose[0][2]]
    box_poses.append(box_pose)

assert(len(partial_pcs)==len(box_poses))
print("num_samples: ", len(partial_pcs))

frame=0#70
box_pose = box_poses[frame]
cone_pose = data["cone_pose"]
partial_pc = partial_pcs[frame]

rewards = []
num_samples = 1000
eef_poses = []
for sample in range(num_samples):
    eef_pose = np.array([0, -0.5, 0.01]) + np.random.uniform(low=[-0.2, -0.2, 0], high=[0.2, 0.2, 0], size=3)
    eef_poses.append(eef_pose)
    rew = compute_predicted_reward(partial_pc, eef_pose, encoder, reward_net, device)
    rewards.append(rew)

max_reward = max(rewards)
min_reward = min(rewards)

for i in range(num_samples):
    heat = (rewards[i] - min_reward) / (max_reward - min_reward)
    plt.plot(eef_poses[i][0], eef_poses[i][1], '.', color=(heat, 0, 0), markersize=20) 

plt.plot(cone_pose[0],cone_pose[1], ".", markersize=70)
plt.plot(box_pose[0], box_pose[1], "s", markersize=30)


# print("cone z: ", data["cone_pose"][2])
# print("box z: ", data["box_poses"][0][2])

# plt.title("Visualization of predicted rewards at multiple states on the scene")
plt.xlabel("x", fontsize=15)
plt.ylabel("y", fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.show()