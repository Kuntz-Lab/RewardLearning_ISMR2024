
import pickle

#vis_data_path = "/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_trace_curve/demos_train_5balls/group 0 sample 47.pickle" #7
# vis_data_path = "/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_trace_curve/demos_train_5balls_varied/group 0 sample 10.pickle" #7
#vis_data_path = "/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_trace_curve/demos_train_5balls/group 0 sample 10.pickle"
#vis_data_path = "/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_trace_curve/demos_train_1ball/group 0 sample 10.pickle" 
#vis_data_path = "/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_trace_curve_corrected/demos_train_straight_3D_flat/group 26 sample 1.pickle" #7
vis_data_path = "/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_trace_curve_corrected/demos_train_straight_3D_flat_2ball/group 0 sample 1.pickle" #7
with open(vis_data_path, 'rb') as handle:
    data = pickle.load(handle)

import torch
import os
import numpy as np
import timeit
import roslib.packages as rp
import sys
sys.path.append("./pointcloud_representation_learning")
pkg_path = rp.get_pkg_dir('shape_servo_control')
sys.path.append(pkg_path + '/src')

#import open3d

from util.isaac_utils import *
#from reward_model import RewardNetPointCloud as RewardNet
from architecture import AutoEncoder
#from utils import *
from reward_lstm import RewardLSTM as RewardNet
import matplotlib.pyplot as plt
from curve import *

def compute_predicted_reward(obj_emb, eef_traj, eef_pose, reward_net, device): 
    eef_pose = torch.Tensor(eef_pose).unsqueeze(0).float().to(device)
    eef_traj = torch.cat((eef_traj, eef_pose), dim=0).unsqueeze(0)
    obj_emb = obj_emb.unsqueeze(0)
    with torch.no_grad():
        reward = reward_net(eef_traj, obj_emb)[0]
    return reward.item()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
reward_net = RewardNet(input_dim=(256, 3), embedding_dim=128,  hidden_dim=128, output_dim=1,  n_layers=1, drop_prob=0).to(device)
reward_net.to(device)

encoder =  AutoEncoder(num_points=256, embedding_size=256).to(device)
encoder.load_state_dict(torch.load("/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_AE_balls_corrected/weights_straight3D_partial_flat_2ball/weights_1/epoch 150"))

reward_model_path = '/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_trace_curve_corrected/weights_straight_3D_flat_2ball_lstm/weights_1'
reward_net.load_state_dict(torch.load(os.path.join(reward_model_path, "epoch 100")))

print("num_balls_total: ", len(data["balls_xyz"]))
print("reach cps: ", data["reach_cps"])
print("num balls reached: ", data["num_balls_reached"])
print("len traj: ", len(data["pcds"]))

partial_pcs = data["pcds"]
total_traj_len = len(partial_pcs)

partial_pc = partial_pcs[0]
balls_xyz = data["balls_xyz"]

partial_pc = np.array(down_sampling(partial_pc, num_pts=256))
pc_traj = torch.Tensor(partial_pc.transpose(1,0)).unsqueeze(0).float().to(device) 
obj_emb = encoder(pc_traj, get_global_embedding=True)
obj_emb = obj_emb.expand(total_traj_len, -1) # (T, 256)

eef_traj = []
for i, eef_state in enumerate(data["eef_states"]):
    eef_pose = np.array(list(eef_state["pose"]["p"]))
    eef_traj.append(eef_pose)
eef_traj = np.array(eef_traj)
assert(len(eef_traj)==total_traj_len)
eef_traj = torch.Tensor(eef_traj).float().to(device) # ( T, 3)


####################################### sampling different eef poses at time: frame+1 #################################
frame = 20
traj_len = frame + 1
weights_list = data["weights_list"]
rewards = []
num_samples = 2000
eef_poses = []

#z=0.1
for sample in range(num_samples):
    # x = np.random.uniform(low=-0.075, high=0.05)
    # y = np.random.uniform(low=-0.52, high=-0.48)
    x = np.random.uniform(low=-0.1, high=0.1)
    y = np.random.uniform(low=-0.6, high=-0.4)
    #eef_pose = np.random.uniform(low=[-0.11, -0.5, height], high=[-0.05, -0.38, height], size=3)
    z = abs(poly3D(x, y, weights_list))
    eef_pose = np.array([x,y,z])
    eef_poses.append(eef_pose)
    rew = compute_predicted_reward(obj_emb[:traj_len+1], eef_traj[:traj_len], eef_pose, reward_net, device)
    #print(rew)
    rewards.append(rew)

######################################### 3D plot ################################################################
# max_reward = max(rewards)
# min_reward = min(rewards)

# fig = plt.figure()
# ax = plt.axes(projection='3d')

# for i in range(num_samples):
#     heat = (rewards[i] - min_reward) / (max_reward - min_reward)
#     ax.scatter(eef_poses[i][0], eef_poses[i][1], eef_poses[i][2], marker='.', c=[[heat, 0, 0]]) 

# for ball_xyz in balls_xyz:
#     ax.scatter(ball_xyz[0],ball_xyz[1], balls_xyz[2], "o")

# for i in range(traj_len+1):
#     ax.scatter(eef_traj[i, 0].item(), eef_traj[i, 1].item(), eef_traj[i, 2].item(), '.', c=[[0,1,0]]) 

# ax.set_title(f"Predicted rewards with different eef poses at frame {frame+1} conditioned on previous traj")
# ax.set_xlabel("x")
# ax.set_ylabel("y")
# ax.set_zlabel("z")
# os.makedirs(f"./figures", exist_ok=True)
# fig.savefig('./figures/scatterplot3D.png')

########################################### 2D plot ###############################################################
max_reward = max(rewards)
min_reward = min(rewards)

fig = plt.figure()
ax = plt.axes(projection='rectilinear')

for i in range(num_samples):
    heat = (rewards[i] - min_reward) / (max_reward - min_reward)
    ax.plot(eef_poses[i][0], eef_poses[i][1], '.', color=(heat, 0, 0), markersize=10) 

for ball_xyz in balls_xyz:
    ax.plot(ball_xyz[0],ball_xyz[1], "o", color=(0, 0, 1), markersize=20)

for i in range(traj_len+1):
    ax.plot(eef_traj[i, 0].item(), eef_traj[i, 1].item(), '.', color=(0, 1, 0)) 

ax.set_title(f"Predicted rewards with different eef poses at frame {frame+1} conditioned on previous traj")
ax.set_xlabel("x")
ax.set_ylabel("y")
os.makedirs(f"./figures", exist_ok=True)
fig.savefig('./figures/scatterplot2D_2ball_lstm.png')



# for i in range(num_samples):
#     heat = (rewards[i] - min_reward) / (max_reward - min_reward)
#     plt.plot(eef_poses[i][0], eef_poses[i][1], '.', color=(heat, 0, 0), markersize=20) 

# for ball_xyz in balls_xyz:
#     plt.plot(ball_xyz[0],ball_xyz[1], "o", markersize=15)


# plt.title(f"Predicted rewards with different eef poses at frame {frame+1} conditioned on previous traj")
# plt.xlabel("x", fontsize=10)
# plt.ylabel("y", fontsize=10)
# plt.xticks(fontsize=8)
# plt.yticks(fontsize=8)

# plt.show()