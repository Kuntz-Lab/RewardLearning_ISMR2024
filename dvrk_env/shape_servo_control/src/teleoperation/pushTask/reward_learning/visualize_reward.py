import os
import numpy as np
import pickle
import torch
import sys
sys.path.append("../pointcloud_representation_learning")
from architecture import AutoEncoder
from reward import RewardNetPointCloud as RewardNet
import matplotlib.pyplot as plt
import argparse


def compute_predicted_reward(state, encoder, reward_net, device): 
    traj = torch.Tensor(state.transpose(1,0)).unsqueeze(0).float().to(device) 
    embedding = encoder(traj, get_global_embedding=True).unsqueeze(0)
    reward = reward_net.cum_return(embedding)[0]
    return reward.cpu().detach().item()

def plot_reward_heat_map(rewards, box_poses, cone_pose, title = "rewards with different push object and a fixed target position top right"):
    fig, ax = plt.subplots()
    max_reward = max(rewards)
    min_reward = min(rewards)   
    for i in range(len(rewards)):
        heat = (rewards[i] - min_reward) / (max_reward - min_reward)
        ax.plot(box_poses[i][0], box_poses[i][1], '.', color=(heat, 0, 0), markersize=20) 
    ax.plot(cone_pose[0], cone_pose[1], ".", markersize=60)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--rmp', default=f"/home/dvrk/LfD_data/ex_push/weights/weights_1/epoch_100", type=str, help="path to reward model")
    parser.add_argument('--vis_data_path', default=f"/home/dvrk/LfD_data/ex_push/vis_reward/sample_0.pickle", type=str, help="path to data for visualization")
    parser.add_argument('--AE_model_path', default="/home/dvrk/LfD_data/ex_AE_boxCone/weights/weights_1/epoch_150", type=str, help="path to pre-trained autoencoder weights")

    args = parser.parse_args()

    AE_model_path = args.AE_model_path
    reward_model_path = args.rmp
    vis_data_path = args.vis_data_path

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    encoder =  AutoEncoder(num_points=256, embedding_size=256).to(device)
    encoder.load_state_dict(torch.load(AE_model_path))

    reward_net = RewardNet().to(device)
    reward_net.load_state_dict(torch.load(reward_model_path))

    with open(vis_data_path, 'rb') as handle:
        data = pickle.load(handle)
        
    partial_pcs = data["partial_pc"]
    box_poses = data["box_poses"]
    assert(len(partial_pcs)==len(box_poses))
    print("num_samples: ", len(partial_pcs))

    rewards = []
    for state in partial_pcs:
        rew = compute_predicted_reward(state, encoder, reward_net, device)
        rewards.append(rew)

    plot_reward_heat_map(rewards, box_poses, data["cone_pose"])