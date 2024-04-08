import os
import timeit
import argparse
import sys
import pickle
sys.path.append("../reward_learning/fully_connected")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from reward import RewardNetPointCloudEEF as RewardNet
from dataset_loader import TrajDataset
import matplotlib.pyplot as plt


def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for sample in dataloader:
            obj_traj_i, obj_traj_j, eef_traj_i, eef_traj_j, label  = sample

            obj_traj_i = obj_traj_i.float().to(device)
            obj_traj_j= obj_traj_j.float().to(device)
            eef_traj_i = eef_traj_i.float().to(device)
            eef_traj_j= eef_traj_j.float().to(device)
            label = torch.from_numpy(np.array(label)).long().to(device)

            outputs, abs_rewards = model(eef_traj_i, obj_traj_i, eef_traj_j, obj_traj_j)

            test_loss += loss_fn(outputs, label).item()
            
            correct += (outputs.argmax(1) == label).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    print("============================")
    return test_loss, correct

def plot_bar_chart(xs, ys, title, path, x_label, y_label, tick_labels):
    fig, ax = plt.subplots()
    ax.bar(xs, ys, width=1.5 ,align='center', tick_label=tick_labels)
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_yticks([0.05*i for i in range(21)])
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    os.makedirs(f"{path}", exist_ok=True)
    fig.savefig(f"{path}/{title}.png")
    plt.cla()
    plt.close(fig)

if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--data_processed_root_path', type=str, help="path to root folder containing processed data")
    parser.add_argument('--reward_model_root_path', type=str, help="path to root folder containing reward model")
    parser.add_argument('--vdp', type=str, help="path to validation data")

    args = parser.parse_args()
    data_processed_root_path = args.data_processed_root_path
    reward_model_root_path = args.reward_model_root_path
    vdp = args.vdp
    vdp_size = len(os.listdir(vdp))

    args = parser.parse_args()

    with open(os.path.join(data_processed_root_path, "num_data_list.pickle"), 'rb') as handle:
        num_data_list = pickle.load(handle)

    print("############## num_data_list: ", num_data_list)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    loss_fn = nn.CrossEntropyLoss()

    train_accuracies = []
    test_accuracies = []
    for i, num_data in enumerate(num_data_list):
        print(f"--- finished {i+1}th num_data : {num_data}")
        reward_net = RewardNet()
        reward_net.to(device)
        reward_net.load_state_dict(torch.load(os.path.join(reward_model_root_path, f"weights_{num_data}/epoch_200")))

        train_dataset_path = os.path.join(data_processed_root_path, f"processed_data_train_{num_data}")
        train_dataset = TrajDataset(train_dataset_path)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=num_data//2, shuffle=True)
        _, train_acc = test(train_loader, reward_net, loss_fn, device)

        test_dataset_path = vdp
        test_dataset = TrajDataset(test_dataset_path)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=vdp_size//2, shuffle=True)
        _, test_acc = test(test_loader, reward_net, loss_fn, device)

        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

    plot_path = os.path.join(data_processed_root_path, "figures")
    plot_bar_chart(xs=[2*j for j in range(len(num_data_list))], ys=train_accuracies, title="train accuracies across training data sizes", path=plot_path, x_label="number of training data", y_label="accuracy", tick_labels=num_data_list)
    plot_bar_chart(xs=[2*j for j in range(len(num_data_list))], ys=test_accuracies, title="test accuracies across training data sizes", path=plot_path, x_label="number of training data", y_label="accuracy", tick_labels=num_data_list)