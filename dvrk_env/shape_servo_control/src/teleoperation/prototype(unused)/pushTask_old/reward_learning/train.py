import torch
import torch.nn as nn
import numpy as np

import torch.optim as optim
import torch.nn.functional as F
import os
from reward import RewardNetPointCloud as RewardNet
import pickle
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
#import sys
#sys.path.append("../")
import numpy as np

import os
from reward import RewardNetPointCloud as RewardNet
from dataset_loader import TrajDataset
import pickle
import argparse

import matplotlib.pyplot as plt

def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    train_loss = 0
    num_batch = 0
    for batch_idx, sample in enumerate(dataloader):
        num_batch+=1
        obj_traj_i, obj_traj_j, eef_traj_i, eef_traj_j, label  = sample

        obj_traj_i = obj_traj_i.float().to(device)
        obj_traj_j= obj_traj_j.float().to(device)
        # eef_traj_i = eef_traj_i.float().to(device)
        # eef_traj_j= eef_traj_j.float().to(device)
        label = torch.from_numpy(np.array(label)).long().to(device)

        # print(obj_traj_i.shape)

        #zero out gradient
        optimizer.zero_grad()

        # compute loss and backward
        outputs, abs_rewards = model(obj_traj_i, obj_traj_j)
        loss = loss_fn(outputs, label) #+ l1_reg * abs_rewards
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        if batch_idx % 10 == 0:
            loss, current = loss.item(), (batch_idx + 1) * len(obj_traj_i)
            print(f"train loss: {loss:>7f}  average train loss: {train_loss/num_batch} progress: [{current:>5d}/{size:>5d}]")

        avg_train_loss = train_loss/num_batch
        return avg_train_loss

            
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
            # eef_traj_i = eef_traj_i.float().to(device)
            # eef_traj_j= eef_traj_j.float().to(device)
            label = torch.from_numpy(np.array(label)).long().to(device)

            outputs, abs_rewards = model(obj_traj_i, obj_traj_j)

            test_loss += loss_fn(outputs, label).item()
            correct += (outputs.argmax(1) == label).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    print("============================")

    return test_loss

def plot_train_test_curves(train_losses, test_losses, title):
    epochs = [i for i in range(len(train_losses))]
    fig, ax = plt.subplots()
    ax.plot(epochs, train_losses, label="train_losses")
    ax.plot(epochs, test_losses, label="test_losses")
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel("epochs")
    ax.set_ylabel("losses")
    os.makedirs(f"./figures", exist_ok=True)
    fig.savefig(f"./figures/{title}.png")
    plt.cla()
    plt.close(fig)


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description='Options')

    parser.add_argument('--rmp', default='/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_push_corrected/weights/weights_no_eef', type=str, help="path to reward model")
    parser.add_argument('--tdp', default='/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_push_corrected/data_processed_train', type=str, help="path to training data")
    parser.add_argument('--vdp', default='/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_push_corrected/data_processed_test', type=str, help="path to validation data")

    args = parser.parse_args()

    os.makedirs(args.rmp, exist_ok=True)
    print("num_total_data: ", len(os.listdir(args.tdp)))

    # Now we create a reward network and optimize it using the training data.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    reward_net = RewardNet()
    reward_net.to(device)
    reward_model_path = args.rmp
    training_data_path = args.tdp
    validation_data_path = args.vdp

    train_dataset = TrajDataset(training_data_path)
    test_dataset = TrajDataset(validation_data_path)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=True)
    
    print("training data size: ", len(train_dataset))
    print("test data size: ", len(test_dataset))
    print("training data path:", training_data_path)

    epochs = 101
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(reward_net.parameters(), lr=0.00015)
    train_losses = []
    test_losses = []
    #scheduler = torch.nn.optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.1)
    for epoch in range(epochs):
        print("============================")
        print(f"Epoch {epoch+1}\n-------------------------------")
        train_loss = train(train_loader, reward_net, loss_fn, optimizer, device)
        train_losses.append(train_loss)
        test_loss = test(test_loader, reward_net, loss_fn, device)
        test_losses.append(test_loss)
        if epoch % 2 == 0:        
            torch.save(reward_net.state_dict(), os.path.join(reward_model_path, "epoch " + str(epoch)))

    plot_train_test_curves(train_losses, test_losses, "train test losses of reward net point cloud")
    
    print("Done!")