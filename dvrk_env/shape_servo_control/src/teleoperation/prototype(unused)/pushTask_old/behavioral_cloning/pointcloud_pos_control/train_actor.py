import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
#import sys
#sys.path.append("../")
from dataset_loader import BCDataset
from policy_BC_PC import ActorPC
import os
import argparse
import matplotlib.pyplot as plt

from torch.distributions import MultivariateNormal

def train(model, dataloader, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    train_loss = 0
    num_batch = 0
    for batch_idx, sample in enumerate(dataloader):
        num_batch += 1
        states, actions = sample
        states = states.float().to(device)
        #print(f"states shape: {states.shape}")
        actions = actions.float().to(device)
        #print(f"actions shape: {actions.shape}")
        action_means = model(states)
        #print(f"action_mean shape: {action_means.shape}")
        covariance = torch.diag(model.log_std.exp() * model.log_std.exp())
        dist = MultivariateNormal(action_means, scale_tril=covariance)
        loss = -dist.log_prob(actions).sum() #+ 500*torch.norm(action_means**2) + 500*torch.norm(model.log_std.exp())
        

        optimizer.zero_grad()
        loss.backward()
        train_loss += loss.item()/len(actions)
        optimizer.step()
        if batch_idx % 10 == 0:
            loss, current = loss.item(), (batch_idx + 1) * len(states)
            print(f"train loss: {loss:>7f}  average train loss: {train_loss/num_batch} progress: [{current:>5d}/{size:>5d}]")

    avg_train_loss = train_loss/num_batch
    return avg_train_loss

def test(model, dataloader, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_L2_loss = 0
    test_log_prob_loss = 0
    with torch.no_grad():
        for sample in dataloader:
            states, actions = sample
            states = states.float().to(device)
            actions = actions.float().to(device)
            # L2 loss
            sampled_actions = model.act(states)
            L2_loss = torch.sum((sampled_actions - actions)**2)
            test_L2_loss += L2_loss.item()/len(actions)
            # log prob loss
            action_means = model(states)
            covariance = torch.diag(model.log_std.exp() * model.log_std.exp())
            dist = MultivariateNormal(action_means, scale_tril=covariance)
            log_prob_loss = -dist.log_prob(actions).sum()
            test_log_prob_loss += log_prob_loss.item()/len(actions)
    test_L2_loss /= num_batches
    test_log_prob_loss /= num_batches
    print(f"Test Error: \n Avg test L2 loss: {test_L2_loss:>8f} \n  Avg test log prob loss: {test_log_prob_loss:>8f}")
    print("============================")
    return test_L2_loss, test_log_prob_loss


def plot_learning_curve(losses, title):
    epochs = [i for i in range(len(losses))]
    fig, ax = plt.subplots()
    ax.plot(epochs, losses)
    ax.set_title(title)
    ax.set_xlabel("epochs")
    ax.set_ylabel("losses")
    os.makedirs(f"./figures", exist_ok=True)
    fig.savefig(f"./figures/{title}.png")

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

    parser = argparse.ArgumentParser(description='Options')

    parser.add_argument('--mp', default='/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_push_BC_corrected/pointcloud_pos_control/weights/weights_max_prob_pc', type=str, help="path to model")
    parser.add_argument('--tdp', default='/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_push_BC_corrected/pointcloud_pos_control/data_processed_train', type=str, help="path to training data")
    parser.add_argument('--vdp', default='/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_push_BC_corrected/pointcloud_pos_control/data_processed_test', type=str, help="path to validation data")


    args = parser.parse_args()
    model_path = args.mp
    training_data_path = args.tdp
    validation_data_path = args.vdp


    os.makedirs(args.mp, exist_ok=True)
    print("total size of dataset: ", len(os.listdir(args.tdp)))

    # Now we create a reward network and optimize it using the training data.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    policy = ActorPC(act_dim=3, initial_std=1.0).to(device)
    policy.to(device)

    # train_len = 14200#28000
    # # test_len = 2840
    # # total_len = train_len + test_len

    # dataset_path = training_data_path
    # dataset = BCDataset(dataset_path)
    # train_dataset = torch.utils.data.Subset(dataset, range(0, train_len))
    #test_dataset = torch.utils.data.Subset(dataset, range(train_len, total_len))
    
    train_dataset = BCDataset(training_data_path)
    test_dataset = BCDataset(validation_data_path)
    
    # generator = torch.Generator().manual_seed(42)
    # train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_len, test_len], generator=generator)
    # assert(len(train_dataset)==train_len)
    # assert(len(test_dataset)==test_len)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=True)

    
    print("training data size: ", len(train_dataset))
    print("test data size: ", len(test_dataset))
    print("training data path:", training_data_path)

    epochs = 201
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.0015) #0.00067
    train_losses = []
    test_L2_losses = []
    test_log_prob_losses = []
    #scheduler = torch.nn.optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.1)
    for epoch in range(epochs):
        print("============================")
        print(f"Epoch {epoch+1}\n-------------------------------")
        epoch_loss = train(policy, train_loader, optimizer, device)
        train_losses.append(epoch_loss)
        test_L2_loss, test_log_prob_loss = test(policy, test_loader, device)
        test_L2_losses.append(test_L2_loss)
        test_log_prob_losses.append(test_log_prob_loss)
        if epoch % 2 == 0:        
            torch.save(policy.state_dict(), os.path.join(model_path, "epoch " + str(epoch)))
    
    print("Done!")

    #plot_learning_curve(train_losses, "train losses of actor trained by BC with 14200 state-actiion pair")
    plot_train_test_curves(train_losses, test_log_prob_losses,"train test neg log prob losses")
    plot_learning_curve(test_L2_losses,"actor test L2 losses")
    print("std", policy.log_std.exp())