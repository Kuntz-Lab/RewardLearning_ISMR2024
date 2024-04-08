import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
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
        actions = actions.float().to(device)
        action_means = model(states)
        covariance = torch.diag(model.log_std.exp() * model.log_std.exp())
        dist = MultivariateNormal(action_means, scale_tril=covariance)
        loss = -dist.log_prob(actions).sum()
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
            test_L2_loss += L2_loss.item()
            # log prob loss
            action_means = model(states)
            covariance = torch.diag(model.log_std.exp() * model.log_std.exp())
            dist = MultivariateNormal(action_means, scale_tril=covariance)
            log_prob_loss = -dist.log_prob(actions).sum()/len(actions)
            test_log_prob_loss += log_prob_loss.item()/len(actions)
    test_L2_loss /= num_batches
    test_log_prob_loss /= num_batches
    print(f"Test Error: \n Avg test L2 loss: {test_L2_loss:>8f} \n  Avg test log prob loss: {test_log_prob_loss:>8f}")
    print("============================")
    return test_L2_loss, test_log_prob_loss


def plot_curve(losses, title):
    epochs = [i for i in range(len(losses))]
    fig, ax = plt.subplots()
    ax.plot(epochs, losses)
    ax.set_title(title)
    ax.set_xlabel("epochs")
    ax.set_ylabel("losses")
    os.makedirs(f"./figures", exist_ok=True)
    fig.savefig(f"./figures/{title}.png")

def plot_curves(xs, ys_1, ys_2, x_label="epochs", y_label="losses", label_1="train_losses", label_2="test_losses", title="train test losses 2 ball", path="./figures"):
    fig, ax = plt.subplots()
    ax.plot(xs, ys_1, label=label_1)
    ax.plot(xs, ys_2, label=label_2)
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    os.makedirs(path, exist_ok=True)
    fig.savefig(f"{path}/{title}.png")
    plt.cla()
    plt.close(fig)    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Options')

    parser.add_argument('--mp', type=str, help="path to model")
    parser.add_argument('--tdp', type=str, help="path to training data")
    parser.add_argument('--vdp', type=str, help="path to validation data")

    args = parser.parse_args()
    model_path = args.mp
    training_data_path = args.tdp
    validation_data_path = args.vdp


    os.makedirs(args.mp, exist_ok=True)
    print("total size of dataset: ", len(os.listdir(args.tdp)))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    policy = ActorPC(act_dim=3, robot_state_dim=3, emb_dim=256, initial_std=1.0).to(device)
    policy.to(device)

    train_dataset = BCDataset(training_data_path)
    test_dataset = BCDataset(validation_data_path)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=5000, shuffle=True) #batchsize 1000
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=5000, shuffle=True)
    
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
        if epoch % 10 == 0:        
            torch.save(policy.state_dict(), os.path.join(model_path, "epoch_" + str(epoch)))
    
    print("Done!")

    xs = [i for i in range(epochs)]
    plot_curves(xs, train_losses, test_log_prob_losses, x_label="epochs", y_label="losses", label_1="train_losses", label_2="test_losses", title="train test neg log prob loss", path="./figures")
    plot_curve(test_L2_losses,"actor test L2 losses")
    plot_curve(test_log_prob_losses,"actor test neg log prob losses")
    print("std", policy.log_std.exp())