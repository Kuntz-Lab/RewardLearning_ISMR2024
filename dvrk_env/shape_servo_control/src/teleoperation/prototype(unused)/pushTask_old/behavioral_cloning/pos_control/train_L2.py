import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import sys
#sys.path.append("../")
from torch.utils.data import DataLoader
from dataset_loader import BCDataset
from policy_BC import BCPolicy
import argparse
import matplotlib.pyplot as plt
import os


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
        pred_actions = model(states)
        loss = torch.sum((pred_actions - actions)**2)
       
        optimizer.zero_grad()
        loss.backward()
        train_loss += loss.item()/len(states)
        optimizer.step()
        if batch_idx % 10 == 0:
            loss, current = loss.item(), (batch_idx + 1) * len(states)
            print(f"train L2 loss: {loss:>7f}  average train L2 loss: {train_loss/num_batch} progress: [{current:>5d}/{size:>5d}]")

    avg_train_loss = train_loss/num_batch
    return avg_train_loss

def test(model, dataloader, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for sample in dataloader:
            states, actions = sample
            states = states.float().to(device)
            actions = actions.float().to(device)
            pred_actions = model(states)
            loss = torch.sum((pred_actions - actions)**2)
            test_loss += loss.item()/len(states)
    test_loss /= (num_batches)
    print(f"Test Error: \n Avg test L2 loss: {test_loss:>8f} \n")
    print("============================")
    return test_loss


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

    parser.add_argument('--mp', default='/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_push_BC_corrected/pos_control/weights/weights_L2', type=str, help="path to model")
    parser.add_argument('--tdp', default='/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_push_BC_corrected/pos_control/data_processed_train', type=str, help="path to training data")
    parser.add_argument('--vdp', default='/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_push_BC_corrected/pos_control/data_processed_test', type=str, help="path to validation data")


    args = parser.parse_args()
    model_path = args.mp
    training_data_path = args.tdp
    validation_data_path = args.vdp


    os.makedirs(args.mp, exist_ok=True)
    print("total size of dataset: ", len(os.listdir(args.tdp)))

    # Now we create a reward network and optimize it using the training data.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    policy =  BCPolicy(obs_dim=9, act_dim=3)
    policy.to(device)

    # train_len = 14200#28000
    # # test_len = 2840
    # # total_len = train_len + test_len

    # dataset = BCDataset(training_data_path)
    # train_dataset = torch.utils.data.Subset(dataset, range(0, train_len))
    # test_dataset = torch.utils.data.Subset(dataset, range(train_len, total_len))
    train_dataset = BCDataset(training_data_path)
    test_dataset = BCDataset(validation_data_path)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=True)

    
    print("training data size: ", len(train_dataset))
    print("test data size: ", len(test_dataset))
    print("training data path:", training_data_path)

    epochs = 201
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.00015) #best: 0.00015
    train_losses = []
    test_losses = []
    #scheduler = torch.nn.optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.1)
    for epoch in range(epochs):
        print("============================")
        print(f"Epoch {epoch+1}\n-------------------------------")
        epoch_loss = train(policy, train_loader, optimizer, device)
        train_losses.append(epoch_loss)
        test_loss = test(policy, test_loader, device)
        test_losses.append(test_loss)
        if epoch % 2 == 0:        
            torch.save(policy.state_dict(), os.path.join(model_path, "epoch " + str(epoch)))
    
    print("Done!")

    plot_train_test_curves(train_losses, test_losses, "train test losses of L2 BC")