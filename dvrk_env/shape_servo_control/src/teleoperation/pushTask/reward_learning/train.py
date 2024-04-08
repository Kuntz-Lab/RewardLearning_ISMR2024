import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

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
    correct = 0
    num_batch = 0
    for batch_idx, sample in enumerate(dataloader):
        num_batch+=1
        obj_traj_i, obj_traj_j, eef_traj_i, eef_traj_j, label  = sample

        obj_traj_i = obj_traj_i.float().to(device)
        obj_traj_j= obj_traj_j.float().to(device)
        # eef_traj_i = eef_traj_i.float().to(device)
        # eef_traj_j= eef_traj_j.float().to(device)
        label = torch.from_numpy(np.array(label)).long().to(device)

        #zero out gradient
        optimizer.zero_grad()

        # compute loss and backward
        outputs, abs_rewards = model(obj_traj_i, obj_traj_j)

        correct += (outputs.argmax(1) == label).type(torch.float).sum().item()

        loss = loss_fn(outputs, label) #+ l1_reg * abs_rewards
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        if batch_idx % 10 == 0:
            loss, current = loss.item(), (batch_idx + 1) * len(obj_traj_i)
            print(f"train loss: {loss:>7f}  average train loss: {train_loss/num_batch} progress: [{current:>5d}/{size:>5d}]")

    avg_train_loss = train_loss/num_batch
    correct/= size
    print(f"Train Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {avg_train_loss:>8f} \n")
    return avg_train_loss, correct

            
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

    return test_loss, correct

def plot_curves(xs, ys_1, ys_2, x_label="epochs", y_label="losses", label_1="train_losses", label_2="test_losses", title="train test losses 2 ball bonus", path="./figures"):
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
    torch.manual_seed(2021)

    # parse arguments
    parser = argparse.ArgumentParser(description='Options')

    parser.add_argument('--rmp', default='/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_push/weights/weights_1', type=str, help="path to reward model")
    parser.add_argument('--tdp', default='/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_push/data_processed_train', type=str, help="path to training data")
    parser.add_argument('--vdp', default='/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_push/data_processed_test', type=str, help="path to validation data")

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
    print("reward model path:", reward_model_path)
    print("training data path:", training_data_path)

    epochs = 101
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(reward_net.parameters(), lr=0.00015)
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    #scheduler = torch.nn.optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.1)
    for epoch in range(epochs):
        print("============================")
        print(f"Epoch {epoch+1}\n-------------------------------")
        train_loss, train_acc = train(train_loader, reward_net, loss_fn, optimizer, device)
        test_loss, test_acc = test(test_loader, reward_net, loss_fn, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        if epoch % 2 == 0:        
            torch.save(reward_net.state_dict(), os.path.join(reward_model_path, "epoch_" + str(epoch)))

    xs = [i for i in range(epochs)]
    plot_curves(xs, train_losses, test_losses, x_label="epochs", y_label="losses", label_1="train_losses", label_2="test_losses", title="train test losses box cone", path="./figures")
    plot_curves(xs, train_accs, test_accs, x_label="epochs", y_label="accuracies", label_1="train_accuracies", label_2="test_accuracies", title="train test accuracies box cone", path="./figures")
    print("Done!")