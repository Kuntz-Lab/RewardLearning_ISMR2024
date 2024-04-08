import torch
import torch.nn as nn
import numpy as np

import torch.optim as optim
import torch.nn.functional as F
import os
from policy_BC import BCPolicy
import pickle
import argparse

def train(policy, optimizer, training_data_path, num_iter, model_path):
    #check if gpu available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assume that we are on a CUDA machine, then this should print a CUDA device:
    # print(device)
    
    cum_loss = 0.0
    
    training_data_files = os.listdir(training_data_path)
    num_data = len(os.listdir(training_data_path))//10
    indices = np.arange(num_data)
    batch_size = 500
    num_batch = num_data//batch_size
    print(f"num_data: {num_data}")
    print(f"num_batch: {num_batch}")
    print(f"num_files: {len(training_data_files)}")

    for epoch in range(num_iter):
        for i in range(num_batch):
            states = []
            actions = []
            for k in range(batch_size):
                np.random.shuffle(indices)
                rand_idx = indices[0]
                file = training_data_files[rand_idx]
                filename = os.path.join(training_data_path, file)
                with open(filename, 'rb') as handle:
                    data = pickle.load(handle)

                # print(type(data["state"]))
                # print(data['state'].shape
                states.append(data["state"].float().to(device))
                actions.append(data["action"].float().to(device))
                #print(data["action"].shape)

            states = torch.cat(states, dim=0)
            actions = torch.cat(actions, dim=0)
            # print(states.shape)
            # print(actions.shape)

            #zero out gradient
            optimizer.zero_grad()

            pred_actions = policy.forward(states)

            loss = torch.sum((pred_actions - actions)**2)#/pred_actions.shape[0]

            loss.backward()
            optimizer.step()

            #print stats to see if learning
            item_loss = loss.item()
            cum_loss += item_loss
            if i % 1 == 0:
                #print(i)
                print("epoch {}:{} Loss {} cumLoss {}".format(epoch, i, item_loss, cum_loss))
                cum_loss = 0.0
                print("check pointing")
        
            if epoch % 2 == 0:        
                torch.save(policy.state_dict(), os.path.join(model_path, "epoch " + str(epoch)))
            
    print("finished training")


def test(policy, test_data_path):
    test_data_files = os.listdir(test_data_path)
    num_traj = len(os.listdir(test_data_path))

    avg_l2_loss = 0
    for i, file in enumerate(test_data_files):
        filename = os.path.join(training_data_path, file)
        with open(filename, 'rb') as handle:
            data = pickle.load(handle)

        states = data["state"].float().to(device)
        actions = data["action"].float().to(device)

        with torch.no_grad():
            pred_actions = policy.forward(states)
            avg_l2_loss += torch.sum((pred_actions - actions)**2)

    avg_l2_loss /= (pred_actions.shape[0]*num_traj)

    print(f"avg l2 loss: {avg_l2_loss}\ttest set size: {num_traj*pred_actions.shape[0]}")

    return avg_l2_loss



parser = argparse.ArgumentParser(description='Options')

parser.add_argument('--mp', default='/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_push_BC/weights/weights_2', type=str, help="path to model")
parser.add_argument('--tdp', default='/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_push_BC/data_processed_train_2', type=str, help="path to training data")

args = parser.parse_args()

os.makedirs(args.mp, exist_ok=True)
print("num_training_data: ", len(os.listdir(args.tdp)))

# Now we create a reward network and optimize it using the training data.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
policy = BCPolicy(obs_dim=9, act_dim=10)
policy.to(device)
model_path = args.mp
training_data_path = args.tdp

lr = 0.00015
weight_decay = 0.0
num_iter = 101 #num times through training data
l2_reg = 0.0


optimizer = optim.Adam(policy.parameters(),  lr=lr, weight_decay=weight_decay)
train(policy, optimizer, training_data_path, num_iter, model_path)