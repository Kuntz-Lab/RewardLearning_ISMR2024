import torch
import torch.nn as nn
import numpy as np

import torch.optim as optim
import torch.nn.functional as F
import os
from reward import RewardNet2 as RewardNet
import pickle
import argparse

# Train the network
def learn_reward(reward_network, optimizer, training_data_path, num_iter, l1_reg, checkpoint_dir):
    #check if gpu available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assume that we are on a CUDA machine, then this should print a CUDA device:
    # print(device)
    loss_criterion = nn.CrossEntropyLoss()
    
    cum_loss = 0.0
    
    num_data_pt = len(os.listdir(training_data_path))

    for epoch in range(num_iter):
        shuffled_idxs = np.arange(num_data_pt)
        np.random.shuffle(shuffled_idxs)
        for i in range(num_data_pt):
            k = shuffled_idxs[i]
            filename = os.path.join(training_data_path, "processed sample " + str(k) + ".pickle")
            with open(filename, 'rb') as handle:
                data = pickle.load(handle)

            
            traj_i, traj_j = data["traj_1"], data["traj_2"]
            label = data["label"]
            traj_i = torch.from_numpy(traj_i).float().to(device)
            traj_j = torch.from_numpy(traj_j).float().to(device)
            label = torch.from_numpy(np.array([label])).long().to(device)

            #zero out gradient
            optimizer.zero_grad()

            #forward + backward + optimize
            outputs, abs_rewards = reward_network.forward(traj_i, traj_j)
            outputs = outputs.unsqueeze(0)
            loss = loss_criterion(outputs, label) + l1_reg * abs_rewards
            loss.backward()
            optimizer.step()

            #print stats to see if learning
            item_loss = loss.item()
            cum_loss += item_loss
            if i % 100 == 0:
                #print(i)
                print("epoch {}:{} Loss {} cumLoss {}".format(epoch,i, item_loss, cum_loss))
                print("abs_rewards:", abs_rewards)
                cum_loss = 0.0
                print("check pointing")
        
        if epoch % 2 == 0:        
            torch.save(reward_net.state_dict(), os.path.join(checkpoint_dir, "epoch " + str(epoch)))
   
    print("finished training")

# parse arguments
parser = argparse.ArgumentParser(description='Options')

parser.add_argument('--rmp', default='/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_indicator_0and1/weights/weights_1_sub0', type=str, help="path to reward model")
parser.add_argument('--tdp', default='/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_indicator_0and1/processed_data_train_1_sub0', type=str, help="path to training data")



args = parser.parse_args()

os.makedirs(args.rmp, exist_ok=True)
print("num_training_data: ", len(os.listdir(args.tdp)))

# Now we create a reward network and optimize it using the training data.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
reward_net = RewardNet()
reward_net.to(device)
reward_model_path = args.rmp
training_data_path = args.tdp

lr = 0.00015
weight_decay = 0.0
num_iter = 101 #num times through training data
l1_reg = 0.0


optimizer = optim.Adam(reward_net.parameters(),  lr=lr, weight_decay=weight_decay)
learn_reward(reward_net, optimizer, training_data_path, num_iter, l1_reg, reward_model_path)
#save reward network
# torch.save(reward_net.state_dict(), reward_model_path)