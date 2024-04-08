import torch
import torch.nn as nn
import numpy as np

import torch.optim as optim
import torch.nn.functional as F
import os
from reward_model import RewardNetPointCloud as RewardNet
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
    # training_data = list(zip(training_inputs, training_outputs))
    
    num_data_pt = len(os.listdir(training_data_path)) #5000

    for epoch in range(0, num_iter):
        print("================================================")
        # np.random.shuffle(training_data)
        # training_obs, training_labels = zip(*training_data)
        # for i in range(len(training_labels)):
        
        shuffled_idxs = np.arange(num_data_pt)
        np.random.shuffle(shuffled_idxs)
        # print("shuffled_idxs:", shuffled_idxs)
        for i in range(num_data_pt):
            
            k = shuffled_idxs[i]
            filename = os.path.join(training_data_path, "processed sample " + str(k) + ".pickle")
            with open(filename, 'rb') as handle:
                data = pickle.load(handle)

            

            # traj_i, traj_j = data["traj_1"], data["traj_2"]
            traj_i, traj_j = data["traj_1"][:,:3], data["traj_2"][:,:3]
            obj_emb = data["obj_embedding"][0] #Change [0]

            labels = data["label"]
            # traj_i = np.array(traj_i)
            # traj_j = np.array(traj_j)
            traj_i = torch.from_numpy(traj_i).float().to(device)
            traj_j = torch.from_numpy(traj_j).float().to(device)

            obj_emb = torch.from_numpy(obj_emb).unsqueeze(0).float().to(device)
            obj_emb_i = obj_emb.repeat(traj_i.shape[0],1)
            obj_emb_j = obj_emb.repeat(traj_j.shape[0],1)
            # print("shapes:", traj_i.shape, traj_j.shape, obj_emb_i.shape, obj_emb_j.shape)

            labels = torch.from_numpy(np.array([labels])).long().to(device)
            # print(labels)
            # print("++++++++++++++++++++++++++++++++++++++++++++++++")
            # print("traj_i.shape, traj_j.shape, labels", traj_i.shape, traj_j.shape, labels)

            #zero out gradient
            optimizer.zero_grad()

            #forward + backward + optimize
            outputs, abs_rewards = reward_network.forward(traj_i, traj_j, obj_emb_i, obj_emb_j)
            outputs = outputs.unsqueeze(0)
            loss = loss_criterion(outputs, labels) + l1_reg * abs_rewards
            loss.backward()
            optimizer.step()

            #print stats to see if learning
            item_loss = loss.item()
            cum_loss += item_loss
            if i % 1000 == 0:
                #print(i)
                print("epoch {}:{} loss {}".format(epoch,i, cum_loss))
                print("abs_rewards:", abs_rewards)
                cum_loss = 0.0
                print("check pointing")
        
        if epoch % 2 == 0:        
            torch.save(reward_net.state_dict(), os.path.join(checkpoint_dir, "epoch " + str(epoch)))
   
    print("finished training")

# parse arguments
parser = argparse.ArgumentParser(description='Options')

parser.add_argument('--rmp', default="/home/dvrk/LfD_data/group_meeting/weights/weights_demos_w_embedding", type=str, help="path to reward model")
parser.add_argument('--tdp', default="/home/dvrk/LfD_data/group_meeting/processed_demos_w_embedding", type=str, help="path to training data")

# parser.add_argument('--rmp', default='/home/baothach/shape_servo_data/teleoperation/sanity_check_examples/ex_3/weights/weights_full_pc', type=str, help="path to reward model")
# parser.add_argument('--tdp', default='/home/baothach/shape_servo_data/teleoperation/sanity_check_examples/ex_3/processed_data_2', type=str, help="path to training data")

args = parser.parse_args()

os.makedirs(args.rmp, exist_ok=True)
reward_model_path = args.rmp
training_data_path = args.tdp

# Now we create a reward network and optimize it using the training data.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
reward_net = RewardNet()
reward_net.to(device)

# reward_net.load_state_dict(torch.load(os.path.join(reward_model_path, "epoch 34")))

### Pretrained model from sim
# pretrain_path = '/home/baothach/shape_servo_data/teleoperation/sanity_check_examples/ex_3/weights/weights_single_ball_partial_w_embedding_1'
reward_net.load_state_dict(torch.load(os.path.join(reward_model_path, "epoch_6_sim")))

lr = 0.00001
weight_decay = 0.0
num_iter = 101 #num times through training data
l1_reg = 0.0


optimizer = optim.Adam(reward_net.parameters(),  lr=lr, weight_decay=weight_decay)
learn_reward(reward_net, optimizer, training_data_path, num_iter, l1_reg, reward_model_path)
#save reward network
# torch.save(reward_net.state_dict(), reward_model_path)