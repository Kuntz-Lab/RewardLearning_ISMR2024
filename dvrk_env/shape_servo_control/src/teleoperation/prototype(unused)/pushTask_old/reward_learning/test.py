import os
import numpy as np
import pickle
import timeit
import torch
import sys
#sys.path.append("./pointcloud_representation_learning")
#from reward_model import RewardNetPointCloud as RewardNet
#from architecture import AutoEncoder
#from utils import *
from reward import RewardNetPointCloud as RewardNet
#import matplotlib.pyplot as plt


def compute_predicted_reward(traj, reward_net, device): 
    traj = traj.float().to(device).unsqueeze(0)
    reward = reward_net.cum_return(traj)[0]
    #print(reward.shape)
    #print(reward.cpu().detach().numpy())
    return reward.cpu().detach().numpy() 


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    reward_net = RewardNet()
    reward_net.to(device)
    reward_net.eval()

    # encoder =  AutoEncoder(num_points=256, embedding_size=256).to(device)
    # encoder.load_state_dict(torch.load("/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_AE_boxConeCloser/weights/weights_1/epoch 140"))

    reward_model_path = '/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_push_corrected/weights/weights_no_eef'
    reward_net.load_state_dict(torch.load(os.path.join(reward_model_path, "epoch 100"))) #epoch 60

    test_dir_path = "/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_push_corrected/data_processed_train"
    processed_sample_files = os.listdir(test_dir_path)
    num_data = 1000

    print("testing ", reward_model_path)

    accuracy = 0
    for i in range(num_data):
        test_data_path = os.path.join(test_dir_path, processed_sample_files[i])
        with open(test_data_path, 'rb') as handle:
            data = pickle.load(handle)

        emb_traj_1 = data["emb_traj_1"]
        emb_traj_2 = data["emb_traj_2"]
        label = data["label"]

        with torch.no_grad():
            rew_1 = compute_predicted_reward(emb_traj_1, reward_net, device)
            rew_2 = compute_predicted_reward(emb_traj_2, reward_net, device)
            # print(rew_1)
            if rew_1 >= rew_2 and label==False:
                accuracy += 1
            elif rew_1 <= rew_2 and label==True:
                accuracy += 1
            
            if i%500==0:
                print(f"finished sample {i}")

    accuracy = accuracy/num_data

    print(f"accuracy: {accuracy}\ttest set size: {num_data}")


        