import os
import numpy as np
import pickle
import torch
from reward import RewardNetPointCloud as RewardNet
import argparse
import random



def compute_predicted_reward(traj, reward_net, device): 
    traj = traj.float().to(device).unsqueeze(0)
    reward = reward_net.cum_return(traj)[0]
    return reward.cpu().detach().numpy() 


if __name__ == "__main__":
    random.seed(2021)
    # parse arguments
    parser = argparse.ArgumentParser(description='Options')

    parser.add_argument('--rmp', default='/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_push/weights/weights_1', type=str, help="path to reward model")
    parser.add_argument('--tdp', default='/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_push/data_processed_test', type=str, help="path to testing data")
    parser.add_argument('--test_len', default=1000, type=int, help="size of the desired testing data subset")
    
    args = parser.parse_args()

    reward_model_path = args.rmp
    test_dir_path = args.tdp
    test_len = args.test_len

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    reward_net = RewardNet()
    reward_net.to(device)
    reward_net.eval()

    reward_net.load_state_dict(torch.load(os.path.join(reward_model_path, "epoch_100")))

    processed_sample_files = os.listdir(test_dir_path)
    random.shuffle(processed_sample_files)
    
    print("test data path: ", test_dir_path)
    print("testing reward", reward_model_path)

    accuracy = 0
    for i in range(test_len):
        test_data_path = os.path.join(test_dir_path, processed_sample_files[i])
        with open(test_data_path, 'rb') as handle:
            data = pickle.load(handle)

        emb_traj_1 = data["emb_traj_1"]
        emb_traj_2 = data["emb_traj_2"]
        label = data["label"]

        with torch.no_grad():
            rew_1 = compute_predicted_reward(emb_traj_1, reward_net, device)
            rew_2 = compute_predicted_reward(emb_traj_2, reward_net, device)
            
            if rew_1 >= rew_2 and label==False:
                accuracy += 1
            elif rew_1 <= rew_2 and label==True:
                accuracy += 1
            
            if i%500==0:
                print(f"finished sample {i}")

    accuracy = accuracy/test_len
    print(f"accuracy: {accuracy}\ttest set size: {test_len}")


        