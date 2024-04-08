import torch
import torch.nn as nn
import torch.nn.functional as F
#import sys
#sys.path.append("../")
import numpy as np

import os
from reward import RewardNetPointCloudEEF as RewardNet
from dataset_loader import TrajDataset
import pickle
import argparse


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
            eef_traj_i = eef_traj_i.float().to(device)
            eef_traj_j= eef_traj_j.float().to(device)
            label = torch.from_numpy(np.array(label)).long().to(device)

            outputs, abs_rewards = model(eef_traj_i, obj_traj_i, eef_traj_j, obj_traj_j)
            # outputs = outputs.unsqueeze(0)

            test_loss += loss_fn(outputs, label).item()
            
            correct += (outputs.argmax(1) == label).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    print("============================")




if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description='Options')

    parser.add_argument('--rmp', default='/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_trace_curve_corrected/weights_straight_flat_2ball/weights_1', type=str, help="path to reward model")
    parser.add_argument('--tdp', default='/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_trace_curve_corrected/processed_data_train_straight_flat_2ball', type=str, help="path to testing data")
    parser.add_argument('--test_len', default=14000, type=int, help="size of the desired testing data subset")
    
    args = parser.parse_args()

    reward_model_path = args.rmp
    dataset_path = args.tdp
    test_len = args.test_len

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    reward_net = RewardNet()
    reward_net.to(device)
    reward_net.load_state_dict(torch.load(os.path.join(reward_model_path, "epoch_200")))

    
    #test_len = 14000
    dataset = TrajDataset(dataset_path)
    test_dataset = torch.utils.data.Subset(dataset, range(0, test_len))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=True)

    print("test data: ", len(test_dataset))
    print("data path:", dataset.dataset_path)


    loss_fn = nn.CrossEntropyLoss()
    # for sample in test_loader:
    #     #print(sample)
    #     print(sample[0].shape, sample[1].shape, sample[2].shape, sample[3].shape, sample[4])

    test(test_loader, reward_net, loss_fn, device)
    
    print("Done!")