import torch
import torch.nn as nn
import torch.nn.functional as F
#import sys
#sys.path.append("../")
import numpy as np

import os
from reward_lstm import RewardLSTM as RewardNet
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

            r_i, _ = model(eef_traj_i, obj_traj_i)
            r_j, _ = model(eef_traj_j, obj_traj_j)
            outputs = torch.cat((r_i, r_j),dim=1)
            test_loss += loss_fn(outputs, label).item()
            correct += (outputs.argmax(1) == label).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    print("============================")




if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description='Options')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    reward_net = RewardNet(input_dim=(256, 3), embedding_dim=128,  hidden_dim=128, output_dim=1,  n_layers=1, drop_prob=0).to(device)
    reward_net.to(device)
    reward_model_path = '/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_trace_curve_corrected/weights_straight_3D_flat_2ball_lstm/weights_1'
    reward_net.load_state_dict(torch.load(os.path.join(reward_model_path, "epoch 100")))

    dataset_path = '/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_trace_curve_corrected/processed_data_train_straight_3D_flat_2ball_lstm'
    test_dataset = TrajDataset(dataset_path)
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=True)

    
    print("test data: ", len(test_dataset))
    print("test data path:", test_dataset.dataset_path)


    loss_fn = nn.CrossEntropyLoss()
    
    test(test_loader, reward_net, loss_fn, device)
    
    print("Done!")