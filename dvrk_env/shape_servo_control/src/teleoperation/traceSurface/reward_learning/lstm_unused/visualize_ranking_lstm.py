import pickle


import os
import numpy as np
import timeit
import roslib.packages as rp
import sys
sys.path.append("./pointcloud_representation_learning")
pkg_path = rp.get_pkg_dir('shape_servo_control')
sys.path.append(pkg_path + '/src')

#import open3d
from curve import *
#from reward_model import RewardNetPointCloud as RewardNet
#from utils import *
from reward_lstm import RewardLSTM as RewardNet
import matplotlib.pyplot as plt
import torch



def get_random_data(data_recording_path, num_processed_samples):
    '''
    '''
    idx = np.random.randint(low=0, high= num_processed_samples)
    file = os.path.join(data_recording_path, f"processed sample {idx}.pickle")
    with open(file, 'rb') as handle:
        data = pickle.load(handle)
    
    return idx, data

def success_stat(data_path):
    files = os.listdir(data_path)
    stat = {i:0 for i in range(11)}
    for i, file in enumerate(files):
        if i%1000==0:
            print("now at sample ", i)
        with open(f"{data_path}/{file}", 'rb') as handle:
            data = pickle.load(handle)
        count = data["num_balls_reached"]
        stat[count] = stat[count]+1
    print(stat)



def show_learned_preference(reward_model, data):
    ################ predicted vs ground truth ##########################
    obj_traj_i, obj_traj_j = data["emb_traj_1"], data["emb_traj_2"]
    eef_traj_i, eef_traj_j = data["eef_traj_1"], data["eef_traj_2"]
    label = data["label"]

    obj_traj_i = obj_traj_i.unsqueeze(0).float().to(device)
    obj_traj_j= obj_traj_j.unsqueeze(0).float().to(device)
    eef_traj_i = eef_traj_i.unsqueeze(0).float().to(device)
    eef_traj_j= eef_traj_j.unsqueeze(0).float().to(device)
    label = torch.from_numpy(np.array(label)).long().to(device)

    with torch.no_grad():
        r_i, _ = reward_model(eef_traj_i, obj_traj_i)
        r_j, _ = reward_model(eef_traj_j, obj_traj_j)
        outputs = torch.cat((r_i, r_j),dim=1)
        print("==========================")
        print("out shape:", outputs.shape)
        print("out: ", outputs)
        print("predict == gt ? : ", outputs.argmax(1) == label)
        print("gt:", label)
        print("argmax:", outputs.argmax(1))
        print("===========================")
        pred = (outputs.argmax(1) == label).type(torch.float).item()

    ################### plot ########################################
    which = ["red better", "green better"]

    traj = [data["eef_traj_1"].cpu().detach().numpy(), data["eef_traj_2"].cpu().detach().numpy()]

    with torch.no_grad():
        traj_i_reward, _ = reward_model.single_return(eef_traj_i, obj_traj_i)
        traj_j_reward, _ = reward_model.single_return(eef_traj_j, obj_traj_j)
        
    traj_i_reward = traj_i_reward.squeeze(0)
    traj_j_reward = traj_j_reward.squeeze(0)
    max_reward_i = torch.max(traj_i_reward).item()
    max_reward_j = torch.max(traj_j_reward).item()
    max_reward = max(max_reward_i, max_reward_j)
    min_reward_i = torch.min(traj_i_reward).item()
    min_reward_j = torch.min(traj_j_reward).item()
    min_reward = min(min_reward_i, min_reward_j)

    balls_xyz = data["balls_xyz"]

    for ball_pose in balls_xyz:
        plt.plot(ball_pose[0], ball_pose[1], "o", markersize=20)

    xs = [traj[0][t,0] for t in range(len(traj[0]))]
    ys = [traj[0][t,1] for t in range(len(traj[0]))]
    heats = [[(traj_i_reward[t].item() - min_reward) / (max_reward - min_reward), 0, 0] for t in range(len(traj[0]))]
    plt.scatter(xs, ys, c=heats) 
    plt.plot(xs, ys) 

    xs = [traj[1][t,0] for t in range(len(traj[1]))]
    ys = [traj[1][t,1] for t in range(len(traj[1]))]
    heats = [[0, (traj_j_reward[t].item() - min_reward) / (max_reward - min_reward), 0] for t in range(len(traj[1]))]
    plt.scatter(xs, ys, c=heats) 
    plt.plot(xs, ys) 

    plt.title(f"GT: {which[int(label)]} model: {which[int(outputs.argmax(1))]}")
    plt.xlabel("x", fontsize=15)
    plt.ylabel("y", fontsize=15)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.show()


def show_learned_preference_3D(reward_model, data):
    ################ predicted vs ground truth ##########################
    obj_traj_i, obj_traj_j = data["emb_traj_1"], data["emb_traj_2"]
    eef_traj_i, eef_traj_j = data["eef_traj_1"], data["eef_traj_2"]
    label = data["label"]

    obj_traj_i = obj_traj_i.unsqueeze(0).float().to(device)
    obj_traj_j= obj_traj_j.unsqueeze(0).float().to(device)
    eef_traj_i = eef_traj_i.unsqueeze(0).float().to(device)
    eef_traj_j= eef_traj_j.unsqueeze(0).float().to(device)
    label = torch.from_numpy(np.array(label)).long().to(device)

    with torch.no_grad():
        r_i, _ = reward_model(eef_traj_i, obj_traj_i)
        r_j, _ = reward_model(eef_traj_j, obj_traj_j)
        outputs = torch.cat((r_i, r_j),dim=1)
        print("==========================")
        print("out shape:", outputs.shape)
        print("out: ", outputs)
        print("predict == gt ? : ", outputs.argmax(1) == label)
        print("gt:", label)
        print("argmax:", outputs.argmax(1))
        print("===========================")
        pred = (outputs.argmax(1) == label).type(torch.float).item()

    ################### plot ########################################
    which = ["red better", "green better"]

    traj = [data["eef_traj_1"].cpu().detach().numpy(), data["eef_traj_2"].cpu().detach().numpy()]

    with torch.no_grad():
        traj_i_reward, _ = reward_model.single_return(eef_traj_i, obj_traj_i)
        traj_j_reward, _ = reward_model.single_return(eef_traj_j, obj_traj_j)
        
    traj_i_reward = traj_i_reward.squeeze(0)
    traj_j_reward = traj_j_reward.squeeze(0)
    max_reward_i = torch.max(traj_i_reward).item()
    max_reward_j = torch.max(traj_j_reward).item()
    max_reward = max(max_reward_i, max_reward_j)
    min_reward_i = torch.min(traj_i_reward).item()
    min_reward_j = torch.min(traj_j_reward).item()
    min_reward = min(min_reward_i, min_reward_j)

    balls_xyz = data["balls_xyz"]

    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")

    for ball_pose in balls_xyz:
        ax.plot(ball_pose[0], ball_pose[1], ball_pose[2], "o", markersize=20)

    xs = [traj[0][t,0] for t in range(len(traj[0]))]
    ys = [traj[0][t,1] for t in range(len(traj[0]))]
    zs = [traj[0][t,2] for t in range(len(traj[0]))]
    heats = [[(traj_i_reward[t].item() - min_reward) / (max_reward - min_reward), 0, 0] for t in range(len(traj[0]))]
    ax.scatter(xs, ys, zs, c=heats, s=64) 
    ax.plot(xs, ys, zs) 

    xs = [traj[1][t,0] for t in range(len(traj[1]))]
    ys = [traj[1][t,1] for t in range(len(traj[1]))]
    zs = [traj[0][t,2] for t in range(len(traj[1]))]
    heats = [[0, (traj_j_reward[t].item() - min_reward) / (max_reward - min_reward), 0] for t in range(len(traj[1]))]
    ax.scatter(xs, ys, zs, c=heats, s=64) 
    ax.plot(xs, ys, zs) 

    plt.title(f"GT: {which[int(label)]} model: {which[int(outputs.argmax(1))]}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    # plt.xticks(fontsize=8)
    # plt.yticks(fontsize=8)
    plt.show()



if __name__ == "__main__":
    ### CHANGE ####
    is_train = True
    suffix = "train" if is_train else "test"

    data_recording_path = f"/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_trace_curve_corrected/processed_data_{suffix}_straight_3D_flat_lstm_vsimple_2"
    num_data_pt = 12000 #Change

    start_time = timeit.default_timer()

    success_stat(f"/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_trace_curve_corrected/demos_{suffix}_straight_3D_flat_vsimple_2") 

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    reward_net = RewardNet(input_dim=(256, 3), embedding_dim=128,  hidden_dim=128, output_dim=1,  n_layers=1, drop_prob=0).to(device)
    reward_net.to(device)

    reward_model_path = '/home/dvrk/shape_servo_data/teleoperation/sanity_check_examples/ex_trace_curve_corrected/weights_straight_3D_flat_lstm_vsimple_2/weights_1'
    reward_net.load_state_dict(torch.load(os.path.join(reward_model_path, "epoch 82")))


    for i in range(num_data_pt):

        if i % 50 == 0:
            print("========================================")
            print("current count:", i, " , time passed:", timeit.default_timer() - start_time)
        
        idx, data = get_random_data(data_recording_path, num_data_pt)  

        show_learned_preference_3D(reward_net, data)




    