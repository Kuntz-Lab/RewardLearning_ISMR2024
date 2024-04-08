import pickle
import os
import numpy as np
import timeit
import sys
import argparse

from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym import gymutil

from reward import RewardNetPointCloudEEF as RewardNet
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

def get_data(data_recording_path, idx):
    '''
    '''
    file = os.path.join(data_recording_path, f"processed sample {idx}.pickle")
    with open(file, 'rb') as handle:
        data = pickle.load(handle)
    
    return idx, data

def success_stat(data_path, max_num_balls=2):
    files = os.listdir(data_path)
    stat = {i:0 for i in range(max_num_balls+1)}
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
        outputs, abs_rewards = reward_model(eef_traj_i, obj_traj_i, eef_traj_j, obj_traj_j)
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
        traj_i_reward = reward_model.single_return(eef_traj_i, obj_traj_i).squeeze(0)
        traj_j_reward = reward_model.single_return(eef_traj_j, obj_traj_j).squeeze(0)
        
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

def show_learned_preference_3D(reward_model, data, only_failure=False):
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
        outputs, abs_rewards = reward_model(eef_traj_i, obj_traj_i, eef_traj_j, obj_traj_j)
        print("==========================")
        print("out shape:", outputs.shape)
        print("out: ", outputs)
        print("predict == gt ? : ", outputs.argmax(1) == label)
        print("gt:", label)
        print("argmax:", outputs.argmax(1))
        print(f"group: ", data["group"])
        print(f"traj_1_idx: ", data["indices"][0])
        print(f"traj_2_idx: ", data["indices"][1])
        print("===========================")
        pred = (outputs.argmax(1) == label).type(torch.float).item()

    ################### plot ########################################
    if only_failure:
        if outputs.argmax(1) == label or data["num_balls_reached_1"]==data["num_balls_reached_2"]:
            print("skip successful or ambiguous case")
            return

    which = ["red better", "green better"]

    traj = [data["eef_traj_1"].cpu().detach().numpy(), data["eef_traj_2"].cpu().detach().numpy()]

    with torch.no_grad():
        traj_i_reward = reward_model.single_return(eef_traj_i, obj_traj_i).squeeze(0)
        traj_j_reward = reward_model.single_return(eef_traj_j, obj_traj_j).squeeze(0)
        
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
    ax.scatter(xs, ys, zs, c=heats, s=[(i+1)*5 for i in range(len(traj[0]))]) 
    ax.plot(xs, ys, zs, color='red', label=f"num balls reached: "+ str(data["num_balls_reached_1"])) 

    xs = [traj[1][t,0] for t in range(len(traj[1]))]
    ys = [traj[1][t,1] for t in range(len(traj[1]))]
    zs = [traj[1][t,2] for t in range(len(traj[1]))]
    heats = [[0, (traj_j_reward[t].item() - min_reward) / (max_reward - min_reward), 0] for t in range(len(traj[1]))]
    ax.scatter(xs, ys, zs, c=heats, s=[(i+1)*5 for i in range(len(traj[1]))]) 
    ax.plot(xs, ys, zs, color='green', label=f"num balls reached: "+ str(data["num_balls_reached_2"])) 

    ax.legend()
    plt.title(f"GT: {which[int(label)]} model: {which[int(outputs.argmax(1))]}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xticks([-0.1+0.02*i for i in range(11)])
    ax.set_yticks([-0.6+0.02*i for i in range(11)])
    ax.set_zticks([0+0.02*i for i in range(11)])
    plt.show()


if __name__ == "__main__":
    is_train = True
    suffix = "train" if is_train else "test"
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--demo_path', type=str, help="path to demos")
    parser.add_argument('--data_processed_path', type=str, help="path to processed data")
    parser.add_argument('--rmp', help="reward_model_path")
    parser.add_argument('--num_data_pt', default=14000, type=int, help="num data points in the processed data to loop through")
    parser.add_argument('--max_num_balls', default=1, type=int, help="max num ball in the demo")


    args = parser.parse_args()
    num_data_pt = args.num_data_pt
    data_recording_path = args.demo_path
    data_processed_path = args.data_processed_path
    max_num_balls = args.max_num_balls
    reward_model_path = args.rmp

    start_time = timeit.default_timer()

    success_stat(data_recording_path, max_num_balls=max_num_balls) 

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    reward_net = RewardNet()
    reward_net.to(device)
    reward_net.load_state_dict(torch.load(reward_model_path))
    reward_net.eval()

    for i in range(num_data_pt):

        if i % 1 == 0:
            print("========================================")
            print("current count:", i, " , time passed:", timeit.default_timer() - start_time)
        
        #idx, data = get_random_data(data_recording_path, num_data_pt)  
        _, data = get_data(data_processed_path, i)

        ## for only showing the group
        # group = data["group"]
        # if (group!=0):
        #     continue

        show_learned_preference_3D(reward_net, data, only_failure=True)




    