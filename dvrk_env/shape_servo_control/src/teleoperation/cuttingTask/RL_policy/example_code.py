from module import ActorCritic
import torch
import numpy as np
import open3d
import sys
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../pointcloud_representation_learning")
from pc_utils.compute_partial_pc import farthest_point_sample_batched
from cuttingTask.pointcloud_representation_learning.architecture import AutoEncoder
from cuttingTask.reward_learning.fully_connected.reward import RewardNetPointCloudEEF as RewardNet

import os
import pickle
import matplotlib.pyplot as plt
import math

EEF_GROUND_Z_OFFSET = 0.03

def to_obj_emb(model, device, pcds):
    '''
    pcds has shape (num_batch, num_points, point_dim)
    '''
    pcd_tensor = torch.from_numpy(pcds.transpose(0,2,1)).float().to(device)
    with torch.no_grad():
        emb = model(pcd_tensor, get_global_embedding=True)
    return emb

def compute_predicted_reward(eef_pose, embedding, reward_net, device): 
    eef_pose = torch.Tensor(eef_pose).unsqueeze(0).float().to(device)
    reward = reward_net.cum_return(eef_pose.unsqueeze(0), embedding.unsqueeze(0))[0][0]
    # print(reward.cpu().detach().numpy())
    return reward.item()

def compute_predicted_traj_reward(eef_poses, embeddings, reward_net, device): 
    reward = reward_net.cum_return(eef_poses.unsqueeze(0), embeddings.unsqueeze(0))[0][0]
    return reward.item()

def sigmoid_clip(input_tensor, min_value, max_value):
        scaled_tensor = torch.sigmoid(input_tensor)
        clipped_tensor = min_value + (max_value - min_value) * scaled_tensor
        return clipped_tensor

def transform_coordinate(actions):
    actions = torch.clone(actions)
    actions[:, 0:2]*=-1
    # actions[:, 2] -= ROBOT_Z_OFFSET
    return actions

def action_to_robot_delta_xyz(action, current_xyz):
    actions_clipped = torch.clone(action)
    actions_clipped[:, 0] = sigmoid_clip(actions_clipped[:, 0], min_value=-0.1, max_value=0.1)
    actions_clipped[:, 1] = sigmoid_clip(actions_clipped[:, 1], min_value=-0.45, max_value=-0.35)
    actions_clipped[:, 2] = sigmoid_clip(actions_clipped[:, 2], min_value=EEF_GROUND_Z_OFFSET, max_value=0.1)
    #actions_clipped_transformed = transform_coordinate(actions_clipped) 
    delta_xyz = actions_clipped - current_xyz
    return delta_xyz

def plot_curves(xs, ys_list, title, path, x_label, y_label, curve_labels):
    fig, ax = plt.subplots()
    assert(len(curve_labels)==len(ys_list))
    if len(ys_list)==2:
        ax.plot(xs, ys_list[0], label=curve_labels[0])
        ax.plot(xs, ys_list[1], label=curve_labels[1], linestyle='dashed')
    else:
        for idx in range(len(ys_list)):
            ax.plot(xs, ys_list[idx], '-o', label=curve_labels[idx])
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    #ax.set_xticks([i for i in range(len(xs))])
    #ax.set_yticks([i*0.015 for i in range(7)])
    os.makedirs(path, exist_ok=True)
    fig.savefig(f"{path}/{title}.png")
    plt.cla()
    plt.close(fig) 

def show_3D_traj(eef_traj, magnitude, plot_path):
    traj = eef_traj

    fig = plt.figure(figsize = (10, 7)) #figsize = (10, 7)
    ax = plt.axes(projection ="3d")

    xs = [traj[t][0].item() for t in range(len(traj))]
    ys = [traj[t][1] for t in range(len(traj))]
    zs = [traj[t][2] for t in range(len(traj))]

    for ball_pose in balls_xyz:
        print("ball: ", balls_xyz)
        ax.plot(ball_pose[0], ball_pose[1], ball_pose[2], "o", markersize=35)


    magnitude = list(magnitude)
    min_magnitude = min(magnitude)
    max_magnitude = max(magnitude)
    magnitude.insert(0, magnitude[0] - min_magnitude)
    heats = [[0, (magnitude[t] - min_magnitude) / (max_magnitude - min_magnitude), 0] for t in range(len(traj))]
    ax.scatter(xs, ys, zs, c=heats, s=[30+2*i for i in range(len(traj))]) 
    ax.plot(xs, ys, zs, color='red') 

    ax.legend()

    plt.title(f"eef traj open loop rollout")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    # ax.set_xticks([-0.1+0.02*i for i in range(11)])
    # ax.set_yticks([-0.6+0.02*i for i in range(11)])
    # ax.set_zticks([0+0.02*i for i in range(11)])
    plt.show()

    fig.savefig(f"{plot_path}/traj_vis.png")
    plt.cla()
    plt.close(fig) 



if __name__=="__main__":
    torch.manual_seed(5) #2021
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    observation_dim = 256 + 3
    state_dim = 256 + 3
    action_dim = 3
    init_noise_std = 1.0
    model_cfg = {'pi_hid_sizes': [256, 128, 64], 'vf_hid_sizes':[256, 128, 64], 'activation':'selu'}
    asymmetric = False

    # policy_model_path = "/home/dvrk/dvrk_ws/src/dvrk_env/shape_servo_control/src/rlgpu/logs/cut_change_gt/model_250.pt" #cut_small_delta/model_125.pt
    policy_model_path = "/home/dvrk/dvrk_ws/src/dvrk_env/shape_servo_control/src/rlgpu/logs/cut_change/model_75.pt" #cut_small_delta/model_125.pt
    AE_model_path = "/home/dvrk/LfD_data/ex_AE_cut/1ball/weights_conv1d/epoch_200"
    reward_model_path = "/home/dvrk/LfD_data/ex_cut/1ball/weights_change/epoch_400"
    
    actor_critic = ActorCritic((observation_dim,), (state_dim,), (action_dim,),
                                                init_noise_std, model_cfg, asymmetric=asymmetric).to(device)
    actor_critic.load_state_dict(torch.load(policy_model_path))


    encoder = AutoEncoder(num_points=256, embedding_size=256).to(device)
    encoder.load_state_dict(torch.load(AE_model_path))

    reward_net = RewardNet().to(device)
    reward_net.load_state_dict(torch.load(reward_model_path))
    reward_net.eval()




    # get pc
    # pc_path = "/home/dvrk/LfD_data/ex_cut/1ball/demos_train/group 20 sample 0.pickle" #10 #26
    # group 0 is the middle one (exp1), group 150 is the right one (exp2), group 5 is the left one (exp3)
    pc_path = "/home/dvrk/LfD_data/RL_cut/1ball/PC/processed sample 0.pickle" #150 #good:0,12

    with open(pc_path, 'rb') as handle:
        data = pickle.load(handle)
    pc = data["partial_pc"]#data["pcds"][0]
    #balls_xyz = data["balls_xyz"]
    
    with open("/home/dvrk/LfD_data/RL_cut/1ball/rigid_state/full_data/group 0.pickle", 'rb') as handle:
        data = pickle.load(handle)
    balls_xyz = data["balls_relative_xyz"]
    soft_xyz = data["soft_xyz"]
    soft_xyz = np.array(soft_xyz)
    balls_xyz[:, :2] = balls_xyz[:, :2] + soft_xyz[:2]
    balls_xyz[:, 2]+=EEF_GROUND_Z_OFFSET


    print("xmax:", np.max(pc[:,0]))
    print("xmin:", np.min(pc[:,0]))
    print("ymax:", np.max(pc[:,1]))
    print("ymin:", np.min(pc[:,1]))
    print("zmax:", np.max(pc[:,2]))
    print("zmin:", np.min(pc[:,2]))
    print("balls_xyz", balls_xyz)
    pc = np.expand_dims(pc, axis=0)
    pc = farthest_point_sample_batched(pc, npoint=256)
    print("+++++++++ after")
    print("xmax:", np.max(pc[:,0]))
    print("xmin:", np.min(pc[:,0]))
    print("ymax:", np.max(pc[:,1]))
    print("ymin:", np.min(pc[:,1]))
    print("zmax:", np.max(pc[:,2]))
    print("zmin:", np.min(pc[:,2]))
    obj_emb = to_obj_emb(encoder, device, pc)

    ######### visualize pointcloud ##############
    vis = True

    if vis:
        with torch.no_grad():
            points = pc[0]
            print(points.shape)

            points = points[np.random.permutation(points.shape[0])]
        
            pcd = open3d.geometry.PointCloud()
            pcd.points = open3d.utility.Vector3dVector(np.array(points))
            pcd.paint_uniform_color([0, 1, 0])

            points_tensor = torch.from_numpy(points.transpose(1,0)).unsqueeze(0).float().to(device)
            reconstructed_points = encoder(points_tensor)
            
            reconstructed_points = np.swapaxes(reconstructed_points.squeeze().cpu().detach().numpy(),0,1)
            reconstructed_points = reconstructed_points[:,:3]
            print(reconstructed_points.shape)

            pcd2 = open3d.geometry.PointCloud()
            pcd2.points = open3d.utility.Vector3dVector(np.array(reconstructed_points))
            pcd2.paint_uniform_color([1, 0, 0])

            open3d.visualization.draw_geometries([pcd, pcd2.translate((0,0,0.1))])

    ###### visualize reward ##############
    if vis:
        rewards = []
        num_samples = 1000
        eef_poses = []

        z = EEF_GROUND_Z_OFFSET+0.002
        for sample in range(num_samples):
            x = np.random.uniform(low=-0.1, high=0.1)
            # y = np.random.uniform(low=-0.5, high=-0.3)
            y = np.random.uniform(low=-0.45, high=-0.35)
            eef_pose = np.array([x,y,z])
            eef_poses.append(eef_pose)
            rew = compute_predicted_reward(eef_pose, obj_emb, reward_net, device)
            rewards.append(rew)

        max_reward = max(rewards)
        print("max reward", max_reward)
        min_reward = min(rewards)
        print("min reward", min_reward)

        fig = plt.figure(figsize = (10, 7))
        ax = plt.axes(projection ="3d")

        for ball_pose in balls_xyz:
            ax.plot(ball_pose[0], ball_pose[1], ball_pose[2], "o", markersize=35)


        xs = [eef_poses[t][0] for t in range(num_samples)]
        ys = [eef_poses[t][1] for t in range(num_samples)]
        heats = [[(rewards[t] - min_reward) / (max_reward - min_reward), 0, 0] for t in range(num_samples)]
        zs = [(heats[t][0]) for t in range(num_samples)] # height of each point is the reward
        #zs = [(rewards[t]) for t in range(num_samples)] # height of each point is the reward
        ax.scatter(xs, ys, zs, c=heats) 

        plt.title(f"function of predicted rewards (heat) with different eef xy poses (fixed z)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.show()

    ##################################################
    ################# rollout ########################
    ##################################################
    num_rollouts = 200 #200 #10
    action_scale = 1 #0.5
    traj_len = 30

    reward_mean = -61.79318511295319
    reward_std = 0.22390996351244896
    
    eef_rollouts = []
    reward_rollouts = []
    for rollout_idx in range(num_rollouts):
        if rollout_idx % 50 == 0:
            print(f"############# rollout idx: {rollout_idx} ################")

        current_xyz = torch.tensor([[0, -0.3, 0.05]]).float().to(device)
        eef_xyzs = [current_xyz.cpu().detach().numpy()]

        for t in range(traj_len):
            obs = torch.cat((current_xyz, obj_emb), dim=-1).float()
            action = actor_critic.act_inference(obs)
            delta_xyz = action_to_robot_delta_xyz(action, current_xyz)
            target_xyz = current_xyz + action_scale*delta_xyz
            #print("target_xyz: ", target_xyz)
            eef_xyzs.append(target_xyz.cpu().detach().numpy())
            current_xyz = target_xyz

        eef_traj_torch = torch.tensor(np.array(eef_xyzs)).squeeze().float().to(device)
        obj_emb_traj = obj_emb.repeat(traj_len+1, 1)
        # print("eef trah shape", eef_traj_torch.shape)
        # print("obj emb traj shape", obj_emb_traj.shape)
        total_return = compute_predicted_traj_reward(eef_traj_torch, obj_emb_traj, reward_net, device)
        total_return = (total_return - reward_mean*(traj_len+1))/reward_std

        eef_rollouts.append(eef_xyzs)
        reward_rollouts.append(total_return)

    reward_rollouts = np.array(reward_rollouts)
    best_rollout_idx = np.argmax(reward_rollouts)
    print(f"####### best reward (normalized): {reward_rollouts[best_rollout_idx]}")

    eef_xyzs = eef_rollouts[best_rollout_idx]
    eef_xyzs = np.concatenate(eef_xyzs, axis=0)
    eef_deltas = eef_xyzs[1:,:] - eef_xyzs[:-1,:]

    assert(len(eef_xyzs)==traj_len+1)
    assert(len(eef_deltas)==traj_len)

    #########################################################################################################
    ################################ visualize rollouts #####################################################
    #########################################################################################################
    plot_path = os.path.join(os.getcwd(), "figures")
    print("current path: ", plot_path)
    os.makedirs(plot_path, exist_ok=True)

    ##### save the rollout somewhere #####
    np.savetxt(os.path.join(plot_path, "traj_xyz.csv"), eef_xyzs, delimiter=",")
    np.savetxt(os.path.join(plot_path, "traj_delta.csv"), eef_deltas, delimiter=",")

    ##### plot rollout for debugging #####
    temporal_eef_diff = np.linalg.norm((eef_xyzs[1:,:] - eef_xyzs[:-1,:]), axis=1) # ||x_(t) - x_(t-1)||
    assert(len(temporal_eef_diff)==traj_len)
    show_3D_traj(eef_xyzs, magnitude = np.insert(temporal_eef_diff,0,0), plot_path=plot_path)
    
    ###### plot eef pos and errors ########
    x_eef_poses = [eef_pos[0] for eef_pos in eef_xyzs]
    y_eef_poses = [eef_pos[1] for eef_pos in eef_xyzs]
    z_eef_poses = [eef_pos[2] for eef_pos in eef_xyzs]
    timesteps = [i for i in range(traj_len+1)]

    plot_curves(xs=timesteps, ys_list=[x_eef_poses], title=f"eef poses x", \
                x_label="timestep", y_label="value of eef pos in component x", curve_labels=["open loop"], path=plot_path)
    plot_curves(xs=timesteps, ys_list=[y_eef_poses], title=f"eef poses y", \
                x_label="timestep", y_label="value of eef pos in component y", curve_labels=["open loop"], path=plot_path)
    plot_curves(xs=timesteps, ys_list=[z_eef_poses], title=f"eef poses z", \
                x_label="timestep", y_label="value of eef pos in component z", curve_labels=["open loop"], path=plot_path)

    timesteps_1off = [i for i in range(1,traj_len+1)]
    plot_curves(xs=timesteps_1off, ys_list=[temporal_eef_diff], title=f"magnitude of one-step eef pos difference\||x_(t) - x_(t-1)||", \
                x_label="timestep", y_label="difference", curve_labels=["open loop"], path=plot_path)

    plot_curves(xs=timesteps, ys_list=[np.linalg.norm(eef_xyzs - balls_xyz, axis=1)], title=f"eef and ball diff", \
                x_label="timestep", y_label="difference", curve_labels=["open loop"], path=plot_path)

    

    ############ filter by identifying temporal eef diff ############
    # filtered_eef = []
    # prev_diff = math.inf
    # prev_diff_2 = math.inf
    # diff_threshold = 0.015
    # #diff_threshold = 0.005
    # cutoff_idx = None
    # for i, diff in enumerate(temporal_eef_diff):
    #     #if diff <= diff_threshold and prev_diff <= diff_threshold and prev_diff_2<=diff_threshold:
    #     if diff <= diff_threshold and prev_diff <= diff_threshold:
    #         cutoff_idx = i+1
    #         break
    #     #print(i+2,": ", diff)
    #     prev_diff_2 = prev_diff
    #     prev_diff = diff

    # print('cutoff_idx by norm:', cutoff_idx)
    #np.savetxt(os.path.join(plot_path, "truncated_traj_norm.csv"), eef_xyzs[:cutoff_idx+1], delimiter=",")

    ############# filter by mean convergence ############
    means = np.array([np.mean(eef_xyzs[:t+1, :], axis=0) for t in range(traj_len+1)])
    stds = np.array([np.std(eef_xyzs[:t+1, :], axis=0) for t in range(traj_len+1)])
    assert(len(means)==traj_len+1==len(stds))
    mean_diff_threshold = 0.005 #0.0009
    std_diff_threshold = 0.001 #0.0007
    cutoff_idx=None
    for t in range(1,len(means)):
        if np.linalg.norm(means[t] - means[t-1]) <= mean_diff_threshold and  np.linalg.norm(stds[t] - stds[t-1]) <= std_diff_threshold:
        #if np.linalg.norm(stds[t] - stds[t-1]) <= std_diff_threshold:
            print(t, " ", np.linalg.norm(stds[t] - stds[t-1]))
            cutoff_idx = t
            break
    
    plot_curves(xs=timesteps_1off[1:], ys_list=[np.linalg.norm(means[1:]-means[:-1], axis=1)[1:]], title=f"temporal diff eef mean", \
                x_label="timestep", y_label="difference", curve_labels=["open loop"], path=plot_path)

    plot_curves(xs=timesteps_1off[1:], ys_list=[np.linalg.norm(stds[1:]-stds[:-1], axis=1)[1:]], title=f"temporal diff eef std", \
                x_label="timestep", y_label="difference", curve_labels=["open loop"], path=plot_path)
        
    print('cutoff_idx by std:', cutoff_idx)
 
    np.savetxt(os.path.join(plot_path, "truncated_traj_std.csv"), eef_xyzs[:cutoff_idx+1], delimiter=",")

    show_3D_traj(eef_xyzs[:cutoff_idx+1], magnitude = temporal_eef_diff[:cutoff_idx+1], plot_path=plot_path)
    
    ########################### running mean and std with window size ###############
    # window = 2
    # #0->w-1, 1->2w-1, 2->3w-1
    # means = np.array([np.mean(eef_xyzs[t-window:t+1, :], axis=0) for t in range(window-1, traj_len+1, window)])
    # stds = np.array([np.std(eef_xyzs[t-window:t+1, :], axis=0) for t in range(window-1, traj_len+1, window)])
    # mean_diff_threshold = 0.0009
    # std_diff_threshold = 0.0007
    # cutoff_idx=None
    # for t in range(1,len(means)):
    #     if np.linalg.norm(means[t] - means[t-1]) <= mean_diff_threshold and  np.linalg.norm(stds[t] - stds[t-1]) <= std_diff_threshold:
    #     #if np.linalg.norm(stds[t] - stds[t-1]) <= std_diff_threshold:
    #         print(t, " ", np.linalg.norm(stds[t] - stds[t-1]))
    #         cutoff_idx = (t+1)*window - 1
    #         break
    
    # plot_curves(xs=range(2*window-1, traj_len+1, window), ys_list=[np.linalg.norm(means[1:]-means[:-1], axis=1)], title=f"temporal diff eef mean", \
    #             x_label="timestep", y_label="difference", curve_labels=["open loop"], path=plot_path)

    # plot_curves(xs=range(2*window-1, traj_len+1, window), ys_list=[np.linalg.norm(stds[1:]-stds[:-1], axis=1)], title=f"temporal diff eef std", \
    #             x_label="timestep", y_label="difference", curve_labels=["open loop"], path=plot_path)
        
    # print('cutoff_idx by std:', cutoff_idx)
 
    # np.savetxt(os.path.join(plot_path, "truncated_traj_std.csv"), eef_xyzs[:cutoff_idx+1], delimiter=",")

    # show_3D_traj(eef_xyzs[:cutoff_idx+1], magnitude = temporal_eef_diff[:cutoff_idx+1], plot_path=plot_path)
    