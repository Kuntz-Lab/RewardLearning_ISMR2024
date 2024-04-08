#!/usr/bin/env python3
# from __future__ import print_function, division, absolute_import

import numpy as np
import os

#from .base.vec_task import VecTask

import sys
import roslib.packages as rp
pkg_path = rp.get_pkg_dir('shape_servo_control')
sys.path.append(pkg_path + '/src')
sys.path.append("/home/dvrk/catkin_ws/src/dvrk_env/shape_servo_control" + '/src')
import os
import numpy as np
from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym import gymutil
from copy import deepcopy
import rospy
import pickle
import timeit
import open3d
import torch

# from behaviors import TaskVelocityControlMultiRobot
# from core import MultiRobot
# import argparse
# from PIL import Image

sys.path.append("/home/dvrk/catkin_ws/src/dvrk_env/shape_servo_control/src/teleoperation")
sys.path.append("/home/dvrk/catkin_ws/src/dvrk_env/shape_servo_control/src/teleoperation/traceSurface/pointcloud_representation_learning")
# sys.path.append("/home/dvrk/catkin_ws/src/dvrk_env/shape_servo_control/src/teleoperation/pc_utils")
from pc_utils.compute_partial_pc import farthest_point_sample_batched, get_all_bin_seq_driver
from pc_utils.get_isaac_partial_pc import get_partial_pointcloud_vectorized_seg_ball

from traceSurface.config_utils.curve import *
from traceSurface.reward_learning.fully_connected.reward import RewardNetPointCloudEEF
from traceSurface.pointcloud_representation_learning.architecture import AutoEncoder

from rlgpu.utils.torch_jit_utils import *
from rlgpu.tasks.base.base_task import BaseTask

'''
Remember to : 
set MAX_NUM_BALLS = number of balls
set what action you want in the cfg file
set what reward you want to use
where to save the training log data

Beware of : 
obs and action space size
episode length
'''

MAX_NUM_BALLS = 2
CURVE_TYPE = "2ballFlatLinear"

def setup_cam_gpu(gym, env, cam_width, cam_height, cam_pos, cam_target):
    cam_props = gymapi.CameraProperties()
    cam_props.width = cam_width
    cam_props.height = cam_height    
    #cam_props.enable_tensors = True
    cam_handle = gym.create_camera_sensor(env, cam_props)
    gym.set_camera_location(cam_handle, env, cam_pos, cam_target)
    return cam_handle, cam_props

def sample_balls_poses(num_balls):
    balls_poses = []

    # rand_offset = np.random.uniform(low=0.02, high=0.1)
    # #rand_offset = 0.02
    # weights_list, xy_curve_weights, balls_xyz = get_balls_xyz_curve(num_balls, degree=degree, offset=rand_offset)
    weights_list, xy_curve_weights, balls_xyz = get_balls_xyz_curve(num_balls, curve_type=CURVE_TYPE)

    for xyz in balls_xyz:
        ball_pose = gymapi.Transform()
        pose = [xyz[0], xyz[1], xyz[2]]
        ball_pose.p = gymapi.Vec3(*pose)
        balls_poses.append(ball_pose)
        
    return weights_list, xy_curve_weights, balls_xyz, balls_poses

def default_poses():
    poses = []
    for i in range(MAX_NUM_BALLS):
        ball_pose = gymapi.Transform()
        pose = [100,100,-100]
        ball_pose.p = gymapi.Vec3(*pose)
        poses.append(ball_pose)
    return poses

def default_xyz():
    poses = []
    for i in range(MAX_NUM_BALLS):
        pose = [100,100, -100]
        poses.append(pose)
    return poses

def set_balls_poses():
    rand_num_balls = MAX_NUM_BALLS # random.randint(MIN_NUM_BALLS, MAX_NUM_BALLS) #1
    weights_list, xy_curve_weights, balls_xyz, balls_poses = sample_balls_poses(num_balls=rand_num_balls)
    # Load ball objects with maximum amount
    all_balls_poses = default_poses()
    all_balls_xyz = default_xyz()
    all_balls_poses[0:rand_num_balls] = balls_poses
    all_balls_xyz[0:rand_num_balls] = balls_xyz
    
    return weights_list, xy_curve_weights, balls_xyz, all_balls_poses, all_balls_xyz, rand_num_balls

def to_obj_emb(model, device, pcds):
    '''
    pcds has shape (num_batch, num_points, point_dim)
    '''
    pcd_tensor = torch.from_numpy(pcds.transpose(0,2,1)).float().to(device)
    with torch.no_grad():
        emb = model(pcd_tensor, get_global_embedding=True)
    return emb

def gt_reward_function(all_envs_eef_pose, all_envs_balls_xyz, remain_mask, last_ball_xyz, overlap, use_bonus=True):
    reshaped_remain_mask = np.expand_dims(remain_mask, axis=2)
    reshaped_remain_mask = np.repeat(reshaped_remain_mask, 3, axis=2) # (num_envs, max_num_balls, 3)
    drop_xyz = np.array([[[100, 100, -100]]])
    drop_xyz = np.repeat(drop_xyz, all_envs_balls_xyz.shape[1], axis=1)
    drop_xyz = np.repeat(drop_xyz, all_envs_balls_xyz.shape[0], axis=0) # (num_envs, max_num_balls, 3)

    remain_balls_xyz = np.where(reshaped_remain_mask==1, all_envs_balls_xyz, drop_xyz)
    all_envs_eef_pose = np.array(all_envs_eef_pose)
    all_envs_eef_pose_reshaped = np.expand_dims(all_envs_eef_pose, axis=1)
    all_envs_eef_pose_reshaped = np.repeat(all_envs_eef_pose_reshaped, all_envs_balls_xyz.shape[1], axis=1) # (num_envs, max_num_balls, 3)

    rewards = np.sum((all_envs_eef_pose_reshaped - remain_balls_xyz)**2, axis=2) # (num_envs, max_num_balls)
    rewards = np.min(rewards, axis=1) # (num_envs, )
    rewards = np.reshape(rewards, -1)

    ######### inv dist
    rewards = 1/(rewards+1e-4)
    # handle when all balls were touched
    remain_counts = np.sum(remain_mask, axis=1)
    reward_copy = np.copy(rewards)
    no_ball_reward = 1/(np.sum((all_envs_eef_pose - last_ball_xyz)**2, axis=1)+1e-4) #(num_envs, )
    rewards = np.where(remain_counts==0, no_ball_reward, reward_copy)

    #### add bonus when ball is touched
    if use_bonus:
        bonus = 1e6
        bonuses = np.sum(overlap, axis=1) * bonus
        rewards += bonuses

    return rewards #-1000


############################################################################################################################################

class PointReacher(BaseTask):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        np.random.seed(2021)
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless
        
        ###################### determine state and action space #######################
        self.use_eef_vel_action = self.cfg["control"]["eef_vel"]
        self.use_eef_pos_action = self.cfg["control"]["eef_pos"]
        assert(int(self.use_eef_vel_action) + int(self.use_eef_pos_action) == 1)
        self.train_w_learned_rew = self.cfg["reward"]["train_with_learned_reward"]
        

        if self.use_eef_vel_action:
            if self.train_w_learned_rew:
                self.cfg["env"]["numObservations"] = 259
            else:
                self.cfg["env"]["numObservations"] = 3 + 3*MAX_NUM_BALLS + MAX_NUM_BALLS # eef_pos, ball_poses, bin_mask
            self.cfg["env"]["numActions"] = 3 
            self.cfg["env"]["episodeLength"] = 30
        elif self.use_eef_pos_action:
            if self.train_w_learned_rew:
                self.cfg["env"]["numObservations"] = 259
            else:
                self.cfg["env"]["numObservations"] = 3 + 3*MAX_NUM_BALLS + MAX_NUM_BALLS # eef_pos, ball_poses, bin_mask
            self.cfg["env"]["numActions"] = 3
            self.cfg["env"]["episodeLength"] = 30 #10

        print("!!!!!!!! obs dim: ", self.cfg["env"]["numObservations"]) 
        print("!!!!!!!! action dim: ", self.cfg["env"]["numActions"]) 
        ###############################################################################

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.max_push_effort = self.cfg["env"]["maxEffort"]
        self.use_abs_eef_action = self.cfg["control"]["abs_eef_pos_action"]

        #Values to be filled in at runtime
        self.states = {}                        # will be dict filled with relevant states to use for reward calculation
        self.handles = {}                       # will be dict mapping names to relevant sim handles
        self.num_dofs = None                    # Total number of DOFs per env
        self.actions = None                     # Current actions to be deployed
        self._balls_states = None                
    
        # Tensor placeholders
        self._root_state = None             # State of root body        (n_envs, 13)
        self._dof_state = None  # State of all joints       (n_envs, n_dof)
        self._rigid_body_state = None  # State of all rigid bodies             (n_envs, n_bodies, 13)
        self._eef_state = None  # end effector state (at grasping point)
        self._global_indices = None         # Unique indices corresponding to all envs in flattened array

        super().__init__(cfg=self.cfg)

        self._refresh()

        ################## set up viewer
        if not self.headless:
            cam_target = gymapi.Vec3(0.0, -0.4, 0.05)
            cam_pos = gymapi.Vec3(0.2, -0.61, 0.4)

            self.gym.viewer_camera_look_at(self.viewer, self.envs[0], cam_pos, cam_target)

        ################# Camera for point cloud setup
        self.cam_handles = []
        cam_width = 600
        cam_height = 600
       
        cam_targets = gymapi.Vec3(0.0, -0.4, 0.05)
        cam_positions = gymapi.Vec3(0.2, -0.7, 0.2)
        
        for i, env in enumerate(self.envs):
            cam_handle, self.cam_prop = setup_cam_gpu(self.gym, self.envs[i], cam_width, cam_height, cam_positions, cam_targets)
            self.cam_handles.append(cam_handle)

        ################# Camera for recording rgb image
        self.img_cam_handles = []
        cam_width = 600
        cam_height = 600
       
        cam_targets = gymapi.Vec3(0.0, -0.4, 0.05)
        cam_positions = gymapi.Vec3(0.11, -0.61, 0.3)
        
        for i, env in enumerate(self.envs):
            cam_handle, self.img_cam_prop = setup_cam_gpu(self.gym, self.envs[i], cam_width, cam_height, cam_positions, cam_targets)
            self.img_cam_handles.append(cam_handle)

        ################# models and some cache ####################
        if self.train_w_learned_rew:
            self.encoder =  AutoEncoder(num_points=256, embedding_size=256).to(self.device)
            self.encoder.load_state_dict(torch.load("/home/dvrk/LfD_data/ex_AE_balls_corrected/weights_straight_flat_2ball/weights_1/epoch_150"))

            self.reward_net = RewardNetPointCloudEEF()
            self.reward_net.to(self.device)
            reward_model_path = '/home/dvrk/LfD_data/ex_trace_curve_corrected/weights_straight_flat_2ball/weights_1'
            self.reward_net.load_state_dict(torch.load(os.path.join(reward_model_path, "epoch_200")))
            self.reward_net.eval()

            self.obj_embs = torch.ones([self.num_envs, self.encoder.embedding_size], dtype=torch.float64, device=self.device)
            self.reset_pc = True

        self.last_ball_xyz = np.ones((self.num_envs, 3))
        self.overlap = np.zeros((self.num_envs, MAX_NUM_BALLS))

        ################ for saving data of each episode for visualization ##################
        self.rl_data_dict = {"gt_cum_reward": np.zeros((self.num_envs,)), "learned_cum_reward": np.zeros((self.num_envs,)), "images": [], "num_balls_reached":None, "balls_reached_per_step": []}
        self.rl_data_idx = 0
        self.save_idx = 10
        self.rl_data_path = "/home/dvrk/RL_data/pointReacher/gt_reward_eef_pos_rand_pc_obs/data"
        os.makedirs(self.rl_data_path, exist_ok=True)

    def create_sim(self):
        # set the up axis to be z-up given that assets are y-up by default
        #self.up_axis = self.cfg["sim"]["up_axis"]
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        
    def _create_ground_plane(self):
        # add ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)# z-up ground
        self.gym.add_ground(self.sim, plane_params)


    def _create_envs(self, num_envs, spacing, num_per_row):
        # define plane on which environments are initialized
        lower = gymapi.Vec3(-spacing, 0.0, -spacing)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        # load ball asset
        asset_root = self.cfg["env"]["asset"]["ballAssetRoot"]
        ball_asset_file = self.cfg["env"]["asset"]["ballAssetFile"]
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.disable_gravity = True

        ball_asset = self.gym.load_asset(self.sim, asset_root, ball_asset_file, asset_options)   

        # define robot (reacher) pose
        reacher_pose = gymapi.Transform()
        reacher_pose.p = gymapi.Vec3(-0.1, -0.5, 0.1)
        self.init_reacher_pose = reacher_pose

        #################### generate balls on the curve, assume all on the same x-y plane ######################
        envs_all_balls_poses = []
        self.all_envs_balls_xyz = []
        self.all_envs_all_balls_xyz = []
        #weights_list, xy_curve_weights, balls_xyz, all_balls_poses, all_balls_xyz, self.rand_num_balls = set_balls_poses()
        for i in range(self.num_envs):
            weights_list, xy_curve_weights, balls_xyz, all_balls_poses, all_balls_xyz, self.rand_num_balls = set_balls_poses()
            envs_all_balls_poses.append(all_balls_poses)
            self.all_envs_balls_xyz.append(balls_xyz)
            self.all_envs_all_balls_xyz.append(all_balls_xyz)

        # cache some common handles for later use
        self.envs = []
        self.reacher_handles = []
        self.ball_handles = []
        self.ball_seg_Ids = [[] for i in range(self.num_envs)] # shape: (num_envs, MAX_NUM_BALLS)
        ball_name = 5
    
        for i in range(self.num_envs):
            
            # create env
            env = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.envs.append(env)

            # add reacher (green)
            reacher_handle = self.gym.create_actor(env, ball_asset, reacher_pose, "reacher", i+1, 1, segmentationId=11)    
            color = gymapi.Vec3(*[0, 1, 0]) 
            self.gym.set_rigid_body_color(env, reacher_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
            self.reacher_handles.append(reacher_handle)  
            
            # add ball obj (all blue)   
            for j in range(MAX_NUM_BALLS):     
                seg_Id = ball_name*10
                ball_actor = self.gym.create_actor(env, ball_asset, envs_all_balls_poses[i][j], f"{ball_name}", ball_name, segmentationId=seg_Id)
                self.ball_seg_Ids[i].append(seg_Id)
                color = gymapi.Vec3(*[0, 0, 1]) #list(np.random.uniform(low=[0,0,0], high=[1,1,1], size=3))
                self.gym.set_rigid_body_color(env, ball_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
                self.ball_handles.append(ball_actor)
                ball_name += 1  

        
        self.ball_seg_Ids = np.array(self.ball_seg_Ids)
        self.all_envs_balls_xyz = np.array(self.all_envs_balls_xyz)
        self.all_envs_all_balls_xyz = np.array(self.all_envs_all_balls_xyz)
        remain_mask = [[1 for j in range(MAX_NUM_BALLS)] for i in range(self.num_envs)]
        self.remain_mask = np.array(remain_mask)
        self.dropped_mask = np.where(self.remain_mask==0, np.ones((self.num_envs, MAX_NUM_BALLS)), np.zeros((self.num_envs, MAX_NUM_BALLS)))
        
        # Setup data
        self.init_data()

    def init_data(self):
        # Setup sim handles
        # self.handles = {
        #     "eef": self.gym.find_actor_rigid_body_handle(self.envs[0], self.dvrk_handles[0], "psm_tool_yaw_link")
        # }
       
        # Get total DOFs
        #self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        # Setup tensor buffers
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        #_dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(self.num_envs, -1, 13)
        #self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs, -1, 2)
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(self.num_envs, -1, 13)

        self._eef_state = self._rigid_body_state[:, 0, :]
        
        self._balls_states = self._root_state[:, 1:1+MAX_NUM_BALLS, :]
      

        # Initialize indices
        self._global_indices = torch.arange(self.num_envs * (1+MAX_NUM_BALLS), dtype=torch.int32,
                                           device=self.device).view(self.num_envs, -1)


    def _update_states(self):
        self.states.update({
            # dvrk
            "eef_pos": self._eef_state[:, :3],
            "eef_quat": self._eef_state[:, 3:7],
            "eef_vel": self._eef_state[:, 7:],
            "balls_pos": self._balls_states[:, :, :3]
        })


    def _refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

        # Refresh states
        self._update_states()


    def compute_reward(self, actions):

        ######################### gt reward ########################
        gt_rewards = gt_reward_function(self.states["eef_pos"].cpu().numpy(), self.all_envs_balls_xyz, self.remain_mask, self.last_ball_xyz, self.overlap, use_bonus=False)
        print("gt reward: ", np.around(gt_rewards[0], decimals=4))
        
        if not self.train_w_learned_rew:
            reset_buf = torch.where((self.progress_buf >= self.max_episode_length - 1) , torch.ones_like(self.reset_buf), self.reset_buf)
            self.rew_buf[:] = torch.tensor(gt_rewards).to(self.device)
            self.reset_buf[:] = torch.clone(reset_buf).to(self.device)

        # reset_buf = torch.where((self.progress_buf >= self.max_episode_length - 1) , torch.ones_like(self.reset_buf), self.reset_buf)
        # self.rew_buf[:] = torch.tensor(gt_rewards).to(self.device)
        # self.reset_buf[:] = torch.clone(reset_buf).to(self.device)


        ######################### learned reward ######################
        if self.train_w_learned_rew:
            eef_state = self.obs_buf[:, :3].unsqueeze(1) # (num_envs, 1, 3)
            emb_state = self.obs_buf[:, 3:].unsqueeze(1) # (num_envs, 1, 256)
            with torch.no_grad():
                rewards, _ = self.reward_net.cum_return(eef_state, emb_state) # (num_envs, 1)
            rewards = rewards.reshape(-1)
            print(f"learned reward: {rewards[0].item()}")

            reset_buf = torch.where((self.progress_buf >= self.max_episode_length - 1) , torch.ones_like(self.reset_buf), self.reset_buf)
            self.rew_buf[:] = rewards
            self.reset_buf[:] = torch.clone(reset_buf).to(self.device)

            self.rl_data_dict["learned_cum_reward"] += rewards.cpu().detach().numpy()

        self.rl_data_dict["gt_cum_reward"] += gt_rewards



    def compute_observations(self, env_ids=None):
        self._refresh()

        if env_ids is None:
            env_ids = np.arange(self.num_envs)

        self.gym.render_all_camera_sensors(self.sim)

        # save image to data_dict
        if self.rl_data_idx % self.save_idx == 0:
            im = self.gym.get_camera_image(self.sim, self.envs[0], self.img_cam_handles[0], gymapi.IMAGE_COLOR).reshape((600,600,4))
            im = np.expand_dims(im, axis=0)
            self.rl_data_dict["images"].append(im)

            num_balls_reached = np.sum(self.dropped_mask, axis=1)
            self.rl_data_dict["balls_reached_per_step"].append(num_balls_reached)

        if self.train_w_learned_rew:
            ############## use pre-processed object embedding as observation############
            empty_tuple = tuple([1 for i in range(MAX_NUM_BALLS)])
            for i in env_ids:
                if tuple(self.dropped_mask[i]) == empty_tuple:
                    continue
                #print(self.emb_dict[tuple(self.dropped_mask[i])].shape)
                self.obj_embs[i] = self.emb_dict[tuple(self.dropped_mask[i])][i]
            self.obs_buf = torch.cat((torch.clone(self.states["eef_pos"]), self.obj_embs), dim=-1).float()
        else:
            ############# no pointcloud embedding observation #####################
            bin_mask = torch.tensor(self.dropped_mask).to(self.device).float()
            obs_list = []
            obs_list.append(torch.clone(self.states["eef_pos"]))
            for i in range(MAX_NUM_BALLS):
                obs_list.append(torch.clone( self._balls_states[:, i, :3]))
            obs_list.append(bin_mask)
            self.obs_buf = torch.cat(obs_list, dim=-1).float()
        return self.obs_buf

    def reset_idx(self, env_ids):
        print("reseting ...")
        #env_ids = env_ids.cpu().numpy().astype(np.int32)
        env_ids = np.array([i for i in range(self.num_envs)])
        num_reset = len(env_ids)

        ##################### save data dict for visualization and resey data dict
        num_balls_reached = np.sum(self.dropped_mask, axis=1)
        self.rl_data_dict["num_balls_reached"] = num_balls_reached
        if self.rl_data_idx % self.save_idx == 0:
            print(f"!!!!!!!!!!!!!!!!!!!!!! save rl data at idx {self.rl_data_idx} !!!!!!!!!!!!!!!!!!!!")
            with open(os.path.join(self.rl_data_path, f"episode {self.rl_data_idx}.pickle"), 'wb') as handle:
                pickle.dump(self.rl_data_dict, handle, protocol=3)   
        self.rl_data_idx += 1

        self.rl_data_dict = {"gt_cum_reward": np.zeros((self.num_envs,)), "learned_cum_reward": np.zeros((self.num_envs,)), "images": [], "num_balls_reached":None, "balls_reached_per_step": []}

       ############################# reset balls ######################################
        #weights_list, xy_curve_weights, self.balls_xyz, all_balls_poses, self.all_balls_xyz, self.rand_num_balls = set_balls_poses()
        for i in env_ids:
            weights_list, xy_curve_weights, balls_xyz, all_balls_poses, all_balls_xyz, self.rand_num_balls = set_balls_poses()
            self.all_envs_balls_xyz[i] = np.array(balls_xyz)
            self.all_envs_all_balls_xyz[i] = np.array(all_balls_xyz)

        root_state = torch.clone(self._root_state).cpu().numpy()
        # reset balls
        balls_states = root_state[:, 1:1+MAX_NUM_BALLS, :]
        new_ball_pos = self.all_envs_all_balls_xyz[env_ids, :, :]
        balls_states[env_ids, :, :3]=new_ball_pos
        balls_states[env_ids, :, 3:7]=torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).repeat(num_reset, MAX_NUM_BALLS, 1).cpu().numpy() # no rotation
        balls_states[env_ids, :, 7:10]=torch.tensor([0.0, 0.0, 0.0], device=self.device).repeat(num_reset, MAX_NUM_BALLS, 1).cpu().numpy() # 0 velocity

        # reset robot
        eef_states = root_state[:, 0, :]
        new_reacher_pos = torch.tensor([self.init_reacher_pose.p.x, self.init_reacher_pose.p.y, self.init_reacher_pose.p.z]).to(self.device)
        eef_states[env_ids, :3]=new_reacher_pos.cpu().numpy()
        eef_states[env_ids, 3:7]=torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).cpu().numpy() # no rotation
        eef_states[env_ids, 7:10]=torch.tensor([0.0, 0.0, 0.0], device=self.device).cpu().numpy() # 0 velocity

        # reset balls and robot states
        multi_env_ids_cubes_int32 = self._global_indices[env_ids, :].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(torch.tensor(root_state).to(self.device)),
            gymtorch.unwrap_tensor(multi_env_ids_cubes_int32), len(multi_env_ids_cubes_int32))

        ########################## reset cache (masks) ########################################
        self.remain_mask[env_ids, :] = np.array([1 for j in range(MAX_NUM_BALLS)])
        self.dropped_mask[env_ids, :] = np.array([0 for j in range(MAX_NUM_BALLS)])
        self.last_ball_xyz[env_ids, :] = np.array([1,1,1])

        ########################## reset color of balls ############################################
        ball_id = 0
        for i in range(1):
            for j in range(MAX_NUM_BALLS):
                color = gymapi.Vec3(*[0, 0, 1])
                self.gym.set_rigid_body_color(self.envs[i], self.ball_handles[ball_id], 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
                ball_id += 1
        
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        if self.train_w_learned_rew:
            self.reset_pc = True
        

    def drop_ball(self):
        print("----------- drop ball --------------")
        reshaped_remain_mask = np.expand_dims(self.remain_mask, axis=2)
        reshaped_remain_mask = np.repeat(reshaped_remain_mask, 3, axis=2) # (num_envs, max_num_balls, 3)
        drop_xyz = np.array([[[100, 100, -100]]])
        drop_xyz = np.repeat(drop_xyz, self.all_envs_balls_xyz.shape[1], axis=1)
        drop_xyz = np.repeat(drop_xyz, self.all_envs_balls_xyz.shape[0], axis=0) # (num_envs, max_num_balls, 3)

        remain_balls_xyz = np.where(reshaped_remain_mask==1, self.all_envs_balls_xyz, drop_xyz)

        all_envs_eef_pose = self.states["eef_pos"].cpu().numpy()
        all_envs_eef_pose = np.expand_dims(all_envs_eef_pose, axis=1)
        all_envs_eef_pose = np.repeat(all_envs_eef_pose, self.all_envs_balls_xyz.shape[1], axis=1) # (num_envs, max_num_balls, 3)

        ######################### drop balls when overlap with eef #############################
        eef_ball_diff = np.sum((all_envs_eef_pose - remain_balls_xyz)**2, axis=2) # (num_envs, max_num_balls)
        overlap = eef_ball_diff <= 0.0001 #0.002#0.0003 #0.00025 #0.0001

        self.remain_mask = np.where(overlap, np.zeros_like(self.remain_mask), self.remain_mask)
        self.dropped_mask = np.where(overlap, np.ones_like(self.remain_mask), self.dropped_mask)

        print(f"ball remained mask: {self.remain_mask[0]}")
        print(f"ball dropped mask: {self.dropped_mask[0]}")

        ############################# update the last ball touched by eef #######################
        last_ball_xyz_copy = np.copy(self.last_ball_xyz)

        # for each env, there is no or at least one overlap. That overlapped ball would be updated as the last ball
        overlap_ball_idx = np.argmax(overlap.astype(int), axis=1)
        overlap_ball_xyz = remain_balls_xyz[np.arange(start=0, stop=self.num_envs), overlap_ball_idx, :] #(num_envs, 3)
        
        overlap_is_exist = np.max(overlap.astype(int), axis=1)
        print(f"any overlap: {overlap_is_exist[0]}")
        overlap_is_exist = np.expand_dims(overlap_is_exist, axis=1)
        overlap_is_exist = np.repeat(overlap_is_exist, 3, axis=1) # (num_envs, 3)

        self.last_ball_xyz = np.where(overlap_is_exist, overlap_ball_xyz, last_ball_xyz_copy)
       
        print(f"last ball touched: {self.last_ball_xyz[0]}")

        ####################### change color of touched ball ######################
        ball_id = 0
        for i in range(1):
            for j in range(MAX_NUM_BALLS):
                if overlap[i, j]:
                    color = gymapi.Vec3(*[1, 0, 0])
                    self.gym.set_rigid_body_color(self.envs[i], self.ball_handles[ball_id], 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
                ball_id += 1

        #### keep track of bonus ###
        self.overlap = overlap.astype(int)

        print("----------- done drop ball --------------")
        

    def sigmoid_clip(self, input_tensor, min_value, max_value):
        scaled_tensor = torch.sigmoid(input_tensor)
        clipped_tensor = min_value + (max_value - min_value) * scaled_tensor
        return clipped_tensor

    def pre_physics_step(self, actions):
        if self.train_w_learned_rew and self.reset_pc:
            print(f"!!!!!! skip pre physics for resetting pc !!!!!!")
            return 
        self._refresh()
        if self.use_eef_vel_action:
            ########################## actions are eef vel ###############################
            print(f"vel action:{actions}")

            actions = torch.clone(actions)

            multi_env_ids_int32 = self._global_indices[:, :].flatten()

            root_state = torch.clone(self._root_state).cpu().numpy()
            eef_states = root_state[:, 0, :]

            eef_states[:, 7:10]=actions.cpu().numpy()

            self.gym.set_actor_root_state_tensor_indexed(
                self.sim, gymtorch.unwrap_tensor(torch.tensor(root_state).to(self.device)),
                gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))
        else:
            ########################## actions are eef pos ###############################
            #### pos control with action=absolute_eef_pose####
            if self.use_abs_eef_action:
                print(f"pos action:{actions}")
                actions_clipped = torch.clone(actions)
                actions_clipped[:, 0] = self.sigmoid_clip(actions_clipped[:, 0], min_value=-0.1, max_value=0.1)
                actions_clipped[:, 1] = self.sigmoid_clip(actions_clipped[:, 1], min_value=-0.6, max_value=-0.4)
                actions_clipped[:, 2] = self.sigmoid_clip(actions_clipped[:, 2], min_value=0.1, max_value=0.11)
                # actions_clipped[:, 1] -= 0.5
                # actions_clipped[:, 2] += 0.1
                print(f"clipped action:{actions_clipped}")
                #####################
                multi_env_ids_int32 = self._global_indices[:, :].flatten()

                root_state = torch.clone(self._root_state).cpu().numpy()
                eef_states = root_state[:, 0, :]

                eef_states[:, :3]=actions_clipped.cpu().numpy()

                self.gym.set_actor_root_state_tensor_indexed(
                    self.sim, gymtorch.unwrap_tensor(torch.tensor(root_state).to(self.device)),
                    gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))
            else:
                #### pos control with action=delta_xyz####
                print(f"delta pos action:{actions}")
                world_delta_xyz = torch.clone(actions)
                current_world_xyz = torch.clone(self.states["eef_pos"])
                next_world_xyz = current_world_xyz + world_delta_xyz
                # clip delta_xyz so that the next world xyz is not outside workspace
                low_x_bound = -0.1
                high_x_bound = 0.1
                low_y_bound = -0.6
                high_y_bound = -0.4
                low_z_bound = 0.011
                high_z_bound = 0.2
                world_delta_xyz[:, 0] = torch.where(next_world_xyz[:, 0]>high_x_bound, high_x_bound - current_world_xyz[:, 0], world_delta_xyz[:, 0])
                world_delta_xyz[:, 0] = torch.where(next_world_xyz[:, 0]<low_x_bound, low_x_bound - current_world_xyz[:, 0], world_delta_xyz[:, 0])
                world_delta_xyz[:, 1] = torch.where(next_world_xyz[:, 1]>high_y_bound, high_y_bound - current_world_xyz[:, 1], world_delta_xyz[:, 1])
                world_delta_xyz[:, 1] = torch.where(next_world_xyz[:, 1]<low_y_bound, low_y_bound - current_world_xyz[:, 1], world_delta_xyz[:, 1])
                world_delta_xyz[:, 2] = torch.where(next_world_xyz[:, 2]>high_z_bound, high_z_bound - current_world_xyz[:, 2], world_delta_xyz[:, 2])
                world_delta_xyz[:, 2] = torch.where(next_world_xyz[:, 2]<low_z_bound, low_z_bound - current_world_xyz[:, 2], world_delta_xyz[:, 2])

                multi_env_ids_int32 = self._global_indices[:, :].flatten()

                root_state = torch.clone(self._root_state).cpu().numpy()
                eef_states = root_state[:, 0, :]

                new_pos = current_world_xyz + world_delta_xyz
                eef_states[:, :3]=new_pos.cpu().numpy()
                
                self.gym.set_actor_root_state_tensor_indexed(
                    self.sim, gymtorch.unwrap_tensor(torch.tensor(root_state).to(self.device)),
                    gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))
        
    def reset_pc_emb(self):
        ##################### one-time cost to compute point cloud embeddings for all envs with ball configs #############################
        self.gym.render_all_camera_sensors(self.sim)
        all_bin_seqs = get_all_bin_seq_driver(MAX_NUM_BALLS)
        all_bin_seqs.remove(tuple([1 for i in range(MAX_NUM_BALLS)]))
        print(f"all binary sequences (representing dropped masks): {all_bin_seqs}")
        emb_dict = {}
        pc_dict = {}
        is_success = True
        max_num_points = 1000
        vinv = np.linalg.inv(np.matrix(self.gym.get_camera_view_matrix(self.sim, self.envs[0], self.cam_handles[0])))
        proj = self.gym.get_camera_proj_matrix(self.sim, self.envs[0], self.cam_handles[0])
        for bin_seq in all_bin_seqs:
            processed_pcs = []
            for i in range(self.num_envs):
                pc = get_partial_pointcloud_vectorized_seg_ball(self.gym, self.sim, self.envs[i], self.cam_handles[i], vinv, proj, self.cam_prop, np.array(bin_seq), self.ball_seg_Ids[i], color=None, min_z = 0.01, visualization=False, device="cuda")
                #pc = get_partial_pointcloud_vectorized_seg_ball(self.gym, self.sim, self.envs[i], self.cam_handles[i], self.cam_prop, np.array(bin_seq), self.ball_seg_Ids[i], color=None, min_z = 0.01, visualization=False, device="cpu")
                print(f"binmask:{bin_seq}, env:{i}, pcshape:{pc.shape}")
                if len(pc)==0:
                    is_success = False
                    print("XXXXXXXXXXXXXXXXXXXXXX pointcloud recording failure, try later XXXXXXXXXXXXXXXXXXXXXXXXX")
                    return is_success
                processed_pc = np.zeros((max_num_points, 3))
                pad_point = pc[-1, :]
                processed_pc[:len(pc), :] = pc
                processed_pc[len(pc):, :] = pad_point
                processed_pcs.append(np.expand_dims(processed_pc, axis=0))

            ################## farthest point sampling in batch and convert to obj emb
            pcs = np.concatenate(processed_pcs, axis=0)
            pcs = np.array(farthest_point_sample_batched(pcs, npoint=256)) # (num_envs, 256, 3)
            emb = to_obj_emb(self.encoder, self.device, pcs) # (num_envs, 256)

            emb_dict[bin_seq] = emb
            pc_dict[bin_seq] = pcs

            vis = False
            if vis:
                with torch.no_grad():
                    points = pcs[1]
                    print(points.shape)

                    points = points[np.random.permutation(points.shape[0])]
                
                    pcd = open3d.geometry.PointCloud()
                    pcd.points = open3d.utility.Vector3dVector(np.array(points))
                    pcd.paint_uniform_color([0, 1, 0])

                    points_tensor = torch.from_numpy(points.transpose(1,0)).unsqueeze(0).float().to(self.device)
                    reconstructed_points = self.encoder(points_tensor)
                    
                    reconstructed_points = np.swapaxes(reconstructed_points.squeeze().cpu().detach().numpy(),0,1)
                    reconstructed_points = reconstructed_points[:,:3]
                    print(reconstructed_points.shape)

                    pcd2 = open3d.geometry.PointCloud()
                    pcd2.points = open3d.utility.Vector3dVector(np.array(reconstructed_points))
                    pcd2.paint_uniform_color([1, 0, 0])
                    open3d.visualization.draw_geometries([pcd, pcd2.translate((0,0,0.1))])

        if is_success:
            self.emb_dict = emb_dict
            #self.rl_data_dict["pc_dict"] = pc_dict
            print("!!!!!!!!!!!! pointcloud recording success !!!!!!!!!!!")

        return is_success

    def post_physics_step(self):
        if self.train_w_learned_rew:
            is_success = True
            if self.reset_pc:
                is_success = self.reset_pc_emb()
            if is_success:
                self.reset_pc = False
            else:
                return  

        self.progress_buf += 1
        
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        
        self.compute_observations()
        self.compute_reward(self.actions)
        self.overlap = np.zeros((self.num_envs, MAX_NUM_BALLS))
        self.drop_ball()
        
        if len(env_ids) > 0:
            self.reset_idx(env_ids)