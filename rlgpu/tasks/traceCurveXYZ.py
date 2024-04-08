#!/usr/bin/env python3
# from __future__ import print_function, division, absolute_import

import numpy as np
import os

#from .base.vec_task import VecTask

#import from Bao's motion planner
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


from behaviors import TaskVelocityControlMultiRobot
from core import MultiRobot
import argparse
from PIL import Image

sys.path.append("/home/dvrk/catkin_ws/src/dvrk_env/shape_servo_control/src/teleoperation")
# sys.path.append("/home/dvrk/catkin_ws/src/dvrk_env/shape_servo_control/src/teleoperation/pc_utils")
# from compute_partial_pc import farthest_point_sample_batched, get_all_bin_seq_driver
# from get_isaac_partial_pc import get_partial_pointcloud_vectorized_seg_ball

from traceSurface.config_utils.curve import *
# from traceSurface.reward_learning.fully_connected.reward import RewardNetPointCloudEEF
# from traceSurface.pointcloud_representation_learning.architecture import AutoEncoder

from rlgpu.utils.torch_jit_utils import *
from rlgpu.tasks.base.base_task import BaseTask

'''
Remember to : 
set MAX_NUM_BALLS = number of balls
set what action you want in the cfg file
where to save the training log data

Beware of : 
obs and action space size
episode length
'''

#############Robot z offset###############
ROBOT_Z_OFFSET = 0.25
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
    # #balls_xyz = [[-0.0, -0.5, 0.1]]
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

class TraceCurveXYZ(BaseTask):

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
        self.use_dof_vel_action = self.cfg["control"]["dof_vel"]
        assert(int(self.use_dof_vel_action) + int(self.use_eef_vel_action) + int(self.use_eef_pos_action) == 1)
        
        if self.use_dof_vel_action:
            self.cfg["env"]["numObservations"] = 3 + 10 + 3*MAX_NUM_BALLS + MAX_NUM_BALLS # eef_pos, dof_pos, ball_poses, bin_mask
            self.cfg["env"]["numActions"] = 10 #num_dofs per dvrk
            self.cfg["env"]["episodeLength"] = 30
        elif self.use_eef_vel_action:
            self.cfg["env"]["numObservations"] = 3 + 3*MAX_NUM_BALLS + MAX_NUM_BALLS # eef_pos, ball_poses, bin_mask
            self.cfg["env"]["numActions"] = 3 
            self.cfg["env"]["episodeLength"] = 30 #60
        elif self.use_eef_pos_action:
            self.cfg["env"]["numObservations"] = 3 + 3*MAX_NUM_BALLS + MAX_NUM_BALLS # eef_pos, ball_poses, bin_mask
            self.cfg["env"]["numActions"] = 3
            self.cfg["env"]["episodeLength"] = 10 #30

        print("!!!!!!!! obs dim: ", self.cfg["env"]["numObservations"]) 
        print("!!!!!!!! action dim: ", self.cfg["env"]["numActions"]) 
        print("episode length: ", self.cfg["env"]["episodeLength"])
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

            middle_env = self.envs[self.num_envs // 2 + int(np.sqrt(self.num_envs)) // 2]
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

        ##################################################
        self.last_ball_xyz = np.ones((self.num_envs, 3))
        self.overlap = np.zeros((self.num_envs, MAX_NUM_BALLS))

        # for saving data of each episode for visualization
        self.rl_data_dict = {"gt_cum_reward": np.zeros((self.num_envs,)), "learned_cum_reward": np.zeros((self.num_envs,)), "images": [], "num_balls_reached":None, "balls_reached_per_step": [], \
        "actual_eef_vels":[], "expected_eef_vels":[], "actual_eef_poses":[], "expected_eef_poses":[]}
        self.rl_data_idx = 0
        self.save_idx = 10
        self.rl_data_path = "/home/dvrk/RL_data/traceCurveXYZ/gt_reward_eef_pos_2ball/data"
        os.makedirs(self.rl_data_path, exist_ok=True)
        
        ####################################
        dummy_desired_positions = np.random.uniform(low=[-0.1, 0, 0.03], high=[0.1, 0.07, 0.05], size=(self.num_envs, 3))
        dt = 1./60.
        self.robots = MultiRobot(self.gym, self.sim, self.envs, self.dvrk_handles, 8, self.num_envs)
        self.tvc_behavior = TaskVelocityControlMultiRobot(list(dummy_desired_positions), self.robots, self.num_envs, dt, 3, vel_limits=self.vel_limits, error_threshold = 5e-3) #5e-3            


    def default_dvrk_asset(self):
        '''
        load the dvrk asset
        '''
        # dvrk asset
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.001
        asset_options.fix_base_link = True
        asset_options.thickness = 0.00025#0.0001

        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.max_angular_velocity = 40000000000.

        asset_root = self.cfg["env"]["asset"]["dvrkAssetRoot"]
        dvrk_asset_file =self.cfg["env"]["asset"]["dvrkAssetFile"]
        print("Loading asset '%s' from '%s'" % (dvrk_asset_file, asset_root))
        return self.gym.load_asset(self.sim, asset_root, dvrk_asset_file, asset_options)


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

        # load robot asset
        dvrk_asset = self.default_dvrk_asset()
        dvrk_pose = gymapi.Transform()
        dvrk_pose.p = gymapi.Vec3(0.0, -0.1, ROBOT_Z_OFFSET)
        dvrk_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)    
       
        self.num_dvrk_bodies = self.gym.get_asset_rigid_body_count(dvrk_asset)
        self.num_dvrk_dofs = self.gym.get_asset_dof_count(dvrk_asset)
        print("num dvrk bodies: ", self.num_dvrk_bodies)
        print("num dvrk dofs: ", self.num_dvrk_dofs)

        #################### assume all on the same x-y plane ######################
        envs_all_balls_poses = []
        self.all_envs_balls_xyz = []
        self.all_envs_all_balls_xyz = []
        #weights_list, xy_curve_weights, balls_xyz, all_balls_poses, all_balls_xyz, self.rand_num_balls = set_balls_poses()
        for i in range(self.num_envs):
            weights_list, xy_curve_weights, balls_xyz, all_balls_poses, all_balls_xyz, self.rand_num_balls = set_balls_poses()
            envs_all_balls_poses.append(all_balls_poses)
            self.all_envs_balls_xyz.append(balls_xyz)
            self.all_envs_all_balls_xyz.append(all_balls_xyz)

        asset_root = self.cfg["env"]["asset"]["ballAssetRoot"]
        ball_asset_file = self.cfg["env"]["asset"]["ballAssetFile"]
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.disable_gravity = True


        ball_asset = self.gym.load_asset(self.sim, asset_root, ball_asset_file, asset_options)   

        
        # cache some common handles for later use
        self.envs = []
        self.dvrk_handles = []
        self.ball_handles = []
        self.ball_seg_Ids = [[] for i in range(self.num_envs)] # shape: (num_envs, MAX_NUM_BALLS)
        #self.all_envs_balls_xyz = [self.balls_xyz for i in range(self.num_envs)]
        ball_name = 5
    
        for i in range(self.num_envs):
            
            # create env
            env = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.envs.append(env)

            # add dvrk
            dvrk_handle = self.gym.create_actor(env, dvrk_asset, dvrk_pose, "dvrk", i+1, 1, segmentationId=11)    
            self.dvrk_handles.append(dvrk_handle)  

            
            # add ball obj       
            for j in range(MAX_NUM_BALLS):     
                seg_Id = ball_name*10
                ball_actor = self.gym.create_actor(env, ball_asset, envs_all_balls_poses[i][j], f"{ball_name}", ball_name, segmentationId=seg_Id)
                self.ball_seg_Ids[i].append(seg_Id)
                color = gymapi.Vec3(*[0, 0, 1]) #list(np.random.uniform(low=[0,0,0], high=[1,1,1], size=3))
                self.gym.set_rigid_body_color(env, ball_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
                self.ball_handles.append(ball_actor)
                ball_name += 1  

            #init_dvrk_joints(self.gym, self.envs[i], self.dvrk_handles[i])
        
        self.ball_seg_Ids = np.array(self.ball_seg_Ids)
        self.all_envs_balls_xyz = np.array(self.all_envs_balls_xyz)
        self.all_envs_all_balls_xyz = np.array(self.all_envs_all_balls_xyz)
        remain_mask = [[1 for j in range(MAX_NUM_BALLS)] for i in range(self.num_envs)]
        self.remain_mask = np.array(remain_mask)
        self.dropped_mask = np.where(self.remain_mask==0, np.ones((self.num_envs, MAX_NUM_BALLS)), np.zeros((self.num_envs, MAX_NUM_BALLS)))
        

        # DOF Properties and Drive Modes 
        dof_props = self.gym.get_actor_dof_properties(self.envs[0], self.dvrk_handles[0])
        dof_props["driveMode"].fill(gymapi.DOF_MODE_VEL) #original: gymapi.DOF_MODE_EFFORT
        # dof_props["stiffness"].fill(200.0)
        # dof_props["damping"].fill(40.0)
        # dof_props["stiffness"].fill(4000)
        # dof_props["damping"].fill(1000)
        # dof_props["stiffness"][8:].fill(100)
        # dof_props["damping"][8:].fill(200)

        dof_props["stiffness"].fill(400000)
        dof_props["damping"].fill(100000)
        dof_props["stiffness"][8:].fill(10000)
        dof_props["damping"][8:].fill(20000)  

        self.vel_limits = dof_props['velocity']
        
        for i, env in enumerate(self.envs):
            self.gym.set_actor_dof_properties(env, self.dvrk_handles[i], dof_props)    # set dof properties 
        
        # Setup data
        self.init_data()

    def init_data(self):
        # Setup sim handles
        self.handles = {
            "eef": self.gym.find_actor_rigid_body_handle(self.envs[0], self.dvrk_handles[0], "psm_tool_yaw_link")
        }
       
        # Get total DOFs
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        # Setup tensor buffers
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(self.num_envs, -1, 13)
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs, -1, 2)
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(self.num_envs, -1, 13)

        self._eef_state = self._rigid_body_state[:, self.handles["eef"], :]
        
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
            "balls_pos": self._balls_states[:, :, :3],
            "dof_pos": self._dof_state[:, :, 0]
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
        
        reset_buf = torch.where((self.progress_buf >= self.max_episode_length - 1) , torch.ones_like(self.reset_buf), self.reset_buf)
        self.rew_buf[:] = torch.tensor(gt_rewards).to(self.device)
        self.reset_buf[:] = torch.clone(reset_buf).to(self.device)

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

        if self.use_eef_vel_action:
            self.rl_data_dict["actual_eef_vels"].append(torch.clone(self.states["eef_vel"]).cpu().detach().numpy())

        bin_mask = torch.tensor(self.dropped_mask).to(self.device).float()
        if self.use_dof_vel_action:
            obs_list = []
            obs_list.append(torch.clone(self.states["eef_pos"]))
            obs_list.append(torch.clone(self.states["dof_pos"]))
            for i in range(MAX_NUM_BALLS):
                obs_list.append(torch.clone( self._balls_states[:, i, :3]))
            obs_list.append(bin_mask)
            self.obs_buf = torch.cat(obs_list, dim=-1).float()
        else:
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

        self.rl_data_dict = {"gt_cum_reward": np.zeros((self.num_envs,)), "learned_cum_reward": np.zeros((self.num_envs,)), "images": [], "num_balls_reached":None, "balls_reached_per_step": [], \
        "actual_eef_vels":[], "expected_eef_vels":[], "actual_eef_poses":[], "expected_eef_poses":[]}

       ############################# reset balls ######################################
        #weights_list, xy_curve_weights, self.balls_xyz, all_balls_poses, self.all_balls_xyz, self.rand_num_balls = set_balls_poses()
        for i in env_ids:
            weights_list, xy_curve_weights, balls_xyz, all_balls_poses, all_balls_xyz, self.rand_num_balls = set_balls_poses()
            self.all_envs_balls_xyz[i] = np.array(balls_xyz)
            self.all_envs_all_balls_xyz[i] = np.array(all_balls_xyz)

    
        root_state = torch.clone(self._root_state).cpu().numpy()
        balls_states = root_state[:, 1:1+MAX_NUM_BALLS, :]

        # Write these new init balls pos to the sim states
        new_pos = self.all_envs_all_balls_xyz[env_ids, :, :]
        balls_states[env_ids, :, :3]=new_pos
        # no rotation
        balls_states[env_ids, :, 3:7]=torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).repeat(num_reset, MAX_NUM_BALLS, 1).cpu().numpy()
        # 0 velocity
        balls_states[env_ids, :, 7:10]=torch.tensor([0.0, 0.0, 0.0], device=self.device).repeat(num_reset, MAX_NUM_BALLS, 1).cpu().numpy()

        # reset balls states
        multi_env_ids_cubes_int32 = self._global_indices[env_ids, 1:].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(torch.tensor(root_state).to(self.device)),
            gymtorch.unwrap_tensor(multi_env_ids_cubes_int32), len(multi_env_ids_cubes_int32))

        ########################## reset cache (masks) ########################################
        self.remain_mask[env_ids, :] = np.array([1 for j in range(MAX_NUM_BALLS)])
        self.dropped_mask[env_ids, :] = np.array([0 for j in range(MAX_NUM_BALLS)])
        self.last_ball_xyz[env_ids, :] = np.array([1,1,1])

        ########################### reset dvrk ############################################
        multi_env_ids_int32 = self._global_indices[env_ids, 0].flatten()

        dof_state = torch.clone(self._dof_state).cpu().numpy()
        dof_state[env_ids, :, 0] = np.array([-0.3193448984150404, 0.08059437628908588, 0.04633803400231634, -0.11489753865971103, 0.23808314280302934, \
        -0.0005194664992932008, 0.010978008812426507, 0.3189222925076953, 0.35, -0.35])
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(torch.tensor(dof_state).to(self.device)),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32),
                                              len(multi_env_ids_int32))

        ########################## reset color of balls ############################################
        ball_id = 0
        for i in range(1):
            for j in range(MAX_NUM_BALLS):
                color = gymapi.Vec3(*[0, 0, 1])
                self.gym.set_rigid_body_color(self.envs[i], self.ball_handles[ball_id], 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
                ball_id += 1

        
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        

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

        #### keep track of bonus ###
        self.overlap = overlap.astype(int)

        ####################### change color of touched ball ######################
        ball_id = 0
        for i in range(1):
            for j in range(MAX_NUM_BALLS):
                if overlap[i, j]:
                    color = gymapi.Vec3(*[1, 0, 0])
                    self.gym.set_rigid_body_color(self.envs[i], self.ball_handles[ball_id], 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
                ball_id += 1

        print("----------- done drop ball --------------")
        

    def transform_coordinate(self, actions):
        actions = torch.clone(actions)
        actions[:, 0:2]*=-1
        # actions[:, 2] -= ROBOT_Z_OFFSET
        return actions

    def sigmoid_clip(self, input_tensor, min_value, max_value):
        scaled_tensor = torch.sigmoid(input_tensor)
        clipped_tensor = min_value + (max_value - min_value) * scaled_tensor
        return clipped_tensor

    def pre_physics_step(self, actions):
        self._refresh()
        if self.use_dof_vel_action:
            ############################# actions are dof vel ##########################
            self.actions = actions.clone().to(self.device)
            #print(self.num_envs * self.num_dof,"++++++++++++++++++++", actions.shape)
            actions_tensor = torch.zeros(self.num_envs * self.num_dofs, device=self.device, dtype=torch.float)
            #mask out the gripper
            mask = torch.ones_like(actions)
            mask[:,9]=0
            mask[:,8]=0
            actions_tensor += (actions*mask).reshape(self.num_envs*self.num_dofs)*self.max_push_effort
            forces = gymtorch.unwrap_tensor(actions_tensor)
            #self.gym.set_dof_actuation_force_tensor(self.sim, forces)
            self.gym.set_dof_velocity_target_tensor(self.sim, forces)
        elif self.use_eef_vel_action:
            ########################## actions are eef vel ###############################
            print(f"vel raw action:{actions}")
            
            # eef_limits = self.tvc_behavior.get_eef_vel_limits()
            # print(f"eef limits: {eef_limits}")
            # eef_limit_max_norm = torch.max(torch.norm(eef_limits[:, :3], dim=1))
            # print("max action norm: ", eef_limit_max_norm)
            
            norm = torch.norm(actions, dim=1).unsqueeze(1) #(num_envs, 1)
            actions = torch.where(norm>=1, torch.clone(actions)/norm*1, torch.clone(actions)) #(num_envs, 3)

            print(f"vel scaled action:{actions}")

            self.rl_data_dict["expected_eef_vels"].append(torch.clone(actions).cpu().detach().numpy())
           
            actions = torch.nn.functional.pad(actions, (0, 3), "constant", 0) # (num_envs, 6)
            assert(actions.shape[1]==6)
            dof_vel = self.tvc_behavior.get_action_from_eef_vel(actions, device="cpu").to(self.device) # (num_envs, 8)
            all_dof_vel = torch.nn.functional.pad(dof_vel, (0, 2), 'constant', 0) # add 0 velocities for the last 2 joints of the eef
            actions_tensor = all_dof_vel # convert to type float32.  
            self.actions = actions_tensor.clone().to(self.device)      
            actions_tensor = actions_tensor.reshape(self.num_envs*self.num_dofs) # do the same action for all environments, then flatten to a 1D vector for compatibility.
            actions_tensor = gymtorch.unwrap_tensor(actions_tensor)  
            self.gym.set_dof_velocity_target_tensor(self.sim, actions_tensor)
        else:
            ########################## actions are eef pos ###############################
            #### pos control with action=absolute_eef_pose####
            if self.use_abs_eef_action:
                print(f"pos action:{actions}")
                actions_clipped = torch.clone(actions)
                actions_clipped[:, 0] = self.sigmoid_clip(actions_clipped[:, 0], min_value=-0.1, max_value=0.1)
                actions_clipped[:, 1] = self.sigmoid_clip(actions_clipped[:, 1], min_value=-0.6, max_value=-0.4)
                actions_clipped[:, 2] = self.sigmoid_clip(actions_clipped[:, 2], min_value=0.1, max_value=0.11)
                print(f"pos clipped action:{actions_clipped}")

                self.rl_data_dict["expected_eef_poses"].append(torch.clone(actions_clipped).cpu().detach().numpy())

                #####################
                actions_clipped_transformed = self.transform_coordinate(actions_clipped) 
                target_xyz = actions_clipped_transformed
                current_xyz = self.robots.get_ee_cartesian_position()[:, :3].to(self.device)
                delta_xyz = target_xyz - current_xyz
                self.tvc_behavior.set_target_pose(delta_xyzs=delta_xyz.detach().cpu().numpy())

                i = 0
                print(f"start position control")
                while(not all(self.tvc_behavior.successes)):
                    all_actions = self.tvc_behavior.get_action()
                    all_actions = np.pad(all_actions, ((0, 0), (0, 2)), mode='constant') # add 0 velocities for the last 2 joints of the eef, 
                    actions_tensor = torch.from_numpy(all_actions.astype('float32')).to(self.device)    # convert to type float32.        
                    actions_tensor = actions_tensor.reshape(self.num_envs*self.num_dofs) # do the same action for all environments, then flatten to a 1D vector for compatibility.
                    actions_tensor = gymtorch.unwrap_tensor(actions_tensor)  
                    self.gym.set_dof_velocity_target_tensor(self.sim, actions_tensor)
                    i+=1     
                    self.gym.simulate(self.sim)
                    self.render()
                    self._refresh()
                self.tvc_behavior.successes = torch.tensor(self.num_envs*[False])
                print("finished pos control")

                self.rl_data_dict["actual_eef_poses"].append(torch.clone(self.states["eef_pos"]).cpu().detach().numpy())
            else:
                #### pos control with action=delta_xyz####
                print(f"action:{actions}")
                world_delta_xyz = torch.clone(actions)
                current_world_xyz = self.transform_coordinate(self.robots.get_ee_cartesian_position()[:, :3].to(self.device))
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

                robot_delta_xyz = self.transform_coordinate(world_delta_xyz)
                print(f"clipped action (world delta_xyz):{world_delta_xyz}")
                self.tvc_behavior.set_target_pose(delta_xyzs=robot_delta_xyz.detach().cpu().numpy())

                i = 0
                print(f"start position control")
                while(not all(self.tvc_behavior.successes)):
                    all_actions = self.tvc_behavior.get_action()
                    all_actions = np.pad(all_actions, ((0, 0), (0, 2)), mode='constant') # add 0 velocities for the last 2 joints of the eef, 
                    actions_tensor = torch.from_numpy(all_actions.astype('float32')).to(self.device)    # convert to type float32.        
                    actions_tensor = actions_tensor.reshape(self.num_envs*self.num_dofs) # do the same action for all environments, then flatten to a 1D vector for compatibility.
                    actions_tensor = gymtorch.unwrap_tensor(actions_tensor)  
                    self.gym.set_dof_velocity_target_tensor(self.sim, actions_tensor)
                    i+=1     
                    self.gym.simulate(self.sim)
                    self.render()
                    self._refresh()
                self.tvc_behavior.successes = torch.tensor(self.num_envs*[False])
                print("finished pos control")
        

    def post_physics_step(self):
        self.progress_buf += 1
        
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        
        self.compute_observations()
        self.compute_reward(self.actions)
        self.overlap = np.zeros((self.num_envs, MAX_NUM_BALLS))
        self.drop_ball()
        
        if len(env_ids) > 0:
            self.reset_idx(env_ids)