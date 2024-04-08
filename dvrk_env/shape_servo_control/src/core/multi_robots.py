from isaacgym import gymapi
from isaacgym import gymtorch
import rospy
import numpy as np
from shape_servo_control.srv import *
import sys
import roslib.packages as rp
pkg_path = rp.get_pkg_dir('shape_servo_control')
sys.path.append(pkg_path + '/src')
from util.math_util import quaternion_to_matrix_batch, matrix_to_quaternion_batch, construct_homo_mat 
from copy import deepcopy
import torch

class MultiRobot:
    """Robot in isaacgym class - Bao"""

    def __init__(self, gym_handle, sim_handle, env_handles, robot_handles, n_arm_dof=8, num_envs=1):
        # Isaac Gym stuff
        self.gym = gym_handle
        self.sim = sim_handle
        self.envs = env_handles
        self.robots = robot_handles   
        self.n_arm_dof = n_arm_dof 
        self.num_envs = num_envs
        self.eef_handle = self.gym.find_actor_rigid_body_handle(self.envs[0], self.robots[0], "psm_tool_yaw_link")  # get handle for end-effector link

        link_names = ['world', 'psm_yaw_link', 'psm_pitch_back_link', 'psm_pitch_bottom_link', 'psm_pitch_end_link', 'psm_main_insertion_link', 'psm_tool_roll_link', 'psm_tool_pitch_link', 'psm_tool_yaw_link']
        self.links_handles = [self.gym.find_actor_rigid_body_handle(self.envs[0], self.robots[0], link_name)\
                            for link_name in link_names]    # get handles for all 9 links between base frame and eef frame
        self.links_handles = torch.tensor(self.links_handles, dtype=torch.long)


    def get_arm_joint_positions(self):
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        dof_state = gymtorch.wrap_tensor(dof_state_tensor).view(self.num_envs, -1, 2)[:, :8, 0].cpu().numpy().astype('float32')  
        # print("=======================xxxxxxxxxxxxxxxxxx dof_state.shape:", dof_state.shape)
        return deepcopy(dof_state)    # all robot dof states, shape (num_envs, 8)
        

    def get_ee_cartesian_position(self):
        """
        Return: end-effector catersian pose (num_envs, 7). 7 = x, y, z + 4 quaternion numbers.
        
        """
        rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)    
        ## gym.acquire_rigid_body_state_tensor(sim): Retrieves buffer for Rigid body states. The buffer has shape (num_rigid_bodies, 13). 
        ## State for each rigid body contains position([0:3]), rotation([3:7]), linear velocity([7:10]), and angular velocity([10:13]).
        
        rigid_body_state = gymtorch.wrap_tensor(rigid_body_state_tensor).view(self.num_envs, -1, 13)        
        
        eef_state = rigid_body_state[:, self.eef_handle, :]      
        eef_pose = eef_state[:,:7].cpu().numpy()  # 7 components: first 3 are the position x, y, z, the last 4 are quaternions
        eef_pose[:,:2] = -eef_pose[:,:2]     # flip signs of the x and y components

        return torch.from_numpy(eef_pose)    # all robot eef poses, shape (num_envs, 7)
    

    def get_all_links_poses(self):
        """
        Return: all 9 links' catersian poses, including the base frame (num_envs, 9, 7). Will be used to construct the 8 4x4 homo mats for jacobian matrix computation later on.
        
        """
        rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)    
        ## gym.acquire_rigid_body_state_tensor(sim): Retrieves buffer for Rigid body states. The buffer has shape (num_rigid_bodies, 13). 
        ## State for each rigid body contains position([0:3]), rotation([3:7]), linear velocity([7:10]), and angular velocity([10:13]).
        
        rigid_body_state = gymtorch.wrap_tensor(rigid_body_state_tensor).view(self.num_envs, -1, 13)        
        
        states = rigid_body_state[:, self.links_handles, :]      
        poses = states[:,:,:7].cpu().numpy()  # 7 components: first 3 are the position x, y, z, the last 4 are quaternions

        return poses    # shape (num_envs, 9, 7)   
        
    def get_all_links_homo_mats(self):
        """ 
        Construct (for each environment) all 8 4x4 homo matrices between the base frame and 8 other frames ('psm_yaw_link', 'psm_pitch_back_link',
        'psm_pitch_bottom_link', 'psm_pitch_end_link', 'psm_main_insertion_link', 'psm_tool_roll_link', 'psm_tool_pitch_link', 'psm_tool_yaw_link').
        
        1. Input: None
                
        2. Output:
        homo_mats (num_envs, 8, 4, 4): for each environment, 8 4x4 homo matrices
        """                

        fix_frame_1 = [0,1,2,3,4,5]
        fix_frame_2 = [6]
        fix_frame_3 = [7]
       
        all_poses = self.get_all_links_poses()
        all_poses = torch.from_numpy(all_poses)
        base_pose_all_envs = all_poses[:,0,:]    # shape (num_envs,7) base_pose = (0., 0., 0.25, 0., 0., 1., 0.) 
        link_poses_all_envs = all_poses[:,1:,:]  # shape (num_envs,8,7)
        
        num_envs = all_poses.shape[0]
        B = num_envs
        num_links = link_poses_all_envs.shape[1]
    
        # Create tensors for fix_frame_type based on link index
        fix_frame_type = torch.zeros(num_links, dtype=torch.long)
        fix_frame_type[fix_frame_1] = 1
        fix_frame_type[fix_frame_2] = 2
        fix_frame_type[fix_frame_3] = 3
        
        fix_frame_types = fix_frame_type.unsqueeze(0).expand(B, -1)  # Shape: (num_envs, num_links)

        base_poses = base_pose_all_envs.unsqueeze(1).expand(-1, num_links, -1)  # Shape: (num_envs, num_links, 7)
        link_poses = link_poses_all_envs  # Shape: (num_envs, num_links, 7)

        homo_mats = torch.zeros(B,num_links,4,4)
        homo_mats[:B,:,:,:] = construct_homo_mat(base_poses.reshape(B*num_links*7,),link_poses.reshape(B*num_links*7,),fix_frame_type).reshape(B,num_links,4,4)

        return homo_mats
        





































# from isaacgym import gymapi
# from isaacgym import gymtorch
# import rospy
# import numpy as np
# from shape_servo_control.srv import *
# import sys
# import roslib.packages as rp
# pkg_path = rp.get_pkg_dir('shape_servo_control')
# sys.path.append(pkg_path + '/src')
# from util.math_util import quaternion_to_matrix_batch, matrix_to_quaternion_batch, construct_homo_mat 
# from copy import deepcopy
# import torch

# class MultiRobot:
#     """Robot in isaacgym class - Bao"""

#     def __init__(self, gym_handle, sim_handle, env_handles, robot_handles, n_arm_dof=8, num_envs=1):
#         # Isaac Gym stuff
#         self.gym = gym_handle
#         self.sim = sim_handle
#         self.envs = env_handles
#         self.robots = robot_handles   
#         self.n_arm_dof = n_arm_dof 
#         self.num_envs = num_envs
#         self.eef_handle = self.gym.find_actor_rigid_body_handle(self.envs[0], self.robots[0], "psm_tool_yaw_link")  # get handle for end-effector link

#         link_names = ['world', 'psm_yaw_link', 'psm_pitch_back_link', 'psm_pitch_bottom_link', 'psm_pitch_end_link', 'psm_main_insertion_link', 'psm_tool_roll_link', 'psm_tool_pitch_link', 'psm_tool_yaw_link']
#         self.links_handles = [self.gym.find_actor_rigid_body_handle(self.envs[0], self.robots[0], link_name)\
#                             for link_name in link_names]    # get handles for all 9 links between base frame and eef frame
#         self.links_handles = torch.tensor(self.links_handles, dtype=torch.long)


#     def get_arm_joint_positions(self):
#         dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
#         dof_state = gymtorch.wrap_tensor(dof_state_tensor).view(self.num_envs, -1, 2)[:, :8, 0].cpu().numpy().astype('float32')  
#         # print("=======================xxxxxxxxxxxxxxxxxx dof_state.shape:", dof_state.shape)
#         return deepcopy(dof_state)    # all robot dof states, shape (num_envs, 8)
        

#     def get_ee_cartesian_position(self):
#         """
#         Return: end-effector catersian pose (num_envs, 7). 7 = x, y, z + 4 quaternion numbers.
        
#         """
#         rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)    
#         ## gym.acquire_rigid_body_state_tensor(sim): Retrieves buffer for Rigid body states. The buffer has shape (num_rigid_bodies, 13). 
#         ## State for each rigid body contains position([0:3]), rotation([3:7]), linear velocity([7:10]), and angular velocity([10:13]).
        
#         rigid_body_state = gymtorch.wrap_tensor(rigid_body_state_tensor).view(self.num_envs, -1, 13)        
        
#         eef_state = rigid_body_state[:, self.eef_handle, :]      
#         eef_pose = eef_state[:,:7].cpu().numpy()  # 7 components: first 3 are the position x, y, z, the last 4 are quaternions
#         eef_pose[:,:2] = -eef_pose[:,:2]     # flip signs of the x and y components

#         return eef_pose    # all robot eef poses, shape (num_envs, 7)
    

#     def get_all_links_poses(self):
#         """
#         Return: all 9 links' catersian poses, including the base frame (num_envs, 9, 7). Will be used to construct the 8 4x4 homo mats for jacobian matrix computation later on.
        
#         """
#         rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)    
#         ## gym.acquire_rigid_body_state_tensor(sim): Retrieves buffer for Rigid body states. The buffer has shape (num_rigid_bodies, 13). 
#         ## State for each rigid body contains position([0:3]), rotation([3:7]), linear velocity([7:10]), and angular velocity([10:13]).
        
#         rigid_body_state = gymtorch.wrap_tensor(rigid_body_state_tensor).view(self.num_envs, -1, 13)        
        
#         states = rigid_body_state[:, self.links_handles, :]      
#         poses = states[:,:,:7].cpu().numpy()  # 7 components: first 3 are the position x, y, z, the last 4 are quaternions

#         return poses    # shape (num_envs, 9, 7)   
        
#     def get_all_links_homo_mats(self):
#         """ 
#         Construct (for each environment) all 8 4x4 homo matrices between the base frame and 8 other frames ('psm_yaw_link', 'psm_pitch_back_link',
#         'psm_pitch_bottom_link', 'psm_pitch_end_link', 'psm_main_insertion_link', 'psm_tool_roll_link', 'psm_tool_pitch_link', 'psm_tool_yaw_link').
        
#         1. Input: None
                
#         2. Output:
#         homo_mats (num_envs, 8, 4, 4): for each environment, 8 4x4 homo matrices
#         """                

#         fix_frame_1 = [0,1,2,3,4,5]
#         fix_frame_2 = [6]
#         fix_frame_3 = [7]
       
#         all_poses = self.get_all_links_poses()        
#         base_pose_all_envs = all_poses[:,0,:]    # shape (num_envs,7) base_pose = (0., 0., 0.25, 0., 0., 1., 0.) 
#         link_poses_all_envs = all_poses[:,1:,:]  # shape (num_envs,8,7)
        
#         for i in range(num_envs):
#             # Example of env 0
#             base_pose = base_pose_all_envs[i,:]  # shape (7,)
#             link_poses = link_poses_all_envs[i,:,:]  # shape (8,7)
            
#             homo_mats = []
#             for i, link_pose in enumerate(link_poses):
#                 fix_frame_type = None
#                 if i in fix_frame_1:
#                     fix_frame_type = 1
#                 if i in fix_frame_2:
#                     fix_frame_type = 2    
#                 if i in fix_frame_3:
#                     fix_frame_type = 3  
#                 homo_mat = construct_homo_mat(base_pose, link_pose, fix_frame_type)
#                 homo_mats.append(homo_mat)     
            
#         # return np.array(homo_mats)  # shape (8,4,4)   
#         return np.array(all_homo_mats)  # shape (num_envs, 8,4,4) 
        






