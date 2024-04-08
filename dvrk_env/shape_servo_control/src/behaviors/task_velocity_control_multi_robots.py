import os
import sys
import numpy as np
from isaacgym import gymapi
import torch
import rospy
import random
from copy import deepcopy

import sys
import roslib.packages as rp
pkg_path = rp.get_pkg_dir('shape_servo_control')
sys.path.append(pkg_path + '/src')
from behaviors import Behavior
from util import ros_util, math_util, isaac_utils
from core import RobotAction

from shape_servo_control.srv import *
import rospy

two_robot_offset = 1.0



class TaskVelocityControlMultiRobot(Behavior):
    '''
    Implementation of resolved rate controller for dvrk robot. Move end-effector some x, y, z in Carteseian space.
    '''

    def __init__(self, delta_xyzs, robot, num_envs, dt, traj_duration, vel_limits=None, \
                        error_threshold = 1e-3):
        super().__init__()

        self.name = "task velocity control"
        self.robot = robot
        self.dt = dt
        self.traj_duration = traj_duration        
        self.err_thres = error_threshold
        self.dq = 10**-5 * np.ones(self.robot.n_arm_dof)
        self.vel_limits = torch.from_numpy(vel_limits)
        self.num_envs = num_envs
        self.vel_limits = self.vel_limits.unsqueeze(0).repeat(num_envs, 1)
        self.actions = torch.zeros((num_envs, 8))
        self.successes = torch.tensor(num_envs*[False])

        self.set_target_pose(delta_xyzs)

    def get_eef_vel_limits(self, device="cpu"):
        B = self.num_envs
        Ts = self.robot.get_all_links_homo_mats().to(device) # shape (B, 8, 4, 4)
        J = self.compute_jacobian_B(Ts).to(device) # shape (B, 6, 8)
        #J_pinv = self.damped_pinv(J).to(device) # shape (B, 8, 6)
        eef_vel_limits = torch.bmm(J, self.vel_limits[:B,:8].unsqueeze(-1)).squeeze(-1)  # shape (B, 8)
        return eef_vel_limits


    def get_action_from_eef_vel(self, eef_vel, device="cpu"):
        '''
        eef_vel has shape (B, 6): vel in x, y, z, #3 quaternion numbers.
        '''
        eef_vel = eef_vel.to(device)
        Ts = self.robot.get_all_links_homo_mats().to(device) # shape (B, 8, 4, 4)
        J = self.compute_jacobian_B(Ts).to(device) # shape (B, 6, 8)
        J_pinv = self.damped_pinv(J).to(device) # shape (B, 8, 6)
        q_vel = torch.bmm(J_pinv, eef_vel.unsqueeze(-1)).squeeze(-1)  # shape (B, 8)

        # scale the q_vel to clip the values
        # B = Ts.shape[0]
        # exceeding_ratios =torch.zeros(B,8)
        # self.vel_limits = self.vel_limits[:B,:8]
        # exceeding_ratios[:B,:] = torch.abs(torch.div(q_vel[:B,:], self.vel_limits[:B,:])) * (self.vel_limits is not None)
       
        # mask_ratios = torch.tensor(B*[False])
        # mask_ratios[:B] = torch.any(exceeding_ratios > 1.0, dim=1)
       
        # scale_factor = torch.ones(B)

        # scale_factor[:B] = torch.where(mask_ratios,torch.max(exceeding_ratios,dim=1).values,scale_factor) # Find the maximum exceeding ratio among the rows with mask=True
        # scale_factor = scale_factor.unsqueeze(1).repeat(1, 8)
        # q_vel = torch.div(q_vel , scale_factor)  # shape (B,8)
      
        return q_vel
        # print(f"J: {J.shape}")
        # print(f"J_pinv: {J_pinv.shape}")

    def get_action(self):

        if self.is_not_started():
            self.set_in_progress()

        B = self.num_envs
        ee_cartesian_pos = self.robot.get_ee_cartesian_position()   # shape (7,)
        
        target_pose = self.target_pose

        #compute the difference between end_effector and target(just distance not rotation)
        delta_ees = torch.sub(target_pose[:,:6] , ee_cartesian_pos[:,:6])   # shape (6,)
        delta_ees[:, 3:6] = 0   # shape (6,)
        abs_delta_ees = torch.abs(delta_ees)

        #masking out when the difference is bigger than error threshold 
        mask = torch.tensor(B*[False])
        #mask[:B] =  torch.any(abs_delta_ees[:B,:] > self.err_thres) #torch.max((abs_delta_ees[:B,:] > self.err_thres).int(), dim=1)==0#

        fail_idxs =  torch.where(abs_delta_ees[:B,:] > self.err_thres)
        mask[fail_idxs[0]] = True
       

        #computing the homo mats, jacobian, inverse of it.
        Ts = self.robot.get_all_links_homo_mats() # shape (B, 8, 4, 4)
        J = self.compute_jacobian_B(Ts)
        J_pinv = self.damped_pinv(J)
        
        q_vel = torch.bmm(J_pinv, delta_ees.unsqueeze(-1)).squeeze(-1)  # shape (B, 8)

        desired_q_vel = torch.zeros(B,8)
        desired_q_vel[:B,:] = torch.where(mask[:B].unsqueeze(-1), q_vel[:B,:], torch.zeros_like(q_vel))*8
    
        self.vel_limits = self.vel_limits[:B,:8]

        exceeding_ratios =torch.zeros(B,8)
        exceeding_ratios[:B,:] = torch.abs(torch.div(desired_q_vel[:B,:], self.vel_limits[:B,:])) * (self.vel_limits is not None)
       
        mask_ratios = torch.tensor(B*[False])
        mask_ratios[:B] = torch.any(exceeding_ratios > 1.0, dim=1)
       
        scale_factor = torch.ones(B)
        
        scale_factor[:B] = torch.where(mask_ratios,torch.max(exceeding_ratios,dim=1).values,scale_factor) # Find the maximum exceeding ratio among the rows with mask=True
        scale_factor = scale_factor.unsqueeze(1).repeat(1, 8)
        desired_q_vel = torch.div(desired_q_vel , scale_factor)  # shape (B,8)
      

        self.successes[:B] = torch.where(
            mask,
            torch.tensor(False),
            torch.tensor(True)
        )
        
        zero_actions = torch.zeros(B,8)

        self.actions[:B,:] = torch.where(
            mask[:B].unsqueeze(-1),
            desired_q_vel[:B,:],
            zero_actions[:B,:]
        )

        return self.actions

    def set_target_pose(self, delta_xyzs):
        pose = self.robot.get_ee_cartesian_position()
        # pose[:,:2] *= -1
        pose[:,:3] += np.array(delta_xyzs) 
        self.target_pose = pose

    def damped_pinv(self, A, rho=0.017):
        B = A.shape[0]
        AA_T = torch.bmm(A,A.transpose(1,2))
        damping = torch.eye(A.shape[1])* rho**2
        inv = torch.linalg.inv((AA_T + damping))
        d_pinv = torch.bmm(A.transpose(1,2), inv)
        return d_pinv

    def null_space_projection(self, q_cur, q_vel, J, J_pinv):
        identity = np.identity(self.robot.n_arm_dof)
        q_vel_null = \
            self.compute_redundancy_manipulability_resolution(q_cur, q_vel, J)
        q_vel_constraint = np.array(np.matmul((
            identity - np.matmul(J_pinv, J)), q_vel_null))[0]
        q_vel_proj = q_vel + q_vel_constraint
        return q_vel_proj    

    def compute_redundancy_manipulability_resolution(self, q_cur, q_vel, J):
        m_score = self.compute_manipulability_score(J)
        J_prime = self.get_pykdl_client(q_cur + self.dq)
        m_score_prime = self.compute_manipulability_score(J_prime)
        q_vel_null = (m_score_prime - m_score) / self.dq
        return q_vel_null

    def compute_manipulability_score(self, J):
        return np.sqrt(np.linalg.det(np.matmul(J, J.transpose())))    
    
    def compute_jacobian_B(self,Ts):
        """ 
        Compute Jacobian matrix using the equation at 4:31 of this Youtube video: https://www.youtube.com/watch?v=C6Zho88S8vY
        
        1. Input:
        Ts: a list of 9 4x4 homogenous transformation matrices that transform the base frame to all other frames along the kinematic chain.
        
        2. Output:
        J: Jacobian matrix. Shape (6,8).
        
        """
        num_joints = 8
        # B = int(Ts.shape[0]/(num_joints))
        B = Ts.shape[0]
        New_Ts = Ts.view(B,8,4,4)
        J = torch.zeros(B, 6, num_joints)

        t_eef = New_Ts[:, 7, :3, 3]
        t_eef = t_eef.unsqueeze(1).repeat(1, 8, 1)
        t_i = New_Ts[:, :num_joints, :3, 3]
        d_i_n = t_eef - t_i
        z_i = New_Ts[:, :num_joints, :3, 2]

        mask = torch.ones_like(z_i)
        mask[:, 4] = 0  # prismatic joint mask

        J[:, :3, :] = ((torch.cross(z_i, d_i_n) * mask) + (z_i * (1-mask))).transpose(1,2)
        J[:, 3:, :] = (z_i * (mask)).transpose(1,2)

        #the weird reasons hear too, for two joints
        # Negate the first and seventh columns of J
        J[:, :, 0] *= -1
        J[:, :, 6] *= -1   

        return J # size of j is Bx6x8

    









































# import os
# import sys
# import numpy as np
# from isaacgym import gymapi
# import torch
# import rospy
# import random
# from copy import deepcopy

# import sys
# import roslib.packages as rp
# pkg_path = rp.get_pkg_dir('shape_servo_control')
# sys.path.append(pkg_path + '/src')
# from behaviors import Behavior
# from util import ros_util, math_util, isaac_utils
# from core import RobotAction

# from shape_servo_control.srv import *
# import rospy

# two_robot_offset = 1.0



# class TaskVelocityControlMultiRobot(Behavior):
#     '''
#     Implementation of resolved rate controller for dvrk robot. Move end-effector some x, y, z in Carteseian space.
#     '''

#     def __init__(self, delta_xyzs, robot, num_envs, dt, traj_duration, vel_limits=None, \
#                         error_threshold = 1e-3):
#         super().__init__()

#         self.name = "task velocity control"
#         self.robot = robot
#         self.dt = dt
#         self.traj_duration = traj_duration        
#         self.err_thres = error_threshold
#         self.dq = 10**-5 * np.ones(self.robot.n_arm_dof)
#         self.vel_limits = vel_limits
#         self.num_envs = num_envs
        
#         self.actions = np.zeros((num_envs, 8))   #[RobotAction() for _ in range(num_envs)]
#         self.successes = [False for _ in range(num_envs)]

#         self.set_target_pose(delta_xyzs)


#     def get_action(self):

#         # if self.is_not_started():
#         #     self.set_in_progress()

#         q_curs = self.robot.get_arm_joint_positions()   # shape (8,)
#         ee_cartesian_pos = self.robot.get_ee_cartesian_position()   # shape (7,)
#         # ee_cartesian_pos[:,:2] *= -1

#         delta_ees = self.target_pose[:,:6] - ee_cartesian_pos[:,:6]   # shape (6,)
#         delta_ees[:, 3:6] = 0   # shape (6,)
        
#         for i in range(self.num_envs):
       
#             if np.any(abs(delta_ees[i]) > self.err_thres):
                
#                 J = self.get_pykdl_client(q_curs[i])   # shape (6,8)
#                 J_pinv = self.damped_pinv(J)
#                 q_vel = np.matmul(J_pinv, delta_ees[i])   # shape (8,)

#                 desired_q_vel = q_vel #* 4
#                 if self.vel_limits is not None:
#                     exceeding_ratios = abs(np.divide(desired_q_vel, self.vel_limits[:8]))
#                     if np.any(exceeding_ratios > 1.0):
#                         scale_factor = max(exceeding_ratios)
#                         desired_q_vel /= scale_factor   # shape (8,)
#                 self.actions[i] = desired_q_vel
              

#             else:
#                 self.successes[i] = True
#                 # self.set_success()
#                 self.actions[i] = np.zeros(8)

#         return self.actions

#     def set_target_pose(self, delta_xyzs):
#         pose = self.robot.get_ee_cartesian_position()
#         # pose[:,:2] *= -1
#         pose[:,:3] += np.array(delta_xyzs) 
#         self.target_pose = pose

#     def damped_pinv(self, A, rho=0.017):
#         AA_T = np.dot(A, A.T)
#         damping = np.eye(A.shape[0]) * rho**2
#         inv = np.linalg.inv(AA_T + damping)
#         d_pinv = np.dot(A.T, inv)
#         return d_pinv

#     def null_space_projection(self, q_cur, q_vel, J, J_pinv):
#         identity = np.identity(self.robot.n_arm_dof)
#         q_vel_null = \
#             self.compute_redundancy_manipulability_resolution(q_cur, q_vel, J)
#         q_vel_constraint = np.array(np.matmul((
#             identity - np.matmul(J_pinv, J)), q_vel_null))[0]
#         q_vel_proj = q_vel + q_vel_constraint
#         return q_vel_proj    

#     def compute_redundancy_manipulability_resolution(self, q_cur, q_vel, J):
#         m_score = self.compute_manipulability_score(J)
#         J_prime = self.get_pykdl_client(q_cur + self.dq)
#         m_score_prime = self.compute_manipulability_score(J_prime)
#         q_vel_null = (m_score_prime - m_score) / self.dq
#         return q_vel_null

#     def compute_manipulability_score(self, J):
#         return np.sqrt(np.linalg.det(np.matmul(J, J.transpose())))    

#     def get_pykdl_client(self, q_cur):
#         '''
#         get Jacobian matrix
#         '''
#         # rospy.loginfo('Waiting for service get_pykdl.')
#         # rospy.wait_for_service('get_pykdl')
#         # rospy.loginfo('Calling service get_pykdl.')
#         try:
#             pykdl_proxy = rospy.ServiceProxy('get_pykdl', PyKDL)
#             pykdl_request = PyKDLRequest()
#             pykdl_request.q_cur = q_cur
#             pykdl_response = pykdl_proxy(pykdl_request) 
        
#         except(rospy.ServiceException, e):
#             rospy.loginfo('Service get_pykdl call failed: %s'%e)
#         # rospy.loginfo('Service get_pykdl is executed.')    
        
#         return np.reshape(pykdl_response.jacobian_flattened, tuple(pykdl_response.jacobian_shape))      