from isaacgym import gymtorch
from isaacgym import gymapi
import torch
import numpy as np
import open3d
import timeit
from compute_partial_pc import compute_pointcloud

def get_partial_pointcloud_vectorized(gym, sim, env, cam_handle, cam_prop, color=None, min_z=0.005, visualization=False, device="cuda", min_depth=-3):
    '''
    Remember to render all camera sensors before calling thsi method in isaac gym simulation
    '''
    cam_width = cam_prop.width
    cam_height = cam_prop.height
    depth_buffer = gym.get_camera_image(sim, env, cam_handle, gymapi.IMAGE_DEPTH)
    seg_buffer = gym.get_camera_image(sim, env, cam_handle, gymapi.IMAGE_SEGMENTATION)
    vinv = np.linalg.inv(np.matrix(gym.get_camera_view_matrix(sim, env, cam_handle)))
    proj = gym.get_camera_proj_matrix(sim, env, cam_handle)

    # compute pointcloud
    D_i = torch.tensor(depth_buffer.astype('float32') )
    S_i = torch.tensor(seg_buffer.astype('float32') )
    V_inv = torch.tensor(vinv.astype('float32') )
    P = torch.tensor(proj.astype('float32') )
    
    points = compute_pointcloud(D_i, S_i, V_inv, P, cam_width, cam_height, min_z, device, min_depth=min_depth)
    
    if visualization:
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(np.array(points))
        if color is not None:
            pcd.paint_uniform_color(color) # color: list of len 3
        open3d.visualization.draw_geometries([pcd]) 

    return points

def get_partial_pointcloud_vectorized_push_box(gym, sim, env, cam_handle, vinv, proj, cam_prop, color=None, min_z=0.005, visualization=False, device="cuda"):
    '''
    Remember to render all camera sensors before calling thsi method in isaac gym simulation
    There is a problem that only env 0 has the correct camera view matrix, so I changed the arguments of the method
    '''
    cam_width = cam_prop.width
    cam_height = cam_prop.height
    depth_buffer = gym.get_camera_image(sim, env, cam_handle, gymapi.IMAGE_DEPTH)
    seg_buffer = gym.get_camera_image(sim, env, cam_handle, gymapi.IMAGE_SEGMENTATION)
    # vinv = np.linalg.inv(np.matrix(gym.get_camera_view_matrix(sim, env, cam_handle)))
    # proj = gym.get_camera_proj_matrix(sim, env, cam_handle)

    # compute pointcloud
    D_i = torch.tensor(depth_buffer.astype('float32') )
    S_i = torch.tensor(seg_buffer.astype('float32') )
    V_inv = torch.tensor(vinv.astype('float32') )
    P = torch.tensor(proj.astype('float32') )
    
    points = compute_pointcloud(D_i, S_i, V_inv, P, cam_width, cam_height, min_z, device)
    
    if visualization:
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(np.array(points))
        if color is not None:
            pcd.paint_uniform_color(color) # color: list of len 3
        open3d.visualization.draw_geometries([pcd]) 

    return points


def get_partial_pointcloud_vectorized_seg_ball(gym, sim, env, cam_handle, vinv, proj, cam_prop, dropped_mask, ball_seg_Ids, color=None, min_z=0.005, visualization=False, device="cuda", min_depth=-3):
    '''
    Remember to render all camera sensors before calling thsi method in isaac gym simulation
    '''
    cam_width = cam_prop.width
    cam_height = cam_prop.height
    depth_buffer = gym.get_camera_image(sim, env, cam_handle, gymapi.IMAGE_DEPTH)
    seg_buffer = gym.get_camera_image(sim, env, cam_handle, gymapi.IMAGE_SEGMENTATION)
    
    # compute pointcloud
    D_i = torch.tensor(depth_buffer.astype('float32') )
    S_i = torch.tensor(seg_buffer.astype('float32') )
    V_inv = torch.tensor(vinv.astype('float32') )
    P = torch.tensor(proj.astype('float32') )

    # print(f"depth buffer: {D_i}")
    # print(f"seg buffer: {S_i}")
    # print(f"proj: {P}")
    # print(f"vinv: {vinv}")
    # print(f"seg ids: {ball_seg_Ids}")
    # print(f"dropped mask: {dropped_mask}")


    for j, seg_Id in enumerate(ball_seg_Ids):
        if dropped_mask[j]==1:
            D_i[S_i==seg_Id] = -10001
    
    points = compute_pointcloud(D_i, S_i, V_inv, P, cam_width, cam_height, min_z, device, min_depth=min_depth)
    
    if visualization:
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(np.array(points))
        if color is not None:
            pcd.paint_uniform_color(color) # color: list of len 3
        open3d.visualization.draw_geometries([pcd]) 

    return points


def get_partial_pointcloud_vectorized_cut(gym, sim, env, cam_handle, vinv, proj, cam_prop, color=None, min_z=0.005, visualization=False, device="cuda", min_depth=-1):
    '''
    Remember to render all camera sensors before calling thsi method in isaac gym simulation
    '''
    cam_width = cam_prop.width
    cam_height = cam_prop.height
    depth_buffer = gym.get_camera_image(sim, env, cam_handle, gymapi.IMAGE_DEPTH)
    seg_buffer = gym.get_camera_image(sim, env, cam_handle, gymapi.IMAGE_SEGMENTATION)
    
    # compute pointcloud
    D_i = torch.tensor(depth_buffer.astype('float32') )
    S_i = torch.tensor(seg_buffer.astype('float32') )
    V_inv = torch.tensor(vinv.astype('float32') )
    P = torch.tensor(proj.astype('float32') )

    # print(f"depth buffer: {D_i}")
    # print(f"seg buffer: {S_i}")
    # print(f"proj: {P}")
    # print(f"vinv: {vinv}")
    # print(f"seg ids: {ball_seg_Ids}")
    # print(f"dropped mask: {dropped_mask}")
    
    points = compute_pointcloud(D_i, S_i, V_inv, P, cam_width, cam_height, min_z, device, min_depth=min_depth)
    
    if visualization:
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(np.array(points))
        if color is not None:
            pcd.paint_uniform_color(color) # color: list of len 3
        open3d.visualization.draw_geometries([pcd]) 

    return points




if __name__ == "__main__":
    w = 300
    h = 300
    D_i = torch.rand(h, w)
    S_i = torch.rand(h, w)
    V_inv = torch.eye(4)
    P = torch.rand(4, 4)
    start_time = timeit.default_timer()
    for i in range(1000):  
        points = compute_pointcloud(D_i, S_i, V_inv, P, w, h, min_z=0.05, device="cuda")
    print("Elapsed time (s): ", timeit.default_timer() - start_time)
    print("shape: ", points.shape)
    