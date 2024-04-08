import open3d
import numpy as np
import torch
import pickle
import trimesh
import torch.nn.functional as F
from utils import *
from architecture import AutoEncoder
from farthest_point_sampling import *
import os
# np.random.seed(0)
def down_sampling(pc, num_pts=1024):
    farthest_indices,_ = farthest_point_sampling(pc, num_pts)
    pc = pc[farthest_indices.squeeze()]  
    return pc

device = torch.device("cuda")
model = AutoEncoder(num_points=256, embedding_size=256).to(device)
#model = AutoEncoder(num_points=512*3, embedding_size=1024).to(device)
model.load_state_dict(torch.load("/home/dvrk/LfD_data/ex_AE_boxCone/weights/weights_1/epoch_150"))
# model.load_state_dict(torch.load("/home/baothach/shape_servo_data/teleoperation/sanity_check_examples/ex_3/point_cloud_dataset/pc_embedding_weights/weights_single_ball_partial_1/epoch 150"))
model.eval()

for k in range(10):
    # base_pos = np.array([0.0, -0.44, 0.16+0.015])
    # delta_x = np.random.uniform(low = -0.1 , high = 0.1)
    # delta_y = np.random.uniform(low = -0.03 , high = 0.02)
    # delta_z = np.random.uniform(low = -0.1 , high = 0.1)    

    # random_pos = base_pos + np.array([delta_x, delta_y, delta_z])

    # mesh = trimesh.creation.icosphere(radius=0.02)#0.02)
    # # mesh = trimesh.intersections.slice_mesh_plane(mesh=mesh, plane_normal=[0,0,1], plane_origin=[0,0,0.0], cap=True)
    # # mesh = trimesh.creation.box((0.1, 0.1, 0.04))
    # T = trimesh.transformations.translation_matrix(random_pos)
    # mesh.apply_transform(T)
    # points = trimesh.sample.sample_surface_even(mesh, count=256)[0]

    # final_pc = []
    # for j in range(3):
    #     final_pc.extend(sample_sphere_pc())

    # mesh_table = trimesh.creation.box((0.3, 0.3, 0.01))
    # pc_table = list(trimesh.sample.sample_surface(mesh_table, count=256)[0])
    # final_pc = []
    # # final_pc.extend(pc_table)
    # for j in range(2):
    #     pc = sample_sphere_table_pc()  
    #     final_pc.extend(pc)

    final_pc = []
    dataset_path = "/home/dvrk/LfD_data/ex_AE_boxCone/processed_data_test"
    with open(os.path.join(dataset_path, "processed sample " + str(np.random.randint(0,100)) + ".pickle"), 'rb') as handle:
        data = pickle.load(handle)   
    #final_pc.extend(pc_table)
    # for j in range(3):
    #     pc = sample_3_objects_pc()  
    #     final_pc.extend(pc)

    #final_pc.extend(sample_cone_box())
    # print(data["partial_pc"].shape)
    final_pc.extend(list(data["partial_pc"]))

    # final_pc = []
    # for j in range(1):
    #     pc = sample_scene(j, num_pts_per_object=256, radius=0.02) 
    #     final_pc.extend(pc)

    # data_recording_path = "/home/baothach/shape_servo_data/teleoperation/sanity_check_examples/ex_3/point_cloud_dataset/partial_pc_single_ball_test"
    # file = os.path.join(data_recording_path, f"sample {k}.pickle")
    # with open(file, 'rb') as handle:
    #     data = pickle.load(handle)   
    #     final_pc = data["partial_pc"] 


    # final_pc = []
    # # final_pc.extend(pc_table)
    # for j in range(5):
    #     pc = sample_5_objects_pc()  
    #     final_pc.extend(pc)

    # final_pc = sample_three_pcs()[100:356]

    # # mesh = trimesh.load("/home/baothach/sim_data/Custom/Custom_mesh/kidney/Ginjal New splified_3.obj")
    # # mesh = trimesh.load("/home/baothach/dvrk_ws/src/dvrk_env/shape_servo_control/src/teleoperation/random_stuff/obstacle_1.obj")
    # r_max = 0.4#np.random.uniform(low = 0.2 , high = 0.4)
    # r_min = 0.3 * r_max #np.random.uniform(low = 0.1 , high = 0.95) * r_max
    # mesh = trimesh.creation.annulus(r_min=r_min, r_max=r_max, height=0.1)
    # mesh = trimesh.intersections.slice_mesh_plane(mesh=mesh, plane_normal=[0,-1,0], plane_origin=[0,0.02,0], cap=True)
    # base_pos = np.array([0.0, -0.44, 0.16+0.015])
    # delta_x = np.random.uniform(low = -0.1 , high = 0.1)
    # delta_y = np.random.uniform(low = -0.03 , high = 0.02)
    # delta_z = np.random.uniform(low = -0.1 , high = 0.1)    

    # random_pos = base_pos + np.array([delta_x, delta_y, delta_z])

    # T = trimesh.transformations.translation_matrix(random_pos)
    # mesh.apply_transform(T)
    # # mesh.apply_transform(trimesh.transformations.random_rotation_matrix())

    # final_pc = trimesh.sample.sample_surface_even(mesh, count=512)[0]

    points = np.array(final_pc)
    print(points.shape)
    # points = np.array(points)
    # points = down_sampling(np.array(final_pc), num_pts=256)

    points = points[np.random.permutation(points.shape[0])]
    # reconstructed_points = data[0]

    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(np.array(points))  
    pcd.paint_uniform_color([0, 1, 0])

    # pcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=8))   
    # normals = np.asarray(pcd.normals)
    # processed_points = np.concatenate((points, normals), axis = 1)
    # processed_points = points
    # print(processed_points.shape)

    points_tensor = torch.from_numpy(points.transpose(1,0)).unsqueeze(0).float().to(device)
    reconstructed_points = model(points_tensor)
    # loss = F.mse_loss(points_tensor.permute(0,2,1), reconstructed_points.permute(0,2,1))
    # print("loss:", loss)
    # print(reconstructed_points.shape)
    reconstructed_points = np.swapaxes(reconstructed_points.squeeze().cpu().detach().numpy(),0,1)
    reconstructed_points = reconstructed_points[:,:3]
    print(reconstructed_points.shape)
    # print(reconstructed_points)
    pcd2 = open3d.geometry.PointCloud()
    pcd2.points = open3d.utility.Vector3dVector(np.array(reconstructed_points)) 
    pcd2.paint_uniform_color([1, 0, 0])
    open3d.visualization.draw_geometries([pcd, pcd2.translate((0,0,0.1))])  
    #open3d.visualization.draw_geometries([pcd])  
    # open3d.visualization.draw_geometries([pcd, pcd2.translate((0,0,0.1))]) 

    # coor = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    # open3d.visualization.draw_geometries([pcd, pcd2, coor])

    # avg_pc = np.mean(final_pc, axis=0)
    # print(avg_pc.shape)
    # center = open3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    # open3d.visualization.draw_geometries([pcd, pcd2, center.translate(tuple(avg_pc))])
