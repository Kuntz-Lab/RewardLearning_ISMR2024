import torch
import numpy as np
# import open3d
import timeit

def compute_pointcloud(D_i, S_i, V_inv, P, w, h, min_z, device="cuda", min_depth=-3):
    '''
    All matrices should be torch tensor: 

    D_i = depth buffer for env i (h x w)
    S_i = segmentation buffer for env i (h x w)
    V_inv = inverse of camera view matrix (4 x 4)
    P = camera projection matrix (4 x 4)
    w = width of camera 
    h = height of camera
    min_z = the lowest z value allowed
    '''
    D_i = D_i.to(device)
    S_i = S_i.to(device)
    V_inv = V_inv.to(device)
    P = P.to(device)

    D_i[S_i==11] = -10001  # segment out the robot

    fu = 2/P[0,0]
    fv = 2/P[1,1]

    center_u = w/2
    center_v = h/2

    # pixel indices
    k = torch.arange(0, w).unsqueeze(0) # shape = (1, w)
    t = torch.arange(0, h).unsqueeze(1) # shape = (h, 1)
    K = k.expand(h, -1).to(device) # shape = (h, w)
    T = t.expand(-1, w).to(device) # shape = (h, w)

    ### forgot minus sign here U = -(K - center_u)/w
    U = -(K - center_u)/w # image-space coordinate
    V = (T - center_v)/h # image-space coordinate

    X2 = torch.cat([(fu*D_i*U).unsqueeze(0), (fv*D_i*V).unsqueeze(0), D_i.unsqueeze(0), torch.ones_like(D_i).unsqueeze(0).to(device)], dim=0) # deprojection vector, shape = (4, h, w)
    X2 = X2.permute(1,2,0).unsqueeze(2) # shape = (h, w, 1, 4)
    V_inv = V_inv.unsqueeze(0).unsqueeze(0).expand(h, w, 4, 4) # shape = (h, w, 4, 4)
    #print(f"Vinv: {V_inv[0,0,:,:]}")
    # Inverse camera view to get world coordinates
    #print(f"X2: {X2[4,4,:,:]}")
    P2 = torch.matmul(X2, V_inv) # shape = (h, w, 1, 4)
    #print(P2.shape)
    #print(f"P2: {P2[4,4,:,:]}")
    
    # filter out low points and get the remaining points
    points = P2.reshape(-1, 4)
    #print("len points: ", len(points))
    depths = D_i.reshape(-1)
    mask = (depths >= min_depth) # 3 for 2 ball # -1 for cutting
    points = points[mask, :]
    #print("len points after mask: ", len(points))
    mask = (points[:, 2]>min_z)
    points = points[mask, :]
    #print("len points after 2nd mask: ", len(points))
    
    return points[:, :3].cpu().numpy().astype('float32') 


def farthest_point_sample_batched(point, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, D] where B= num batches, N=num points, D=point dim (typically D=3)
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint, D]
    """
    B, N, D = point.shape
    xyz = point[:, :, :3]
    centroids = np.zeros((B, npoint))
    distance = np.ones((B, N)) * 1e10
    farthest = np.random.uniform(low=0, high=N, size=(B,)).astype(np.int32) #np.random.randint(0, N)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[np.arange(0, B), farthest, :] # (B, D)
        centroid = np.expand_dims(centroid, axis=1) # (B, 1, D)
        dist = np.sum((xyz - centroid) ** 2, -1) # (B, N)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1) # (B,)
    point = point[np.arange(0, B).reshape(-1, 1), centroids.astype(np.int32), :]
    return point

def get_all_bin_seq_driver(length):
    results = set()
    seq = [0 for i in range(length)]
    get_all_bin_seq(seq, 0, results, length)
    return results

def get_all_bin_seq(seq, i, results, length):
    if i == length:
        return 
    new_seq = seq[:]
    for num in [0,1]:
        new_seq[i] = num
        results.add(tuple(new_seq))
        get_all_bin_seq(new_seq, i+1, results, length)

# def get_extrinsic_intrinsic(cam_view_mat, cam_proj_mat):
#     extr = cam_view_mat.T.astype(np.float64)
#     fu = 2/cam_proj_mat[0,0]
#     fv = 2/cam_proj_mat[1,1]
#     intr = np.array([[fu,0,300/2], \
#                     [0,fv,300/2],  \
#                     [0,0,1]]).astype(np.float64)
#     return extr,intr




if __name__ == "__main__":
    torch.random.manual_seed(2021)
    w = 10
    h = 10
    D_i = torch.rand(h, w)
    S_i = torch.rand(h, w)
    V_inv = torch.rand(4,4)#torch.eye(4)
    P = torch.rand(4, 4)
    start_time = timeit.default_timer()
    for i in range(1):  
        points = compute_pointcloud(D_i, S_i, V_inv, P, w, h, min_z=0.05, device="cuda")
    print("Elapsed time (s): ", timeit.default_timer() - start_time)
    print("shape: ", points.shape)

    # print("verify mat mul correctness: ")
    # X2 = torch.tensor([[ 0.2068, -0.4045,  0.7421,  1.0000]])
    # print(f"!!!!!!!!!!{torch.matmul(X2, V_inv)}")

    print(f"max xyz in P: x{np.max(points[:, 0])} y{np.max(points[:, 1])} z{np.max(points[:, 2])}")
    print(f"min xyz in P: x{np.min(points[:, 0])} y{np.min(points[:, 1])} z{np.min(points[:, 2])}")

    new_points = farthest_point_sample_batched(points[np.newaxis,:], 100)

    new_points = new_points[0]
    print(f"max xyz in new P: x{np.max(points[:, 0])} y{np.max(points[:, 1])} z{np.max(points[:, 2])}")
    print(f"min xyz in new P: x{np.min(points[:, 0])} y{np.min(points[:, 1])} z{np.min(points[:, 2])}")



    # results = get_all_bin_seq_driver(1)
    # print(results)
    # print(tuple([0 for i in range(2)]))
    
    