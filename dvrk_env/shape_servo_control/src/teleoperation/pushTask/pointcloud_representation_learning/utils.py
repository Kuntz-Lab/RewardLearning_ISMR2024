import numpy as np
import trimesh
import random

def sample_scene(idx, num_pts_per_object, radius):
    base_pos = np.array([0.0, -0.44, 0.16+0.015])

    if idx == 0:
        mesh = trimesh.creation.icosphere(radius=radius)
    # elif idx == 1:
    #     mesh = trimesh.creation.cone(radius = radius, height=0.08)


    random_pos = np.array(base_pos) + np.random.uniform(low=[-0.1,-0.03,-0.05], high=[0.1,0.02,0.05], size=3)
    T = trimesh.transformations.translation_matrix(random_pos)
    mesh.apply_transform(T)


    pc = trimesh.sample.sample_surface_even(mesh, count=num_pts_per_object)[0]

    return list(pc)

def sample_sphere_pc():
    base_pos = np.array([0.0, -0.44, 0.16+0.015])
    delta_x = np.random.uniform(low = -0.1 , high = 0.1)
    delta_y = np.random.uniform(low = -0.03 , high = 0.02)
    delta_z = np.random.uniform(low = -0.1 , high = 0.1)    
    # delta_x = np.random.uniform(low = -0.01 , high = 0.01)
    # delta_y = np.random.uniform(low = -0.03 , high = 0.02)
    # delta_z = np.random.uniform(low = -0.01 , high = 0.01)  
    random_pos = base_pos + np.array([delta_x, delta_y, delta_z])

    mesh = trimesh.creation.icosphere(radius=0.02)
    T = trimesh.transformations.translation_matrix(random_pos)
    mesh.apply_transform(T)

    pc = trimesh.sample.sample_surface_even(mesh, count=256)[0]

    return list(pc)

radius = 0.05
def sample_sphere_table_pc():
    base_pos = np.array([0.0, 0.0, 0.0])
    delta_x = np.random.uniform(low = -0.15 , high = 0.15)
    delta_y = np.random.uniform(low = -0.15 , high = 0.15)
    delta_z = radius#np.random.uniform(low = -0.1 , high = 0.1)    
    # delta_x = np.random.uniform(low = -0.2 , high = 0.2)
    # delta_y = np.random.uniform(low = -0.1 , high = 0.1)
    # delta_z = np.random.uniform(low = -0.2 , high = 0.2)   
    

    
    # mesh = trimesh.creation.icosphere(radius=np.random.uniform(low = 0.02 , high = 0.06))
    choice = random.choice(['sphere', 'cylinder', 'cone']) 
    if choice == "sphere":
        mesh = trimesh.creation.icosphere(radius=radius)
    elif choice == "cylinder":
        mesh = trimesh.creation.cylinder(radius = radius, height=0.2)
        delta_z = 0.1
    elif choice == "cone":
        mesh = trimesh.creation.cone(radius = radius, height=0.2)
        delta_z = 0

    random_pos = base_pos + np.array([delta_x, delta_y, delta_z])

    T = trimesh.transformations.translation_matrix(random_pos)
    mesh.apply_transform(T)

    pc = trimesh.sample.sample_surface_even(mesh, count=384)[0]
    # if pc.shape[0] != 256:
    #     return None

    return list(pc)


def sample_5_objects_pc():
    base_pos = np.array([0.0, 0.0, 0.0])
    # delta_x = np.random.uniform(low = -0.15 , high = 0.15)
    # delta_y = np.random.uniform(low = -0.15 , high = 0.15)
    # delta_z = radius#np.random.uniform(low = -0.1 , high = 0.1)    
    # delta_x = np.random.uniform(low = -0.2 , high = 0.2)
    # delta_y = np.random.uniform(low = -0.1 , high = 0.1)
    # delta_z = np.random.uniform(low = -0.2 , high = 0.2)   
    delta_x = np.random.uniform(low = -0.4 , high = 0.4)
    delta_y = np.random.uniform(low = -0.4 , high = 0.4)
    delta_z = 0#radius#np.random.uniform(low = -0.1 , high = 0.1)        
    

    
    # mesh = trimesh.creation.icosphere(radius=np.random.uniform(low = 0.02 , high = 0.06))
    choice = random.choice(['sphere', 'cylinder', 'cone']) 
    if choice == "sphere":
        mesh = trimesh.creation.icosphere(radius=radius)
    elif choice == "cylinder":
        mesh = trimesh.creation.cylinder(radius = radius, height=0.2)
        delta_z = 0.1
    elif choice == "cone":
        mesh = trimesh.creation.cone(radius = radius, height=0.2)
        delta_z = 0

    random_pos = base_pos + np.array([delta_x, delta_y, delta_z])

    T = trimesh.transformations.translation_matrix(random_pos)
    mesh.apply_transform(T)

    # pc = trimesh.sample.sample_surface_even(mesh, count=256)[0]
    # if pc.shape[0] != 256:
    #     return None

    pc = trimesh.sample.sample_surface_even(mesh, count=256)[0]
    if pc.shape[0] != 256:
        return None

    return list(pc)

radius = 0.05
num_pts_per_object = 512
num_pc_per_scene = 3 #5#3
def sample_3_objects_pc():
    base_pos = np.array([0.0, 0.0, 0.0])
    # delta_x = np.random.uniform(low = -0.15 , high = 0.15)
    # delta_y = np.random.uniform(low = -0.15 , high = 0.15)
    # delta_z = radius#np.random.uniform(low = -0.1 , high = 0.1)    
    delta_x = np.random.uniform(low = -0.2 , high = 0.2)
    delta_y = np.random.uniform(low = -0.2 , high = 0.2)
    delta_z = radius#np.random.uniform(low = -0.2 , high = 0.2)   
    # delta_x = np.random.uniform(low = -0.4 , high = 0.4)
    # delta_y = np.random.uniform(low = -0.4 , high = 0.4)
    # delta_z = 0#radius#np.random.uniform(low = -0.1 , high = 0.1)        
    

    
    # mesh = trimesh.creation.icosphere(radius=np.random.uniform(low = 0.02 , high = 0.06))
    choice = random.choice(['sphere', 'cylinder', 'cone']) 
    if choice == "sphere":
        mesh = trimesh.creation.icosphere(radius=radius)
    elif choice == "cylinder":
        mesh = trimesh.creation.cylinder(radius = radius, height=0.2)
        delta_z = 0.1
    elif choice == "cone":
        mesh = trimesh.creation.cone(radius = radius, height=0.2)
        delta_z = 0

    random_pos = base_pos + np.array([delta_x, delta_y, delta_z])

    T = trimesh.transformations.translation_matrix(random_pos)
    mesh.apply_transform(T)

    # pc = trimesh.sample.sample_surface_even(mesh, count=256)[0]
    # if pc.shape[0] != 256:
    #     return None

    pc = trimesh.sample.sample_surface_even(mesh, count=num_pts_per_object)[0]
    if pc.shape[0] != num_pts_per_object:
        return None

    return list(pc)




# def sample_sphere_pc():
#     base_pos = np.array([0.0, -0.44, 0.16+0.015])
#     delta_x = np.random.uniform(low = -0.1 , high = 0.1)
#     delta_y = np.random.uniform(low = -0.03 , high = 0.02)
#     delta_z = np.random.uniform(low = -0.1 , high = 0.1)    
#     # delta_x = np.random.uniform(low = -0.2 , high = 0.2)
#     # delta_y = np.random.uniform(low = -0.1 , high = 0.1)
#     # delta_z = np.random.uniform(low = -0.2 , high = 0.2)   
#     random_pos = base_pos + np.array([delta_x, delta_y, delta_z])

    
#     # mesh = trimesh.creation.icosphere(radius=np.random.uniform(low = 0.02 , high = 0.06))
#     choice = random.choice(['sphere', 'cylinder', 'cone']) 
#     if choice == "sphere":
#         mesh = trimesh.creation.icosphere(radius=0.02)
#     elif choice == "cylinder":
#         mesh = trimesh.creation.cylinder(radius = 0.02, height=0.06)
#     elif choice == "cone":
#         mesh = trimesh.creation.cone(radius = 0.02, height=0.06)

#     T = trimesh.transformations.translation_matrix(random_pos)
#     mesh.apply_transform(T)

#     pc = trimesh.sample.sample_surface_even(mesh, count=256)[0]

#     return list(pc)

def sample_three_pcs():
    def sample(delta_x, delta_y, delta_z):
        base_pos = np.array([0.0, -0.44, 0.16+0.015])
        # delta_x = np.random.uniform(low = -0.1 , high = 0.1)
        # delta_y = np.random.uniform(low = -0.03 , high = 0.02)
        # delta_z = np.random.uniform(low = -0.1 , high = 0.1)    
        random_pos = base_pos + np.array([delta_x, delta_y, delta_z])

        mesh = trimesh.creation.icosphere(radius=0.02)
        T = trimesh.transformations.translation_matrix(random_pos)
        mesh.apply_transform(T)

        pc = trimesh.sample.sample_surface_even(mesh, count=256)[0]

        return list(pc)    
    
    final_pc = []
    pc_1 = sample(*[-0.1,0,0.05])
    pc_2 = sample(*[0.1,0,0.05])
    pc_3 = sample(*[0.0,0,0.1])
    # pc_1 = sample(*[-0.01,0,0.005])
    # pc_2 = sample(*[0.01,0,0.005])
    # pc_3 = sample(*[0.0,0,0.01])
    final_pc = pc_1 + pc_2 + pc_3

    return final_pc

def sample_cone_box():
    base_pos = np.array([0.0, -0.0, 0.1])
    random_pos1 = base_pos + np.random.uniform(low=[-0.2,-0.6,0], high=[0.2,1,0], size=3)
    random_pos2 = base_pos + np.random.uniform(low=[-0.2,-0.6,0], high=[0.2,1,0], size=3)
    # random_pos1 = base_pos
    # random_pos2 = base_pos + np.array([-0.5, -100, -100])
    mesh1 = trimesh.creation.box(extents=[0.025,0.05, 0.04], transform=None)
    T1 = trimesh.transformations.translation_matrix(random_pos1)
    mesh1.apply_transform(T1)
    pc1 = trimesh.sample.sample_surface_even(mesh1, count=128)[0]

    mesh2 = trimesh.creation.cone(radius=0.05, height=0.05, sections=None, transform=None)
    T2 = trimesh.transformations.translation_matrix(random_pos2)
    mesh2.apply_transform(T2)
    pc2 = trimesh.sample.sample_surface_even(mesh2, count=128)[0]
    
    final_pc = list(pc1) + list(pc2)
    return final_pc
