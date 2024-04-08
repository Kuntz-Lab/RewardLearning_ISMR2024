#!/usr/bin/env python3
#from __future__ import print_function, division, absolute_import

import numpy as np
import trimesh
import open3d

# import sys
# import roslib.packages as rp
# pkg_path = rp.get_pkg_dir('shape_servo_control')
# sys.path.append(pkg_path + '/src')
# from util.isaac_utils import *
import numpy as np

'''
sample coordinates and pointclouds of balls along some curve
'''

LOW_X = -0.1
UP_X = 0.1
LOW_Y = -0.6
UP_Y = -0.4
LOW_Z = 0.011
UP_Z = 0.2

def is_in_workspace(x, y, z):
    low_x = x>=LOW_X
    up_x = x<=UP_X
    up_y = y<=UP_Y
    low_y = y>=LOW_Y
    low_z = z>=LOW_Z
    up_z = z<=UP_Z

    return low_x and up_x and up_y and low_y and low_z and up_z

def get_balls_xyz_around_workspace(LOW_X, UP_X, LOW_Y, UP_Y):
    segment_1 = np.linspace(start=[LOW_X, UP_Y, 0.1], stop=[UP_X, UP_Y, 0.1], num=10)
    segment_2 = np.linspace(start=[UP_X, UP_Y, 0.1], stop=[UP_X, LOW_Y, 0.1], num=10)
    segment_3 = np.linspace(start=[UP_X, LOW_Y, 0.1], stop=[LOW_X, LOW_Y, 0.1], num=10)
    segment_4 = np.linspace(start=[LOW_X, LOW_Y, 0.1], stop=[LOW_X, UP_Y, 0.1], num=10)

    boundary = np.concatenate((segment_1, segment_2, segment_3, segment_4), axis=0) # also concat np.array([[0,0,0.1], [0.2,0,0.1], [-0.2,0,0.1], [0,-0.7,0.1], [0,-0.3,0.1], [-0.2,-0.5,0.1], [0.2,-0.5,0.1]])
    return boundary

def is_overlap(p1, p2, max_dist=0.00025):
    return (p1.p.x-p2.p.x)**2 + (p1.p.y-p2.p.y)**2 + (p1.p.z-p2.p.z)**2 <=max_dist
    #return np.allclose(np.array([p1.p.x, p1.p.y, p1.p.z]), np.array([p2.p.x, p2.p.y, p2.p.z]), rtol=0, atol=0.058)

def poly2D(x, weights):
    '''
    y = f(x) polynomial
    '''
    y = 0
    for i, w in enumerate(weights):
        y += w*(x**i)
    return y

def poly3D(x, y, weights_list):
    '''
    # z = g(x,y) polynomial
    '''
    degree = len(weights_list)-1
    z = 0
    for n in range(degree+1):
        for k, w in enumerate(weights_list[n]):
            z += w*(x**n-k)*(y**k)
    return z

def get_rand_weights_list(degree):
    '''
    weights_list[i] are the coefficients of the product of i variables
    1) len(weights_list)==degree+1
    2) for i in range(degree+1):
           assert(len(weights_list[i])==i+1)
    '''
    weights_list = []
    for i in range(degree+1):
        rand_i = np.random.uniform(low=-0.1, high=0.1, size=i+1)
        weights_list.append(rand_i)
    #weights_list[0][0] += 0.03

    return weights_list

def visualize_poly3D(weights_list):
    has_neg = False
    num_samples = 2000
    final_pc = []
    for sample in range(num_samples):
        x = np.random.uniform(low=-0.1, high=0.1)
        y = np.random.uniform(low=-0.4, high=-0.6)
        z = poly3D(x, y, weights_list)
        if z<0:
            has_neg = True
        pc = get_ball_pc(np.array([x, y, z]), num_pts_per_ball=256, radius=0.005)
        final_pc.extend(pc)
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(final_pc)
    open3d.visualization.draw_geometries([pcd])      
    return has_neg   

def get_ball_pc(pos, num_pts_per_ball, radius):
    mesh = trimesh.creation.icosphere(radius=radius)
    T = trimesh.transformations.translation_matrix(pos)
    mesh.apply_transform(T)

    pc = trimesh.sample.sample_surface_even(mesh, count=num_pts_per_ball)[0]

    return list(pc)


# def get_balls_xyz_curve(num_balls, degree, offset=0.01, verbose=True):
def get_balls_xyz_curve(num_balls, curve_type, verbose=True):
    '''
    sample coordinates along a random curve
    '''
    if verbose:
        print("======================balls========================")
    balls_xyz = []
    ############################## random curve ###########################################
    # rand_weights_list = get_rand_weights_list(degree)
    # xy_curve_weights = [-0.35, np.random.uniform(low=-1, high=3),np.random.uniform(low=-3, high=-3)]
    # x = np.random.uniform(low=-0.1, high=-0.05)
    ############################## random straight line ###################################
    # rand_weights_list = [np.array([0.1]), np.array([-1, 0])]
    # xy_curve_weights = [-0.5, np.random.uniform(low=-1, high=1)]
    # x = np.random.uniform(low=-0.1, high=-0.05)
    ############################## flat random straight line #####################################
    # rand_weights_list = [np.array([0.1])]
    # xy_curve_weights = [-0.5, np.random.uniform(low=-1, high=1)]
    # x = np.random.uniform(low=-0.1, high=-0.05)
    ############################## flat random straight line 2 ball#####################################
    offset = None
    rand_weights_list = None
    xy_curve_weights = None
    x = None
    if curve_type == "2ballFlatLinear":
        offset = np.random.uniform(low=0.02, high=0.1)
        rand_weights_list = [np.array([0.1])]
        xy_curve_weights = [-0.5, np.random.uniform(low=-1, high=1)]
        x = np.random.uniform(low=-0.1, high=-0.05)
    ############################## fixed straight line ####################################
    # rand_weights_list = [np.array([0.1]), np.array([-1, 0])]
    # xy_curve_weights = [-0.5, -1]
    # x = -0.1
    #######################################################################################
    for i in range(num_balls):
        y = poly2D(x, xy_curve_weights)
        z = abs(poly3D(x, y, rand_weights_list))
        if verbose:
            print(f"{i+1}. x:{x}, y:{y}, z:{z}")
        balls_xyz.append([x,y,z])
        x += offset
    if verbose:
        print("==================================================")
    return rand_weights_list, xy_curve_weights, balls_xyz


def get_ball_pcs_curve(num_balls, degree, num_pts_per_ball=512, radius=0.005):
    final_pc = []
    rand_weights_list, xy_curve_weights, balls_xyz = get_balls_xyz_curve(num_balls, degree, 2*radius)
    for xyz in balls_xyz:
        pc = get_ball_pc(np.array([xyz[0], xyz[1], xyz[2]]), num_pts_per_ball, radius)
        final_pc.extend(pc)
    return np.array(final_pc), rand_weights_list, xy_curve_weights, balls_xyz


def get_balls_xyz_discrete(num_balls):
    '''
    sample coordinates based on the previous generated coordinates without following any curve
    '''
    print("======================balls========================")
    balls_xyz = []
    x = np.random.uniform(low=-0.1, high=-0.05)
    y = np.random.uniform(low=-0.4, high=-0.5)
    z = np.random.uniform(low=0.05, high= 0.06)
    for i in range(num_balls):
        balls_xyz.append([x,y,z])
        print(f"{i+1}. x:{x}, y:{y}, z:{z}")
        x += 0.01#np.random.uniform(low=-0.005, high=0.005)
        y = y + np.random.uniform(low=-0.005, high=0.005)
        z = abs(z + np.random.uniform(low=-0.005, high=0.005))
    print("==================================================")
    return balls_xyz

def get_ball_pcs_discrete(num_balls, num_pts_per_ball=512, radius=0.005):
    final_pc = []
    balls_xyz = get_balls_xyz_discrete(num_balls)
    for xyz in balls_xyz:
        pc = get_ball_pc(np.array([xyz[0], xyz[1], xyz[2]]), num_pts_per_ball, radius)
        final_pc.extend(pc)
    return np.array(final_pc), balls_xyz




if __name__ == "__main__":
    rand_weights_list, xy_curve_weights, balls_xyz = get_balls_xyz_curve(20, 3, 0.005)
    #rand_weights_list = get_rand_weights_list(degree=3)
    print(visualize_poly3D(rand_weights_list))


