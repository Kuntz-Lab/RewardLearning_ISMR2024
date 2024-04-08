#!/usr/bin/env python
import os
import sys
import trimesh
import rospy
import numpy as np
from copy import deepcopy

from ll4ma_util import math_util


class PoseCandidate:

    def __init__(self, position, x_axis, y_axis, bb_y_extent=0, bb_z_extent=0):
        if not np.dot(x_axis, y_axis) == 0.:
            raise ValueError("Input axes must be orthogonal")
        self.position = position
        self.set_rotation(np.stack([x_axis, y_axis, np.cross(x_axis, y_axis)]))
        self.bb_y_extent = bb_y_extent
        self.bb_z_extent = bb_z_extent
        # These will be set with post-processing on all candidates
        self.is_top = False
        self.is_bottom = False
        self.is_side = False

    def set_rotation(self, R):
        self.rotation_matrix = R
        T = np.eye(4)
        T[:3, :3] = self.rotation_matrix
        T[:3, 3] = self.position
        self.set_homogeneous(T)

    def set_homogeneous(self, T):
        self.homogeneous_matrix = T
        self.rotation_matrix = self.homogeneous_matrix[:3, :3]
        self.position = self.homogeneous_matrix[:3, 3]
        self.quaternion = math_util.rotation_to_quat(self.rotation_matrix)

    def get_pose(self):
        return np.concatenate([self.position, self.quaternion])
        
        
def get_rotated_candidates(candidate):
    rotated_candidates = []
    # Rotate 90 degrees to get each axis-aligned pose in the plane
    rotations_90_deg = [np.pi/2., np.pi, 3*np.pi/2.]
    # rotations_90_deg = [np.pi/2.]
    rotated = deepcopy(candidate)
    for theta in rotations_90_deg:
        new_rot = np.dot(candidate.rotation_matrix, math_util.get_x_axis_rotation(theta))
        rotated = deepcopy(rotated)
        rotated.set_rotation(new_rot)
        # Swap extents since it's a 90 degree rotation
        temp = rotated.bb_y_extent
        rotated.bb_y_extent = rotated.bb_z_extent
        rotated.bb_z_extent = temp
        rotated_candidates.append(rotated)
    return rotated_candidates
    

def get_bb_dims(mesh_filename='', prim_type='', prim_dims=None):
    # TODO can add option for point cloud also (can also do in Trimesh to get bounds)
    
    if mesh_filename:
        mesh = trimesh.load(mesh_filename)
        x_extent = mesh.bounds[1, 0] - mesh.bounds[0, 0]
        y_extent = mesh.bounds[1, 1] - mesh.bounds[0, 1]
        z_extent = mesh.bounds[1, 2] - mesh.bounds[0, 2]
    elif prim_type and prim_dims is not None:
        if prim_type == 'box':
            x_extent, y_extent, z_extent = prim_dims
        else:
            raise ValueError(f"Unknown primitive type for pose candidates: {prim_type}")
    else:
        raise ValueError("Must specify mesh_filename, or both prim_type and prim_dims")
    return x_extent, y_extent, z_extent


def get_bb_pose_candidates(obj_pose, mesh_filename='', prim_type='',
                           prim_dims=None, offset=0., tolerance=1e-5):

    # TODO need to account for different alignment axis, this will
    # assume you're aligning x-axis towards center of BB

    x_extent, y_extent, z_extent = get_bb_dims(mesh_filename, prim_type, prim_dims)
        
    x_offset = 0.5 * x_extent + offset
    y_offset = 0.5 * y_extent + offset
    z_offset = 0.5 * z_extent + offset
    x_axis, y_axis, z_axis = np.eye(3)

    # Create nominal poses at each bounding box face s.t. each x-axis points into the box
    candidates = [
        PoseCandidate([ x_offset, 0., 0.], -x_axis,  y_axis, y_extent, z_extent),
        PoseCandidate([-x_offset, 0., 0.],  x_axis,  y_axis, y_extent, z_extent),
        PoseCandidate([0.,  y_offset, 0.], -y_axis, -x_axis, x_extent, z_extent),
        PoseCandidate([0., -y_offset, 0.],  y_axis,  x_axis, x_extent, z_extent),
        PoseCandidate([0., 0.,  z_offset], -z_axis, -y_axis, y_extent, x_extent),
        PoseCandidate([0., 0., -z_offset],  z_axis, -y_axis, y_extent, x_extent)
    ]

    # Generate 90 degree rotations of each candidate
    rotated_candidates = []
    for candidate in candidates:
        rotated_candidates += get_rotated_candidates(candidate)
    candidates += rotated_candidates

    # Transform everything based on object pose
    T = math_util.pose_to_homogeneous(obj_pose[:3], obj_pose[3:])
    for candidate in candidates:
        candidate.set_homogeneous(np.dot(T, candidate.homogeneous_matrix))

    # Post-process to identify them as top/bottom/side poses. Note this won't be super
    # robust but should work for objects that are on some supporting surface
    candidates = sorted(candidates, key=lambda x: x.position[2])
    lower_z = candidates[0].position[2] + tolerance
    upper_z = candidates[-1].position[2] - tolerance
    for candidate in candidates:
        candidate.is_top = candidate.position[2] >= upper_z
        candidate.is_bottom = candidate.position[2] <= lower_z
        candidate.is_side = not candidate.is_top and not candidate.is_bottom
            
    return candidates


def remove_bottom_pose_candidates(candidates):
    """
    Removes any poses for aligned to the bottom face of BB.
    """
    return [c for c in candidates if not c.is_bottom]


def remove_top_pose_candidates(candidates):
    return [c for c in candidates if not c.is_top]


def remove_side_pose_candidates(candidates):
    return [c for c in candidates if not c.is_side]


def remove_aligned_short_bb_face(candidates, align_axis='z'):
    # If the extents are equal this will not remove anything
    return [c for c in candidates if ((c.bb_y_extent >= c.bb_z_extent and align_axis == 'z') or
                                      (c.bb_z_extent >= c.bb_y_extent and align_axis == 'y'))]


def remove_aligned_long_bb_face(candidates, align_axis='z'):
    # If the extents are equal this will not remove anything
    return [c for c in candidates if ((c.bb_y_extent < c.bb_z_extent and align_axis == 'z') or
                                      (c.bb_z_extent < c.bb_y_extent and align_axis == 'y'))]


def get_top_pose_candidates(candidates):
    """
    Returns pose candidates that are facing the top face of BB.
    """
    return [c for c in candidates if c.is_top]


def get_bottom_pose_candidates(candidates):
    return [c for c in candidates if c.is_bottom]


def get_side_pose_candidates(candidates):
    return [c for c in candidates if c.is_side]



if __name__ == '__main__':
    rospy.init_node('test_manip_util')
    
    from ll4ma_util import ros_util
    from visualization_msgs.msg import MarkerArray, Marker
    from geometry_msgs.msg import Pose, PoseArray, PoseStamped

    resource = os.path.join(ros_util.get_path("ll4ma_isaacgym"),
                            "src", "ll4ma_isaacgym", "assets", "cleaner.stl")
    ros_resource = "package://ll4ma_isaacgym/src/ll4ma_isaacgym/assets/cleaner.stl"
    
    pos = [1, 0, 0.12529]
    quat = [0, 0, 0, 1]
    # quat = math_util.random_quat().tolist()
    pose = pos + quat
    offset = 0.05
    candidates = get_bb_pose_candidates(pose, resource, offset)

    # candidates = remove_bottom_pose_candidates(candidates)
    # candidates = remove_top_pose_candidates(candidates)
    # candidates = get_top_pose_candidates(candidates)
    candidates = remove_aligned_short_bb_face(candidates, 'z')
    print("Y", candidates[0].bb_y_extent)
    print("Z", candidates[0].bb_z_extent)

    origin_msg = PoseStamped()
    origin_msg.header.frame_id = 'world'
    origin_msg.pose.position.x = pose[0]
    origin_msg.pose.position.y = pose[1]
    origin_msg.pose.position.z = pose[2]
    origin_msg.pose.orientation.x = pose[3]
    origin_msg.pose.orientation.y = pose[4]
    origin_msg.pose.orientation.z = pose[5]
    origin_msg.pose.orientation.w = pose[6]

    mesh_msg = ros_util.get_marker_msg(pos, quat, shape='mesh', mesh_resource=ros_resource,
                                       color='goldenrod', alpha=0.8)
    
    pose_msg = PoseArray()
    pose_msg.header.frame_id = 'world'
    face_msg = MarkerArray()
    for i, c in enumerate(candidates):
        p = Pose()
        p.position.x = c.position[0]
        p.position.y = c.position[1]
        p.position.z = c.position[2]
        p.orientation.x = c.quaternion[0]
        p.orientation.y = c.quaternion[1]
        p.orientation.z = c.quaternion[2]
        p.orientation.w = c.quaternion[3]
        pose_msg.poses.append(p)

        scale = np.array([0.001, c.bb_y_extent, c.bb_z_extent])
        m = ros_util.get_marker_msg(c.position, c.quaternion, scale=scale, marker_id=i, alpha=0.5)
        face_msg.markers.append(m)

    origin_pub = rospy.Publisher("/obj_origin", PoseStamped, queue_size=1)
    pose_pub = rospy.Publisher("/pose_candidates", PoseArray, queue_size=1)
    mesh_pub = rospy.Publisher("/mesh", Marker, queue_size=1)
    face_pub = rospy.Publisher("/faces", MarkerArray, queue_size=1)
    
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        origin_pub.publish(origin_msg)
        pose_pub.publish(pose_msg)
        mesh_pub.publish(mesh_msg)
        face_pub.publish(face_msg)
        rate.sleep()
    
