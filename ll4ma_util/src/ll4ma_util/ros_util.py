import os
import numpy as np
import torch
from matplotlib import colors
import xml.etree.ElementTree as ET
from time import time

try:
    import cv2
    from cv_bridge import CvBridge
except ModuleNotFoundError:
    pass

import rospy
import rospkg
rospack = rospkg.RosPack()
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped, Pose, PoseStamped
from visualization_msgs.msg import (
    InteractiveMarker,
    InteractiveMarkerControl,
    Marker,
    MarkerArray,
)
from tf2_msgs.msg import TFMessage
from moveit_msgs.msg import (
    DisplayRobotState,
    DisplayTrajectory,
    ObjectColor,
    RobotState,
    RobotTrajectory,
)
from std_msgs.msg import ColorRGBA
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from ll4ma_util import data_util, math_util, file_util


def rgb_to_msg(array):
    return img_to_msg(cv2.cvtColor(array, cv2.COLOR_RGB2BGR))


def msg_to_rgb(msg):
    return cv2.cvtColor(msg_to_img(msg), cv2.COLOR_BGR2RGB)


def seg_to_msg(seg, seg_ids, colors):
    return rgb_to_msg(data_util.segmentation_to_rgb(seg, seg_ids, colors))


def depth_to_msg(array):
    return img_to_msg(array)


def msg_to_depth(msg):
    return msg_to_img(msg)


def img_to_msg(array):
    return CvBridge().cv2_to_imgmsg(array)


def msg_to_img(msg):
    img = CvBridge().imgmsg_to_cv2(msg)
    return img


def flow_to_msg(flow):
    magnitude, angle = cv2.cartToPolar(flow[:,:,0], flow[:,:,1])
    flow_img = np.zeros((flow.shape[0], flow.shape[1], 3))
    flow_img[:,:,0] = angle * 180 / np.pi / 2
    flow_img[:,:,2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    flow_img = flow_img.astype(np.uint8)
    flow_img = cv2.cvtColor(flow_img, cv2.COLOR_HSV2BGR)
    flow_msg = CvBridge().cv2_to_imgmsg(flow_img)
    return flow_msg


def msg_to_rgba(msg):
    return msg.r, msg.g, msg.b, msg.a


def rgba_to_msg(r, g, b, a):
    return ColorRGBA(r, g, b, a)


def array_to_pose(arr):
    pose = Pose()
    pose.position.x = arr[0]
    pose.position.y = arr[1]
    pose.position.z = arr[2]
    pose.orientation.x = arr[3]
    pose.orientation.y = arr[4]
    pose.orientation.z = arr[5]
    pose.orientation.w = arr[6]
    return pose


def pose_to_homogeneous(pose):
    p = pose_to_position(pose)
    q = pose_to_quaternion(pose)
    T = math_util.pose_to_homogeneous(p, q)
    return T


def pose_to_position(pose):
    return np.array([pose.position.x, pose.position.y, pose.position.z])


def pose_to_quaternion(pose):
    return np.array([
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w
    ])


def pose_to_rotation(pose):
    return math_util.quat_to_rotation(pose_to_quaternion(pose))


def array_to_homogeneous(arr):
    p = arr[:3]
    q = arr[3:]
    T = math_util.pose_to_homogeneous(p, q)
    return T


def homogeneous_to_pose(T):
    p, q = math_util.homogeneous_to_pose(T)
    pose = Pose()
    pose.position.x = p[0]
    pose.position.y = p[1]
    pose.position.z = p[2]
    pose.orientation.x = q[0]
    pose.orientation.y = q[1]
    pose.orientation.z = q[2]
    pose.orientation.w = q[3]
    return pose


def homogeneous_to_array(T):
    p, q = math_util.homogeneous_to_pose(T)
    arr = np.concatenate((p, q))
    return arr


def get_joint_state_msg(joint_pos=[], joint_vel=[], joint_tau=[], joint_names=[]):
    joint_state_msg = JointState()
    if len(joint_names) > 0:
        joint_state_msg.name = joint_names
    if len(joint_pos) > 0:
        joint_state_msg.position = joint_pos
    if len(joint_vel) > 0:
        joint_state_msg.velocity = joint_vel
    if len(joint_tau) > 0:
        joint_state_msg.effort = joint_tau
    return joint_state_msg


def get_color(color_id, alpha=1):
    """
    Color can be any name from this page:
        https://matplotlib.org/stable/gallery/color/named_colors.html
    """
    if isinstance(color_id, str):
        converter = colors.ColorConverter()
        c = converter.to_rgba(colors.cnames[color_id])
    elif len(color_id) == 3:
        c = list(color_id) + [1]
    else:
        c = color_id
    color = ColorRGBA(*c)
    color.a = alpha
    return color


def get_display_robot_msg(joints, joint_names, link_names=[], color=None, alpha=1.):
    msg = DisplayRobotState()
    msg.state.joint_state.name = joint_names
    msg.state.joint_state.position = joints
    if color is not None and link_names:
        if isinstance(color, list):
            msg.highlight_links = [ObjectColor(id=l, color=get_color(c))
                                   for l, c in zip(link_names, color)]
        else:
            msg.highlight_links = [ObjectColor(id=l, color=get_color(color)) for l in link_names]
        if isinstance(alpha, list):
            for link, a in zip(msg.highlight_links, alpha):
                link.color.a = a
        else:
            for link in msg.highlight_links:
                link.color.a = alpha
    return msg


def get_tf_msg(tf, idx):
    """
    Constructs a TFMessage for visualizing TF data in rviz.

    TODO (adam): I think I used this for visualizing Isaac Gym data, the input
    data seems specialized for that purpose so should probably make this more
    generic.
    """
    tf_msg = TFMessage()
    for child_frame, data in tf.items():
        tf_msg.transforms.append(
            get_tf_stamped_msg(
                data['position'][idx],
                data['orientation'][idx],
                data['parent_frame'],
                child_frame
            )
        )
    return tf_msg


def get_tf_stamped_msg(position, orientation, frame_id, child_frame_id):
    """
    Constructs a TransformStamped message from input data.
    """
    tf_stmp = TransformStamped()
    tf_stmp.header.frame_id = frame_id
    tf_stmp.child_frame_id = child_frame_id
    tf_stmp.transform.translation.x = position[0]
    tf_stmp.transform.translation.y = position[1]
    tf_stmp.transform.translation.z = position[2]
    tf_stmp.transform.rotation.x = orientation[0]
    tf_stmp.transform.rotation.y = orientation[1]
    tf_stmp.transform.rotation.z = orientation[2]
    tf_stmp.transform.rotation.w = orientation[3]
    return tf_stmp


def get_pose_stamped_msg(position=None, orientation=None, frame_id='world'):
    msg = PoseStamped()
    msg.header.frame_id = frame_id
    if position is not None:
        msg.pose.position.x = position[0]
        msg.pose.position.y = position[1]
        msg.pose.position.z = position[2]
    if orientation is not None:
        msg.pose.orientation.x = orientation[0]
        msg.pose.orientation.y = orientation[1]
        msg.pose.orientation.z = orientation[2]
        msg.pose.orientation.w = orientation[3]
    return msg


def get_marker_msg(
        position=[0, 0, 0],
        orientation=[0, 0, 0, 1],
        pose=None,
        scale=[1, 1, 1],
        shape='cube',
        color='blue',
        alpha=1,
        marker_id=1,
        frame_id='world',
        text='',
        mesh_resource='',
        use_embedded_materials=False,
        points=[]
):
    marker = Marker()
    marker.id = marker_id
    if shape == 'cube':
        marker.type = Marker.CUBE
    elif shape == 'sphere':
        marker.type = Marker.SPHERE
    elif shape == 'arrow':
        marker.type = Marker.ARROW
        marker.points = points
    elif shape == 'text':
        marker.type = Marker.TEXT_VIEW_FACING
        marker.text = text
    elif shape == 'mesh':
        marker.type = Marker.MESH_RESOURCE
        marker.mesh_resource = mesh_resource
        marker.mesh_use_embedded_materials = use_embedded_materials
    # else:
    #     raise ValueError(f"Unknown shape for marker: {shape}")
    marker.action = Marker.MODIFY
    if pose is not None:
        marker.pose = pose
    else:
        marker.pose.position.x = position[0]
        marker.pose.position.y = position[1]
        marker.pose.position.z = position[2]
        marker.pose.orientation.x = orientation[0]
        marker.pose.orientation.y = orientation[1]
        marker.pose.orientation.z = orientation[2]
        marker.pose.orientation.w = orientation[3]
    marker.header.frame_id = frame_id
    marker.scale.x = scale[0]
    marker.scale.y = scale[1]
    marker.scale.z = scale[2]
    marker.color = get_color(color, alpha)
    return marker


def get_imarker_msg(
        position=[0, 0, 0],
        orientation=[0, 0, 0, 1],
        pose=None,
        scale=[1, 1, 1],
        shape='cube',
        color='blue',
        alpha=1,
        marker_id=1,
        frame_id='world',
        mesh_resource='',
        imarker_scale=0.3,
        imarker_name='',
        imarker_desc='',
        imarker_move_x=True,
        imarker_move_y=True,
        imarker_move_z=True,
        imarker_rotate_x=True,
        imarker_rotate_y=True,
        imarker_rotate_z=True
):
    marker = get_marker_msg(
        position=position,
        orientation=orientation,
        pose=pose,
        scale=scale,
        shape=shape,
        color=color,
        alpha=alpha,
        marker_id=marker_id,
        frame_id=frame_id,
        mesh_resource=mesh_resource,
    )

    imarker = InteractiveMarker()
    imarker.header.frame_id = frame_id
    imarker.scale = imarker_scale
    imarker.name = imarker_name
    imarker.pose = marker.pose
    imarker.description = imarker_desc

    # Add the marker
    control = InteractiveMarkerControl()
    control.always_visible = True
    control.markers.append(marker)
    imarker.controls.append(control)

    # Add the controls
    if imarker_move_x:
        control = InteractiveMarkerControl()
        control.orientation.w = 0.70710678
        control.orientation.x = 0.70710678
        control.orientation.y = 0
        control.orientation.z = 0
        control.name = "move_x"
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        imarker.controls.append(control)
    if imarker_rotate_x:
        control = InteractiveMarkerControl()
        control.orientation.w = 0.70710678
        control.orientation.x = 0.70710678
        control.orientation.y = 0
        control.orientation.z = 0
        control.name = "rotate_x"
        control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        imarker.controls.append(control)
    if imarker_move_y:
        control = InteractiveMarkerControl()
        control.orientation.w = 0.70710678
        control.orientation.x = 0
        control.orientation.y = 0
        control.orientation.z = 0.70710678
        control.name = "move_y"
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        imarker.controls.append(control)
    if imarker_rotate_y:
        control = InteractiveMarkerControl()
        control.orientation.w = 0.70710678
        control.orientation.x = 0
        control.orientation.y = 0
        control.orientation.z = 0.70710678
        control.name = "rotate_y"
        control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        imarker.controls.append(control)
    if imarker_move_z:
        control = InteractiveMarkerControl()
        control.orientation.w = 0.70710678
        control.orientation.x = 0
        control.orientation.y = 0.70710678
        control.orientation.z = 0
        control.name = "move_z"
        control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        imarker.controls.append(control)
    if imarker_rotate_z:
        control = InteractiveMarkerControl()
        control.orientation.w = 0.70710678
        control.orientation.x = 0
        control.orientation.y = 0.70710678
        control.orientation.z = 0
        control.name = "rotate_z"
        control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        imarker.controls.append(control)
    return imarker
    

def publish_msg(msg, publisher):
    if getattr(msg, 'header', None) is not None:
        msg.header.stamp = rospy.Time.now()
    publisher.publish(msg)


def publish_tf_msg(msg, publisher):
    for tf_stamped in msg.transforms:
        tf_stamped.header.stamp = rospy.Time.now()
        publisher.publish(msg)


def publish_marker_msg(msg, publisher):
    for marker in msg.markers:
        marker.header.stamp = rospy.Time.now()
        publisher.publish(msg)


def interpolate_joint_trajectory(nominal_traj, dt=None):
    if len(nominal_traj.points) <= 1:
        raise ValueError("Trajectory must have at least 2 points to interpolate")
    if dt is None:
        # Compute dt that was used to generate the trajectory
        dt = nominal_traj.points[-1].time_from_start.to_sec() - \
             nominal_traj.points[-2].time_from_start.to_sec()
    # if dt <= 0 or dt > 1:
    #     # Want to be conservative here since this will go on the real robot
    #     raise ValueError(f"Maybe a bad dt value?: {dt}")
    duration = nominal_traj.points[-1].time_from_start.to_sec()
    old_time = np.linspace(0, duration, len(nominal_traj.points))
    new_time = np.linspace(0, duration, int(duration / float(dt)))

    old_pos = np.stack([p.positions for p in nominal_traj.points])
    n_dims = old_pos.shape[-1]
    new_pos = np.stack([np.interp(new_time, old_time, old_pos[:,i])
                        for i in range(n_dims)], axis=-1)
    if nominal_traj.points[0].velocities:
        old_vel = np.stack([p.velocities for p in nominal_traj.points])
        new_vel = np.stack([np.interp(new_time, old_time, old_vel[:,i])
                            for i in range(n_dims)], axis=-1)
    else:
        new_vel = None
    if nominal_traj.points[0].accelerations:
        old_acc = np.stack([p.accelerations for p in nominal_traj.points])
        new_acc = np.stack([np.interp(new_time, old_time, old_acc[:,i])
                            for i in range(n_dims)], axis=-1)
    else:
        new_acc = None

    new_traj = JointTrajectory()
    new_traj.joint_names = nominal_traj.joint_names
    for t in range(len(new_time)):
        point = JointTrajectoryPoint()
        point.time_from_start = rospy.Duration(new_time[t])
        point.positions = new_pos[t,:]
        if new_vel is not None:
            point.velocities = new_vel[t,:]
        if new_acc is not None:
            point.accelerations = new_acc[t,:]
        new_traj.points.append(point)
    return new_traj


def plot_joint_trajectory(traj):
    duration = traj.points[-1].time_from_start.to_sec()
    time = np.linspace(0, duration, len(traj.points))
    pos = np.stack([p.positions for p in traj.points])
    vel = np.stack([p.velocities for p in traj.points])
    acc = np.stack([p.accelerations for p in traj.points])
    fig, axes = plt.subplots(3, 1)
    fig.set_size_inches(4, 12)
    axes[0].plot(time, pos)
    axes[1].plot(time, vel)
    axes[2].plot(time, acc)
    plt.tight_layout()
    plt.show()


def wait_for_publisher(pub, timeout=5.):
    """
    Waits for a publisher to have subscribers.

    This is important when you create a publisher and then immediately try
    to publish a message, the timing can work out that no one subscribed in
    the time it took to initialize the publisher and publish the message,
    so your message can get lost. This ensures someone will receive the messgae.
    """
    start = time()
    while pub.get_num_connections() == 0:
        rospy.sleep(0.1)
        if time() - start > timeout:
            return False
    return True


def display_rviz_trajectory(traj, joint_state, model_id, display_topic="/display_trajectory"):
    robot_traj = RobotTrajectory()
    robot_traj.joint_trajectory = traj
    robot_state = RobotState()
    robot_state.joint_state = joint_state
    display_traj = DisplayTrajectory()
    display_traj.model_id = model_id
    display_traj.trajectory = [robot_traj]
    display_traj.trajectory_start = robot_state

    pub = rospy.Publisher(display_topic, DisplayTrajectory, queue_size=1)
    rospy.loginfo("Initializing trajectory display publisher...")
    if not wait_for_publisher(pub):
        rospy.logwarn("Could not display trajectory in rviz. Is rviz running and is a "
                      "Trajectory instance subscribed to '{}'?".format(display_topic))
        return False
    else:
        pub.publish(display_traj)
        rospy.loginfo("Trajectory is displayed in rviz")
        return True


def display_rviz_mesh(
        mesh_resource,
        pose_stmp,
        display_topic="/display_mesh",
        color='springgreen',
        alpha=1
):
    position = [pose_stmp.pose.position.x,
                pose_stmp.pose.position.y,
                pose_stmp.pose.position.z]
    orientation = [pose_stmp.pose.orientation.x,
                   pose_stmp.pose.orientation.y,
                   pose_stmp.pose.orientation.z,
                   pose_stmp.pose.orientation.w]
    marker = get_marker_msg(position, orientation, shape='mesh', color=color, alpha=alpha,
                            frame_id=pose_stmp.header.frame_id, mesh_resource=mesh_resource)

    pub = rospy.Publisher(display_topic, Marker, queue_size=1)
    rospy.loginfo("Initializing mesh display publisher...")
    if not wait_for_publisher(pub):
        rospy.logwarn("Could not display mesh in rviz. Is rviz running and is a "
                      "Marker instance subscribed to '{}'?".format(display_topic))
    else:
        pub.publish(marker)
        rospy.loginfo("Mesh is displayed in rviz")


def call_service(req, service_name, service_type):
    service_call = rospy.ServiceProxy(service_name, service_type)
    resp = None
    try:
        resp = service_call(req)
        success = resp.success if hasattr(resp, 'success') else True
    except rospy.ServiceException as e:
        rospy.logerr("Service call failed: {}".format(e))
        success = False
    return resp, success


def get_path(pkg_name):
    return rospack.get_path(pkg_name)


def resolve_ros_package_path(path):
    if path.startswith('package://'):
        path = path.replace('package://', '')
        pkg_path = get_path(path.split('/')[0])
        rel_path = '/'.join(path.split('/')[1:])
        path = os.path.join(pkg_path, rel_path)
        return path
    else:
        print("No ROS path to resolve. Expected string to start with 'package://'")



def get_mesh_filename_from_urdf(urdf_path, collision=False):
    """
    Extracts mesh filename from a URDF file.

    TODO can add option for looking for specific link, right now assuming there
    is only one link in the urdf

    TODO can add option to resolve ROS path or keep relative, right now relative
    """
    if urdf_path.startswith('package://'):
        urdf_path = resolve_ros_package_path(urdf_path)
    file_util.check_path_exists(urdf_path, "URDF file")

    xml = ET.parse(urdf_path)
    if collision:
        node = xml.find('link').find('collision')
    else:
        node = xml.find('link').find('visual')
    mesh_filename = node.find('geometry').find('mesh').attrib['filename']
    # TODO assumes relative path to URDF, can make more general
    mesh_filename = os.path.join(os.path.dirname(urdf_path), mesh_filename)

    return mesh_filename


def tensor2rosTraj(traj, dt, max_vel_factor):

    from moveit_msgs.msg import RobotState, RobotTrajectory
    from trajectory_msgs.msg import JointTrajectoryPoint
    from moveit_interface.srv import GetPlanResponse

    # see ~/isaac_ws/src/ll4ma_moveit/moveit_interface/srv/GetPlan.srv
    rostraj = GetPlanResponse()

    # probably unnecessary to define
    # see http://docs.ros.org/en/lunar/api/moveit_msgs/html/msg/RobotState.html
    # rostraj.start_state = RobotState()

    # see http://docs.ros.org/en/api/moveit_msgs/html/msg/RobotTrajectory.html
    rostraj.trajectory = RobotTrajectory()
    rostraj.trajectory.joint_trajectory.header.stamp = rospy.Time.now()
    velocs, accels = positions2velocities(traj), positions2accelerations(traj)
    # if (velocs > max_vel_factor).any():
    #     import pdb; pdb.set_trace()
    for ti, (pos, vel, acc) in enumerate(zip(traj.tolist(), velocs.tolist(), accels.tolist())):
        # see http://docs.ros.org/en/api/trajectory_msgs/html/msg/JointTrajectoryPoint.html
        pt = JointTrajectoryPoint()
        pt.positions = pos
        pt.velocities = vel
        pt.accelerations = acc
        # pt.effort = asdf
        pt.time_from_start = rospy.Duration(dt*ti)
        rostraj.trajectory.joint_trajectory.points.append(pt)
    return rostraj


def positions2velocities(traj):
    velocs = torch.zeros_like(traj)
    # velocs[0] = 0*traj[0] # initial velocity is 0
    velocs[1:-1] = traj[1:-1] - traj[0:-2]
    velocs[-1] = traj[-1] - traj[-2]
    return velocs


def positions2accelerations(traj):
    accels = torch.zeros_like(traj)
    accels[0] = traj[1] - traj[0] # assumes starting velocity is 0
    accels[1:-1] = traj[2:] - 2*traj[1:-1] + traj[:-2]
    accels[-1] = traj[-1] - traj[-2] # assumes ending velocity is 0
    return accels


def invertTransform(trans):
    R = trans[:3,:3]
    t = trans[:3,3]
    trans[:3,3] = -R.T @ t
    trans[:3,:3] = R.T
    return trans


if __name__ == '__main__':
    filename = '/home/adam/isaac_ws/src/ll4ma_isaac/ll4ma_isaacgym/src/ll4ma_isaacgym/assets/cleaner.urdf'
    print(get_mesh_filename_from_urdf(filename, True))
