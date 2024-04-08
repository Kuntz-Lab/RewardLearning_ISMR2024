#!/usr/bin/env python
import os
import sys
import tf2_ros as tf
import yaml
import rospy
import rospkg
import transformations
from geometry_msgs.msg import PoseStamped, Pose, TransformStamped

def lookup_transform(parent_link, child_link, disp_msg=False):
    try:
        tf_stmp = tf_buffer.lookup_transform(parent_link, child_link, rospy.Time.now(),
                                             rospy.Duration(10.0))
        if disp_msg:
            rospy.loginfo("Tranform between '{}' and '{}' found!".format(parent_link, child_link))
        return tf_stmp
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
        rospy.logerr(e)
        rospy.logerr("Could not find tranform between '{}' and '{}'.".format(parent_link,
                                                                             child_link))
        return None


if __name__ == '__main__':
    rospy.init_node("camera_to_robot_transform_saver")

    parent_link   = "PSM1_psm_base_link"  
    child_link =  "PSM1_tool_tip_link"    # orientation
    # child_link =  "PSM1_tool_wrist_sca_ee_link_0"  # position

    # parent_link   = "PSM1_tool_tip_link"  "psm_base_link" ##""PSM1_base" 
    # child_link =  "psm_tool_yaw_link"
    
 
    tf_buffer = tf.Buffer()
    tf_listener = tf.TransformListener(tf_buffer)

    # Wait a few seconds to let the camera image stabilize
    rospy.sleep(1.0)
    


    transform = lookup_transform(parent_link, child_link, disp_msg=True)
    print(transform)

    
    eulers = transformations.euler_from_quaternion([transform.transform.rotation.y, transform.transform.rotation.x,transform.transform.rotation.z,transform.transform.rotation.w])
    print(eulers)


    transform_2 = rospy.wait_for_message('/PSM1/measured_cp', TransformStamped)
    eulers_2 = transformations.euler_from_quaternion([transform_2.transform.rotation.y, transform_2.transform.rotation.x,transform_2.transform.rotation.z,transform_2.transform.rotation.w])
    print(eulers_2)
   
    """_summary_
    Zero position: (pi/2,0,0) (xyz) = (0,0,0)
    order: roll, pitch, yaw

    """
   
    # transform: 
    # translation: 
    #     x: -0.38652575201
    #     y: -0.249997084113
    #     z: 0.253687490864
    # rotation: 
    #     x: -0.500001187324
    #     y: -0.500006167576
    #     z: 0.500000774889
    #     w: 0.499991870105




    # poses = {}
    # poses["{}__to__{}".format(base_link, camera_link)] =  {
    #     "position" : {
    #         'x' : base_camera_stmp.transform.translation.x,
    #         'y' : base_camera_stmp.transform.translation.y,
    #         'z' : base_camera_stmp.transform.translation.z },
    #     "orientation" : {
    #         'x' : base_camera_stmp.transform.rotation.x,
    #         'y' : base_camera_stmp.transform.rotation.y,
    #         'z' : base_camera_stmp.transform.rotation.z,
    #         'w' : base_camera_stmp.transform.rotation.w } }
    # if use_table:
    #     poses["{}__to__{}".format(base_link, table_link)] = {
    #         "position" : {
    #             'x' : base_table_stmp.transform.translation.x,
    #             'y' : base_table_stmp.transform.translation.y,
    #             'z' : base_table_stmp.transform.translation.z },
    #         "orientation" : {
    #             'x' : base_table_stmp.transform.rotation.x,
    #             'y' : base_table_stmp.transform.rotation.y,
    #             'z' : base_table_stmp.transform.rotation.z,
    #             'w' : base_table_stmp.transform.rotation.w } }

    # rospy.loginfo("Writing transforms to file...")
    # rospack = rospkg.RosPack()
    # path = rospack.get_path("robot_aruco_calibration")
    # filename = os.path.join(path, "config/robot_camera_calibration.yaml")
    # with open(filename, 'w') as f:
    #     yaml.dump(poses, f, default_flow_style=False)
    # rospy.loginfo("Transforms saved to file: {}".format(filename))
