#!/usr/bin/env python3

#  Author(s):  Bao Thach
#  Created on: 2022-12



import rospy
import time

from std_msgs.msg import Bool, Float64, Empty

class console(object):
    """Simple dVRK console API wrapping around ROS messages
    """

    # initialize the console
    def __init__(self, console_namespace = ''):
        # base class constructor in separate method so it can be called in derived classes
        self.__init_console(console_namespace)


    def __init_console(self, console_namespace = ''):
        """Constructor.  This initializes a few data members. It
        requires a arm name, this will be used to find the ROS topics
        for the console being controlled.  The default is
        'console' and it would be necessary to change it only if
        you have multiple dVRK consoles"""
        # data members, event based
        self.__console_namespace = console_namespace + 'console'
        self.__teleop_scale = 0.0

        # publishers
        self.__power_off_pub = rospy.Publisher(self.__console_namespace
                                               + '/power_off',
                                               Empty, latch = True, queue_size = 1)
        self.__power_on_pub = rospy.Publisher(self.__console_namespace
                                              + '/power_on',
                                              Empty, latch = True, queue_size = 1)
        self.__home_pub = rospy.Publisher(self.__console_namespace
                                          + '/home',
                                          Empty, latch = True, queue_size = 1)
        self.__teleop_enable_pub = rospy.Publisher(self.__console_namespace
                                                   + '/teleop/enable',
                                                   Bool, latch = True, queue_size = 1)
        self.__teleop_set_scale_pub = rospy.Publisher(self.__console_namespace
                                                      + '/teleop/set_scale',
                                                      Float64, latch = True, queue_size = 1)

        # subscribers
        rospy.Subscriber(self.__console_namespace
                         + '/teleop/scale',
                         Float64, self.__teleop_scale_cb)

        # create node
        if not rospy.get_node_uri():
            rospy.init_node('console_api', anonymous = True, log_level = rospy.WARN)
        else:
            rospy.logdebug(rospy.get_caller_id() + ' -> ROS already initialized')


    def __teleop_scale_cb(self, data):
        """Callback for teleop scale.

        :param data: the latest scale requested for the dVRK console"""
        self.__teleop_scale = data.data


    def power_off(self):
        rate = rospy.Rate(10)    
        for _ in range(5):
            self.__power_off_pub.publish()
            rate.sleep()

    def power_on(self):  
        rate = rospy.Rate(10)    
        for _ in range(5):
            self.__power_on_pub.publish()
            rate.sleep()

    def home(self):       
        rate = rospy.Rate(10)    
        for _ in range(5):
            self.__home_pub.publish()
            rate.sleep()

    def teleop_start(self):
        rate = rospy.Rate(10)    
        for _ in range(5):
            self.__teleop_enable_pub.publish(True)
            rate.sleep()

    def teleop_stop(self):       
        rate = rospy.Rate(10)    
        for _ in range(15):
            self.__teleop_enable_pub.publish(False)
            rate.sleep()

    def teleop_set_scale(self, scale):       
        rate = rospy.Rate(10)    
        for _ in range(5):
            self.__teleop_set_scale_pub.publish(scale)
            rate.sleep()

    def teleop_get_scale(self):
        return self.__teleop_scale


def reset_mtm(console, verbose=True):
    console.power_off()
    rospy.sleep(0.5)
    console.power_on()
    rospy.sleep(0.5)       
    console.home() 
    rospy.sleep(2)
    console.teleop_start()
    rospy.sleep(1)
    
    if verbose:
        print("Succesfully reset MTM")

class _WFM(object):
    def __init__(self):
        self.msg = None
    def cb(self, msg):
        if self.msg is None:
            self.msg = msg
            


def get_pedal_message(topic_type, pedal_name="camera"):
    wfm = _WFM()
    topic = f"/footpedals/{pedal_name}"
    rospy.topics.Subscriber(topic, topic_type, wfm.cb)
    return wfm.msg

# cs = console()

# cs.power_on()
# cs.teleop_start()
# cs.teleop_stop()

# cs.home()
# rospy.sleep(2)
# cs.power_off()
