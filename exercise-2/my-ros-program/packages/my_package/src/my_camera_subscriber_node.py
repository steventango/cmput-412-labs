#!/usr/bin/env python3

import os
import rospy
from duckietown.dtros import DTROS, NodeType
from std_msgs.msg import String

class MyCameraSubscriberNode(DTROS):

    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(MyCameraSubscriberNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        # construct publisher
        hostname = os.environ.get('VEHICLE_NAME')
        self.sub = rospy.Subscriber(f'/{hostname}/camera_node/image/compressed', String, self.callback)

    def callback(self, data):
        rospy.loginfo("I heard %s", data.data)

if __name__ == '__main__':
    # create the node
    node = MyCameraSubscriberNode(node_name='my_subscriber_node')
    # keep spinning
    rospy.spin()
