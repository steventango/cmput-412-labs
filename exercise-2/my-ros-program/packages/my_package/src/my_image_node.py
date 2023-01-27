#!/usr/bin/env python3

import os

import cv2
import numpy as np
import rospy
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage


class MyImageNode(DTROS):

    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(MyImageNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        self.hostname = os.environ.get('VEHICLE_NAME')
        self.pub = rospy.Publisher(f"~image/compressed", CompressedImage, queue_size=1)
        self.sub = rospy.Subscriber(f'camera_node/image/compressed', CompressedImage, self.callback)

    def callback(self, data):
        array = np.frombuffer(data.data, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
        rospy.loginfo(f"Size of the image {image.shape}")
        self.pub.publish(data)

if __name__ == '__main__':
    # create the node
    node = MyImageNode(node_name='my_image_node')
    # keep spinning
    rospy.spin()
