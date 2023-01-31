#!/usr/bin/env python3
import os

import numpy as np
import rosbag
import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import WheelEncoderStamped, WheelsCmdStamped
from std_msgs.msg import Float32


class OdometryNode(DTROS):

    def __init__(self, node_name):
        """Wheel Encoder Node
        This implements basic functionality with the wheel encoders.
        """

        # Initialize the DTROS parent class
        super(OdometryNode, self).__init__(
            node_name=node_name, node_type=NodeType.PERCEPTION
        )
        self.veh_name = rospy.get_namespace().strip("/")

        # Get static parameters
        self._radius = rospy.get_param(
            f'/{self.veh_name}/kinematics_node/radius', 100
        )

        # Subscribing to the wheel encoders
        self.sub_encoder_ticks = {}
        self.pub_integrated_distance = {}
        self.wheels = {}
        for wheel in ["left", "right"]:
            self.wheels[wheel] = {
                "sub_encoder_ticks": rospy.Subscriber(
                    f'/{self.veh_name}/{wheel}_wheel_encoder_node/tick',
                    WheelEncoderStamped,
                    lambda msg, wheel=wheel: self.cb_encoder_data(wheel, msg)
                ),
                "pub_integrated_distance": rospy.Publisher(
                    f'~{wheel}_wheel_integrated_distance',
                    Float32,
                    queue_size=10
                ),
                "distance": 0,
                "direction": 1,
                "ticks": 0,
                "velocity": 0
            }
        self.sub_executed_commands = rospy.Subscriber(
            f'/{self.veh_name}/wheels_driver_node/wheels_cmd_executed',
            WheelsCmdStamped,
            self.cb_executed_commands
        )

        self.bag = rosbag.Bag('/data/bags/odometry.bag', 'w')

        self.log("Initialized")

    def cb_encoder_data(self, wheel, msg):
        """ Update encoder distance information from ticks.
        """
        self.bag.write(f'/{self.veh_name}/{wheel}_wheel_encoder/tick', msg)

        self.wheels[wheel]["distance"] += (
            self.wheels[wheel]["direction"] * 2 * np.pi * self._radius
            * (msg.data - self.wheels[wheel]["ticks"]) / msg.resolution
        )
        self.wheels[wheel]["ticks"] = msg.data
        self.wheels[wheel]["pub_integrated_distance"].publish(self.wheels[wheel]["distance"])

    def cb_executed_commands(self, msg):
        """ Use the executed commands to determine the direction of travel of each wheel.
        """
        self.bag.write(f'/{self.veh_name}/wheels_cmd_executed', msg)
        for wheel in self.wheels:
            velocity = getattr(msg, f"vel_{wheel}")
            self.wheels[wheel]["velocity"] = velocity
            self.wheels[wheel]["direction"] = 1 if velocity > 0 else -1
            print(f"{wheel:5} wheel direction: {self.wheels[wheel]['direction']}")
        print(f"{wheel:5} wheel distance: {self.wheels[wheel]['distance']:.2f} m")


if __name__ == '__main__':
    node = OdometryNode(node_name='my_odometry_node')
    # Keep it spinning to keep the node alive
    rospy.spin()
    rospy.loginfo("wheel_encoder_node is up and running...")
    node.bag.close()
