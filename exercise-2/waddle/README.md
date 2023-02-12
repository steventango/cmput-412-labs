# waddle

This workspace implements the exercise 2 requirements. Namely,
it implements mobile robot kinematics, odometry, LED control, and implements code for controlling the Duckiebot to move in a square and then a circle.

This ROS program is based on the `duckietown/template-ros` template.

## Installation and Usage

```bash
docker -H csc22902.local build -t duckietown/waddle:latest-arm64v8 .
dts devel run -H csc22902.local -v /data:/data
```

Replace `csc22902` with the hostname of your Duckiebot.

## Packages

### led_controls

Nodes:
* `led_controls_node`: This node contains the `led_control_service` service which can be called to switch the LED colors. It implements this by publishing to the `/{hostname}/led_emitter_node/led_pattern` topic.

### odometry_node

Nodes:
* `odometry_driver_node`: This node implements the odometry based driver control node. It subscribes to the `/{hostname}/left_wheel_integrated_distance`, `/{hostname}/right_wheel_integrated_distance`, and `world_kinematics`. It makes requests
to the `led_control_service` service to switch the LED colors.
It uses a variety of hard coded methods to drive the Duckiebot in a square and then a circle. It publishes to the `/{hostname}/wheels_driver_node/wheels_cmd` topic to control the robot.
* `odometry_publisher_node`: This node implements the forward kinematics of the Duckiebot. It subscribes to the `/{hostname}/right_wheel_encoder_node/tick`, `/{hostname}/left_wheel_encoder_node/tick`, and `/{hostname}/wheels_driver_node/wheels_cmd_executed` topics. It publishes to the `/{hostname}/right_wheel_integrated_distance`, `/{hostname}/left_wheel_integrated_distance`, and `/{hostname}/world_kinematics` topics.
