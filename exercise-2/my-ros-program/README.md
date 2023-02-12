# my-ros-program

This is a basic ROS program based on the `duckietown/template-ros` template.

## Installation and Usage

```bash
docker -H csc22902.local build -t duckietown/my-ros-program:latest-arm64v8 .
dts devel run -H csc22902.local -v /data:/data
```


## Packages

### my_package

Nodes:
* `my_publisher_node`: This node publishes a hello from `VEHCILE_NAME` message to the `~chatter` topic.
* `my_subscriber_node`: This node subscribes to the `~chatter` topic and prints the received message.
* `my_image_node`: This node subscribes to the `/{self.veh_name}/camera_node/image/compressed` topic and publishes the received image to the `~image/compressed` topic.
* `odometry_node`: This node subscribes to the `/{self.veh_name}/{wheel}_wheel_encoder_node/tick` and the `/{self.veh_name}/wheels_driver_node/wheels_cmd_executed` topic and publishes the integrated wheel distances to the `~{wheel}_wheel_integrated_distance` topics.
