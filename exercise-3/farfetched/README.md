# farfetched

This workspace implements the exercise 3 requirements. Namely,
it implements computer vision, a PID controller, and localization using
sensor fusion.

This ROS program is based on the `duckietown/template-ros` template.

## Installation and Usage

```bash
docker -H csc22902.local build -t duckietown/duckietown:latest-arm64v8 .
dts devel run -H csc22902.local -v /data:/data
```

Replace `csc22902` with the hostname of your Duckiebot.

## Packages

### apriltag

This package implements AprilTag detection and localization, it uses the
teleport publisher to update the robot's pose.

### augmented_reality_apriltag

This package implements the [Unit A-4 Advanced Augmented Reality Exercise](https://docs.duckietown.org/daffy/duckietown-classical-robotics/out/cra_apriltag_augmented_reality_exercise.html).

### augmented_reality_basics

This package implements the [Unit A-3 Basic Augmented Reality Exercise](https://docs.duckietown.org/daffy/duckietown-classical-robotics/out/cra_basic_augmented_reality_exercise.html).

### deadreckoning

This package implements dead reckoning with a teleport subscriber that
is used for sensor fusion. This package also publishes any static transforms.

### farfetched_msgs

This package contains the custom messages used in this workspace.

### lane_finder

This package implements the lane finder node, which uses OpenCV to detect
the lane lines in the image and publishes an error.

### lane_follower

This package implements the lane follower node, which uses a PID controller
to follow the lane lines.
