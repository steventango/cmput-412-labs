#!/usr/bin/env python3
import os
import sys
from time import sleep

import cv2
import numpy as np

sys.path.append('/code/catkin_ws/src/dt-duckiebot-interface/packages/camera_driver/src/')
from jetson_nano_camera_node import JetsonNanoCameraNode

N_SPLITS = os.environ.get("N_SPLITS", 1)

# https://github.com/duckietown/dt-duckiebot-interface/blob/daffy/packages/
# camera_driver/src/jetson_nano_camera_node.py
class CapturableJetsonNanoCameraNode(JetsonNanoCameraNode):
    def capture(self):
        """Image capture procedure.
        Captures a frame from the /dev/video2 image sink and returns it
        """
        if self._device is None or not self._device.isOpened():
            self.logerr("Device was found closed")
            return
        # get first frame
        return self._device.read() if self._device else (False, None)


camera_node = CapturableJetsonNanoCameraNode()
camera_node.start()

while(True):
    # Capture frame-by-frame
    ret, frame = camera_node.capture()
    print('a')
    if ret:
        print('b')
        width = frame.shape[1]
        sectors = np.split(frame, N_SPLITS, axis=1)
        print('Mean color in each sector (RGB): ')
        for i, sector in enumerate(sectors):
            color = np.mean(sector, axis=(0, 1))
            print(f'\tSector {i}: {color}')
        print()
    sleep(1)
