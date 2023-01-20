#!/usr/bin/env python3
from time import sleep
import os
import cv2
import numpy as np


def gst_pipeline_string():
    # Parameters from the camera_node
    # Refer here : https://github.com/duckietown/dt-duckiebot-interface/blob/daffy/packages/camera_driver/config/jetson_nano_camera_node/duckiebot.yaml
    gst_pipeline = f"nvarguscamerasrc ! nvjpegenc ! appsink"
    print("Using GST pipeline: `{}`".format(gst_pipeline))
    return gst_pipeline

N_SPLITS = os.environ.get("N_SPLITS", 1)

cap = cv2.VideoCapture()
cap.open(gst_pipeline_string(), cv2.CAP_GSTREAMER)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    print(f"{ret}: {frame is not None}")
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
