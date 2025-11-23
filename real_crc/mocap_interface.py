# MoCap interface template: replace with your lab's API (Vicon/OptiTrack/Qualisys)
# Example assumes MoCap publishes UDP JSON packets or ROS topic. This is a stub.

import time
import numpy as np

class MoCapInterface:
    def __init__(self, config):
        self.endpoint = config.get('endpoint', None)
        self.rate = config.get('rate', 100)

    def get_pose(self, body_name='crc'):
        # Return (x,y,yaw) for the CRC in meters/radians
        # Replace with actual MoCap client code
        t = time.time()
        x = 0.0
        y = 0.0
        yaw = 0.0
        return x,y,yaw
