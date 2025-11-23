# CRC control stub: implement the CRC-specific communication here (serial/UDP/ROS)
import time

class CRCController:
    def __init__(self, config):
        self.interface = config.get('interface','sim')
    def send_control(self, steer, accel):
        # convert to CRC actuation commands (PWM or velocity commands)
        print(f'[CRC] send_control: steer={steer:.3f}, accel={accel:.3f}')
