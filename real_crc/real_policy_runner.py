import time
from real_crc.mocap_interface import MoCapInterface
from real_crc.crc_control import CRCController

class RealPolicyRunner:
    def __init__(self, policy, mocap_cfg, crc_cfg):
        self.policy = policy
        self.mocap = MoCapInterface(mocap_cfg)
        self.crc = CRCController(crc_cfg)

    def run_episode(self, timeout=30.0):
        start = time.time()
        while time.time()-start < timeout:
            x,y,yaw = self.mocap.get_pose()
            obs = self._make_obs(x,y,yaw)
            steer, accel = self.policy.act(obs)
            self.crc.send_control(steer, accel)
            time.sleep(0.05)

    def _make_obs(self, x,y,yaw):
        # transform to policy observation structure
        return [x,y,yaw,0.0, 0.0,0.0,0.0, 10.0,10.0,10.0]
