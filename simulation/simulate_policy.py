import argparse, yaml, os
import numpy as np
from env.parking_env import ParkingEnv

def simulate(cfg):
    env = ParkingEnv(cfg['env'])
    if 'policy' in cfg:
        policy = cfg['policy']
    else:
        policy = None
    n = cfg.get('n', 10)
    for i in range(n):
        obs = env.reset()
        done = False
        traj = []
        while not done:
            if policy is None:
                action = [0.0, 0.0]
            else:
                action = policy.act(obs)
            obs, r, done, info = env.step(action)
            traj.append(obs)
        print('Run', i, 'termination', info.get('termination','-'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='simulation/config_sim.yaml')
    args = parser.parse_args()
    cfg = {}
    if os.path.exists(args.cfg):
        with open(args.cfg,'r') as f:
            cfg = yaml.safe_load(f)
    simulate(cfg)
