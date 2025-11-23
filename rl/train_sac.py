import argparse, yaml, os
import numpy as np
from env.parking_env import ParkingEnv
from rl.agent_sac import SACAgent

def train(cfg):
    env = ParkingEnv(cfg['env'])
    obs = env.reset()
    obs_dim = len(obs)
    act_dim = 2
    agent = SACAgent(obs_dim, act_dim, cfg['agent'])
    for ep in range(cfg['train']['episodes']):
        obs = env.reset()
        done = False
        while not done:
            action = agent.select_action(obs)
            obs2, r, done, info = env.step(action)
            agent.store(obs, action, r, obs2, done)
            agent.update()
            obs = obs2
        if ep % cfg['train'].get('save_every',50) == 0:
            agent.save(os.path.join(cfg['train']['out_dir'],'agent_ckpt.pt'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='rl/config_rl.yaml')
    args = parser.parse_args()
    cfg = {}
    if os.path.exists(args.cfg):
        with open(args.cfg,'r') as f:
            cfg = yaml.safe_load(f)
    train(cfg)
