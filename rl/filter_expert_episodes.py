# rl/filter_expert_episodes.py

import os
import argparse
import pickle
import shutil

import numpy as np
import yaml

from env.parking_env import ParkingEnv


def filter_expert_folder(in_dir: str, out_dir: str, cfg_env_path: str = "config_env.yaml"):
    with open(cfg_env_path, "r") as f:
        cfg_env = yaml.safe_load(f)
    env = ParkingEnv(cfg_env)

    os.makedirs(out_dir, exist_ok=True)

    files = sorted(
        f for f in os.listdir(in_dir)
        if f.startswith("episode_") and f.endswith(".pkl")
    )

    kept = 0
    total = 0

    for fname in files:
        total += 1
        in_path = os.path.join(in_dir, fname)

        with open(in_path, "rb") as fh:
            traj = pickle.load(fh)  # list of (obs, action)

        if len(traj) == 0:
            continue

        last_obs, last_action = traj[-1]
        last_obs = np.asarray(last_obs, dtype=float)
        last_action = np.asarray(last_action, dtype=float)

        # state is the first 4 entries of obs
        last_state = last_obs[:4]
        next_state = env.model.step(
            last_state,
            (last_action[0], last_action[1]),
            dt=env.dt,
        )

        if env._is_success(next_state):
            out_path = os.path.join(out_dir, fname)
            shutil.copy2(in_path, out_path)
            kept += 1

    print(f"[FILTER] Scanned {total} episodes from {in_dir}")
    print(f"[FILTER] Kept {kept} successful episodes in {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", type=str, default="data/expert_trajectories")
    parser.add_argument("--out-dir", type=str, default="data/expert_success")
    parser.add_argument("--cfg-env", type=str, default="config_env.yaml")
    args = parser.parse_args()

    filter_expert_folder(args.in_dir, args.out_dir, args.cfg_env)
