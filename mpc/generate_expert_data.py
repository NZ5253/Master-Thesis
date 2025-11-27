# mpc/generate_expert_data.py

import os
import pickle
from typing import List

import numpy as np
import yaml

from env.parking_env import ParkingEnv
from mpc.teb_mpc import TEBMPC, VehicleState, ParkingGoal, Obstacle


def _env_obstacles_to_teb(env: ParkingEnv) -> List[Obstacle]:
    """Convert env.obstacles.obstacles to TEB obstacles."""
    obs_list: List[Obstacle] = []
    for o in env.obstacles.obstacles:
        cx = o["x"]
        cy = o["y"]
        w = o["w"]
        h = o["h"]

        # skip thin walls near border for MPC performance
        is_thin = (w < 0.1) or (h < 0.1)
        near_border = (abs(cx) > 1.8) or (abs(cy) > 1.8)
        if is_thin and near_border:
            continue

        hx = w / 2.0
        hy = h / 2.0
        obs_list.append(Obstacle(cx=cx, cy=cy, hx=hx, hy=hy))
    return obs_list


def _next_episode_index(out_dir: str) -> int:
    if not os.path.exists(out_dir):
        return 0
    max_idx = -1
    for fname in os.listdir(out_dir):
        if fname.startswith("episode_") and fname.endswith(".pkl"):
            try:
                idx = int(fname[len("episode_"): -len(".pkl")])
                max_idx = max(max_idx, idx)
            except ValueError:
                continue
    return max_idx + 1


def generate(cfg_env, n_episodes: int = 50, out_dir: str = "data/expert_trajectories") -> None:
    os.makedirs(out_dir, exist_ok=True)

    env = ParkingEnv(cfg_env)
    teb = TEBMPC()

    start_idx = _next_episode_index(out_dir)
    print(f"[INFO] Starting generation from ID {start_idx}")

    for ep_offset in range(n_episodes):
        ep_idx = start_idx + ep_offset
        print(f"[EPISODE {ep_idx}] starting...")

        # 1. Randomize Env
        obs = env.reset(randomize=True)

        # 2. CAPTURE the Goal and Obstacles for this specific episode!
        #    (Crucial for visualization later)
        episode_goal = env.goal.copy()  # [x, y, yaw]
        episode_obstacles = [o.copy() for o in env.obstacles.obstacles]

        traj = []
        done = False
        step = 0

        while not done:
            # reconstruct VehicleState from env.state
            x, y, yaw, v = env.state
            state = VehicleState(x=x, y=y, yaw=yaw, v=v)

            gx, gy, gyaw = env.goal
            goal = ParkingGoal(x=gx, y=gy, yaw=gyaw)

            obstacles = _env_obstacles_to_teb(env)

            try:
                sol = teb.solve(state, goal, obstacles, profile="perpendicular")
                steer = float(sol.controls[0, 0])
                accel = float(sol.controls[0, 1])
                action = np.array([steer, accel], dtype=float)
            except Exception as e:
                print(f"  TEB-MPC error at step {step}: {e}")
                action = np.zeros(2, dtype=float)

            # store obs and action
            traj.append((obs.copy(), action.copy()))

            obs, reward, done, info = env.step(action)
            step += 1

        term = info.get("termination", "unknown")

        # debug info
        x, y, yaw, v = env.state
        gx, gy, gyaw = env.goal
        pos_err = np.hypot(gx - x, gy - y)
        yaw_err = abs(((gyaw - yaw + np.pi) % (2 * np.pi)) - np.pi)

        print(
            f"[EPISODE {ep_idx}] finished in {step} steps, "
            f"termination={term}, pos_err={pos_err:.3f}, yaw_err={yaw_err:.3f}"
        )

        # 3. Save as a DICTIONARY containing everything needed to replay
        save_data = {
            "traj": traj,
            "goal": episode_goal,
            "obstacles": episode_obstacles,
            "termination": term
        }

        out_path = os.path.join(out_dir, f"episode_{ep_idx:04d}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(save_data, f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate expert trajectories with TEB-MPC.")
    parser.add_argument(
        "--episodes", "-n",
        type=int,
        default=50,
        help="Number of NEW episodes to generate.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/expert_trajectories",
        help="Output directory.",
    )

    args = parser.parse_args()

    with open("config_env.yaml", "r") as f:
        cfg_env = yaml.safe_load(f)

    generate(cfg_env, n_episodes=args.episodes, out_dir=args.out_dir)