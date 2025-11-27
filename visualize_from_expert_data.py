# simulation/visualize_from_expert_data.py
import os
import pickle
import yaml
import matplotlib.pyplot as plt
import numpy as np

from env.parking_env import ParkingEnv
# Import the updated drawing functions
from visualize_mpc_episode import draw_car, draw_obstacles, draw_goal


def visualize_episode(pkl_path: str, cfg_env: dict):
    # Dummy env just for defaults
    env = ParkingEnv(cfg_env)

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    # --- 1. Detect New Dictionary Format ---
    if isinstance(data, dict):
        traj = data["traj"]
        goal = data["goal"]  # The ACTUAL goal used in that episode
        obstacles = data["obstacles"]  # The ACTUAL obstacles
        term = data.get("termination", "?")
        print(f"[VIS] Loaded Episode. Steps: {len(traj)}, Term: {term}")
    else:
        # Legacy list support
        traj = data
        goal = env.goal
        obstacles = None
        print(f"[VIS] Loaded Legacy Data (Goal/Obstacles might be wrong).")

    # Extract [x, y, yaw, v] from trajectory tuples
    states = []
    for item in traj:
        # item is (obs, action)
        obs = item[0]
        states.append(obs[:4])
    states = np.array(states)

    # --- 2. Plotting ---
    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw Obstacles (Blue neighbors, Black walls)
    if obstacles:
        draw_obstacles(ax, obstacles)
    else:
        draw_obstacles(ax, env)

    # Draw Goal
    draw_goal(ax, goal)

    # Draw Path
    ax.plot(states[:, 0], states[:, 1], "b-", lw=2, label="Trajectory", alpha=0.6)

    # Draw Ego Car at Start, Middle, End
    idxs = [0, len(states) // 2, len(states) - 1]
    for i in idxs:
        draw_car(ax, states[i])

    # --- 3. FIX THE VIEW (No zooming) ---
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 2.2)
    ax.set_aspect("equal")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"{os.path.basename(pkl_path)}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=int, default=0)
    args = parser.parse_args()

    with open("config_env.yaml") as f:
        cfg_env = yaml.safe_load(f)

    pkl_path = os.path.join("data", "expert_trajectories", f"episode_{args.episode:04d}.pkl")

    if os.path.exists(pkl_path):
        visualize_episode(pkl_path, cfg_env)
    else:
        print(f"File not found: {pkl_path}")