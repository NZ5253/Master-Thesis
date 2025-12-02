# simulation/visualize_from_expert_data.py
import os
import pickle
import yaml
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from env.parking_env import ParkingEnv
from visualize_mpc_episode import draw_car, draw_obstacles, draw_goal, _env_obstacles_to_teb


def draw_teb_obstacles(ax, env):
    """Draws the neighbor 'Pins' (Red)."""
    teb_obstacles = _env_obstacles_to_teb(env)
    for o in teb_obstacles:
        # base radius (not really needed anymore, but fine to keep)
        radius = (o.hx + o.hy) / 2.0

        # walls (big hx/hy) → slightly bigger circle
        if max(o.hx, o.hy) > 1.0:
            radius = 0.30
        # parked cars / pins → smaller circle
        else:
            radius = 0.07

        circle = Circle((o.cx, o.cy), radius, color='red', alpha=0.4, zorder=20)
        ax.add_patch(circle)
        ax.plot(o.cx, o.cy, 'k.', markersize=2, zorder=21)



def draw_ego_circles(ax, state, vehicle_cfg):
    """
    Draws the 4 internal circles of the Ego Vehicle (Green).
    """
    x, y, yaw = state[0], state[1], state[2]
    L = float(vehicle_cfg.get("length", 0.36))
    W = float(vehicle_cfg.get("width", 0.26))
    radius = W / 4

    # Match the logic in teb_mpc.py
    dist_ra_to_center = L / 2.0 - 0.05
    step = L / 8.0

    # 4 offsets along the spine
    offsets = [
        dist_ra_to_center + 3 * step,  # Front
        dist_ra_to_center + 1 * step,
        dist_ra_to_center - 1 * step,
        dist_ra_to_center - 3 * step  # Rear
    ]

    c_theta = np.cos(yaw)
    s_theta = np.sin(yaw)

    for off in offsets:
        cx = x + off * c_theta
        cy = y + off * s_theta
        # Draw Green Circle (Ego Body)
        circle = Circle((cx, cy), radius, color='#00FF00', alpha=0.5, zorder=25)
        ax.add_patch(circle)
        # Center dot
        ax.plot(cx, cy, 'g.', markersize=2, zorder=26)


def visualize_episode(pkl_path: str, cfg_full: dict):
    # Load Configs
    cfg_env = cfg_full.copy()
    vehicle_cfg = cfg_full.get("vehicle", {})

    # Use parallel profile for obstacle generation if needed
    if "scenarios" in cfg_full and "parallel" in cfg_full["scenarios"]:
        cfg_env.update(cfg_full["scenarios"]["parallel"])

    env = ParkingEnv(cfg_env)

    print(f"[VIS] Loading: {pkl_path}")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        traj = data["traj"]
        goal = data["goal"]
        # overwrite obstacles with the ones from this episode
        env.obstacles.obstacles = data["obstacles"]
        term = data.get("termination", "?")
        print(f"[VIS] Steps: {len(traj)}, Termination: {term}")
    else:
        traj = data
        goal = env.goal

    # extract states (x, y, yaw, v)
    states = np.array([item[0][:4] for item in traj])

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))

    for k, state in enumerate(states):
        ax.clear()

        # 1. Real obstacles (blue rectangles)
        draw_obstacles(ax, env)

        # 2. TEB safety zones (red circles)
        draw_teb_obstacles(ax, env)

        # 3. goal
        draw_goal(ax, goal)

        # 4. trajectory so far
        if k > 0:
            ax.plot(states[:k+1, 0], states[:k+1, 1], "b-", lw=2, alpha=0.6)

        # 5. ego car at this step
        draw_car(ax, state)
        draw_ego_circles(ax, state, vehicle_cfg)

        ax.set_xlim(-2.2, 2.2)
        ax.set_ylim(-2.2, 2.2)
        ax.set_aspect("equal")
        ax.set_title(f"{os.path.basename(pkl_path)} | step {k+1}/{len(states)}")

        plt.pause(0.02)   # replay speed (smaller = faster)

    plt.ioff()
    plt.show()



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to .pkl file")
    args = parser.parse_args()

    # Load Full Config to get Vehicle Params
    with open("config_env.yaml") as f:
        cfg_full = yaml.safe_load(f)

    # Also need mpc config for vehicle dims? Usually in config_mpc.
    # But let's assume config_env has basic vehicle dims or we default them.
    # If vehicle dims are missing, the visualizer defaults to 0.36/0.26

    if os.path.exists(args.file):
        visualize_episode(args.file, cfg_full)
    else:
        print(f"File not found: {args.file}")