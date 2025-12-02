# simulation/visualize_mpc_episode.py
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon

from env.parking_env import ParkingEnv
from mpc.teb_mpc import TEBMPC, VehicleState, ParkingGoal, Obstacle

CAR_LENGTH = 0.36
CAR_WIDTH = 0.26


def draw_car(ax, state, length=0.36, width=0.26):
    x_rear, y_rear, yaw = state[0], state[1], state[2]

    # Calculate the True Geometric Center
    # (Approx 0.13m ahead of the rear axle for this car size)
    dist_to_center = length / 2.0 - 0.05

    x_center = x_rear + dist_to_center * np.cos(yaw)
    y_center = y_rear + dist_to_center * np.sin(yaw)

    # Draw the box centered at the TRUE center
    corners = np.array([
        [length / 2, width / 2], [length / 2, -width / 2],
        [-length / 2, -width / 2], [-length / 2, width / 2]
    ])
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    world_corners = corners @ R.T + np.array([x_center, y_center])

    poly = Polygon(world_corners, fill=False, edgecolor='blue', linewidth=1.5)
    ax.add_patch(poly)
    ax.arrow(x_rear, y_rear, 0.2 * c, 0.2 * s, head_width=0.05, fc='blue', ec='blue')

# simulation/visualize_mpc_episode.py

def draw_obstacles(ax, env_or_obstacles):
    if hasattr(env_or_obstacles, "obstacles"):
        obstacles = env_or_obstacles.obstacles.obstacles
    else:
        obstacles = env_or_obstacles

    for o in obstacles:
        cx, cy, w, h = o["x"], o["y"], o["w"], o["h"]
        theta = o.get("theta", 0.0)

        # Style walls vs cars
        if w > 1.5 or h > 1.5:
            color = 'black';
            alpha = 0.3;
            zorder = 1
        else:
            color = 'skyblue';
            alpha = 0.8;
            zorder = 5

        # 1. Create corners centered at 0,0
        corners = np.array([
            [w / 2, h / 2], [w / 2, -h / 2],
            [-w / 2, -h / 2], [-w / 2, h / 2]
        ])

        # 2. Rotate them manually
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        rotated_corners = corners @ R.T + np.array([cx, cy])

        # 3. Draw Polygon (Not Rectangle)
        from matplotlib.patches import Polygon  # Ensure this is imported
        poly = Polygon(rotated_corners, closed=True, facecolor=color, alpha=alpha, zorder=zorder)
        ax.add_patch(poly)

        # Border
        border = Polygon(rotated_corners, closed=True, fill=False, edgecolor='black', linewidth=1, zorder=zorder)
        ax.add_patch(border)

def draw_goal(ax, env_or_goal):
    if hasattr(env_or_goal, "goal"):
        gx, gy, gyaw = env_or_goal.goal
    else:
        gx, gy, gyaw = env_or_goal
    ax.plot(gx, gy, "rx", markersize=10, markeredgewidth=2, zorder=10)


def _env_obstacles_to_teb(env: ParkingEnv):
    """
    Converts environment obstacles into TEB obstacles.
    Updated to use 'Corner Pins' for neighbor cars to match generate_expert_data.py
    """
    obs_list = []

    for o in env.obstacles.obstacles:
        cx, cy, w, h = o["x"], o["y"], o["w"], o["h"]

        # 1. Skip thin walls or far away objects (Optimization)
        is_thin = (w < 0.1) or (h < 0.1)
        if is_thin:
            # Walls are fine as simple rectangles/circles
            obs_list.append(Obstacle(cx=cx, cy=cy, hx=w / 2.0, hy=h / 2.0))
            continue

        # 2. CHECK: Is this a Neighbor Car? (Fat object)
        # Split into 4 corner pins + center
        if w > 0.2 and h > 0.2:
            half_w = w / 2.0
            half_h = h / 2.0
            pin_radius = 0.05  # 5cm radius

            # Corner 1: Top-Left
            obs_list.append(Obstacle(cx=cx - half_w, cy=cy + half_h, hx=pin_radius, hy=pin_radius))
            # Corner 2: Top-Right
            obs_list.append(Obstacle(cx=cx + half_w, cy=cy + half_h, hx=pin_radius, hy=pin_radius))
            # Corner 3: Bottom-Left
            obs_list.append(Obstacle(cx=cx - half_w, cy=cy - half_h, hx=pin_radius, hy=pin_radius))
            # Corner 4: Bottom-Right
            obs_list.append(Obstacle(cx=cx + half_w, cy=cy - half_h, hx=pin_radius, hy=pin_radius))

    return obs_list

def run_episode(full_cfg, scenario_name="perpendicular"):
    # Merge scenario specific config
    cfg_env = full_cfg.copy()
    if "scenarios" in full_cfg and scenario_name in full_cfg["scenarios"]:
        cfg_env.update(full_cfg["scenarios"][scenario_name])

    env = ParkingEnv(cfg_env)
    teb = TEBMPC()

    print(f"Running scenario: {scenario_name}")
    obs = env.reset(randomize=True)
    done = False

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))

    step = 0
    while not done:
        x, y, yaw, v = env.state
        state = VehicleState(x=x, y=y, yaw=yaw, v=v)
        goal = ParkingGoal(x=env.goal[0], y=env.goal[1], yaw=env.goal[2])
        obstacles = _env_obstacles_to_teb(env)

        try:
            # Pass scenario_name as profile so MPC uses correct weights
            sol = teb.solve(state, goal, obstacles, profile=scenario_name)
            u = teb.first_control(sol)
            control = np.array(u, dtype=float)
        except Exception as e:
            control = np.zeros(2)

        obs, r, done, info = env.step(control)

        ax.clear()
        draw_obstacles(ax, env)
        draw_goal(ax, env)
        draw_car(ax, env.state)

        ax.set_xlim(-2.2, 2.2)
        ax.set_ylim(-2.2, 2.2)
        ax.set_aspect("equal")
        ax.set_title(f"Scenario: {scenario_name} | Step {step} | {info.get('termination', '')}")

        plt.pause(0.01)
        step += 1

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="perpendicular", help="perpendicular or parallel")
    args = parser.parse_args()

    with open("config_env.yaml", "r") as f:
        full_cfg = yaml.safe_load(f)

    run_episode(full_cfg, scenario_name=args.scenario)