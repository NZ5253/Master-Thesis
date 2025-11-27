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


def draw_car(ax, state, length=CAR_LENGTH, width=CAR_WIDTH):
    """Draw the car as a blue rotated rectangle."""
    arr = np.asarray(state, dtype=float).flatten()
    if arr.shape[0] < 3: return
    x, y, yaw = arr[0], arr[1], arr[2]

    corners = np.array([
        [length / 2, width / 2], [length / 2, -width / 2],
        [-length / 2, -width / 2], [-length / 2, width / 2],
    ])
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    world_corners = corners @ R.T + np.array([x, y])

    # Blue outline for the Ego Car
    poly = Polygon(world_corners, fill=False, edgecolor='blue', linewidth=1.5, zorder=10)
    ax.add_patch(poly)

    # Direction arrow
    ax.arrow(x, y, 0.2 * c, 0.2 * s, head_width=0.05, head_length=0.05, fc='blue', ec='blue', zorder=10)


def draw_obstacles(ax, env_or_obstacles):
    """Draws walls in black/gray and neighbor cars in light blue."""
    if hasattr(env_or_obstacles, "obstacles"):
        obstacles = env_or_obstacles.obstacles.obstacles
    else:
        obstacles = env_or_obstacles

    for o in obstacles:
        cx, cy, w, h = o["x"], o["y"], o["w"], o["h"]

        # Heuristic: Walls are usually long or thick (> 1.5m). 
        # Cars are small (~0.26 x 0.36).
        if w > 1.5 or h > 1.5:
            # Wall
            color = 'black'
            alpha = 0.3
            zorder = 1
        else:
            # Neighbor Car / Obstacle
            color = 'skyblue'
            alpha = 0.8
            zorder = 5

        rect = Rectangle(
            (cx - w / 2.0, cy - h / 2.0), w, h,
            fill=True, color=color, alpha=alpha, zorder=zorder
        )
        ax.add_patch(rect)

        # Border
        ax.add_patch(Rectangle(
            (cx - w / 2.0, cy - h / 2.0), w, h,
            fill=False, color='black', linewidth=1, zorder=zorder
        ))


def draw_goal(ax, env_or_goal):
    """Draw goal as a red 'x'."""
    if hasattr(env_or_goal, "goal"):
        gx, gy, gyaw = env_or_goal.goal
    else:
        gx, gy, gyaw = env_or_goal

    ax.plot(gx, gy, "rx", markersize=10, markeredgewidth=2, zorder=10)


def _env_obstacles_to_teb(env: ParkingEnv):
    """Convert env obstacles to TEB Obstacle objects."""
    obs_list = []
    for o in env.obstacles.obstacles:
        cx, cy, w, h = o["x"], o["y"], o["w"], o["h"]
        obs_list.append(Obstacle(cx=cx, cy=cy, hx=w / 2.0, hy=h / 2.0))
    return obs_list


def run_episode(cfg_env, scenario_name="perpendicular"):
    env = ParkingEnv(cfg_env)
    teb = TEBMPC()
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
            sol = teb.solve(state, goal, obstacles, profile="perpendicular")
            u = teb.first_control(sol)
            control = np.array(u, dtype=float)
        except Exception as e:
            control = np.zeros(2)

        obs, r, done, info = env.step(control)

        ax.clear()
        draw_obstacles(ax, env)
        draw_goal(ax, env)
        draw_car(ax, env.state)

        # FORCE 4x4 WORLD VIEW
        ax.set_xlim(-2.2, 2.2)
        ax.set_ylim(-2.2, 2.2)
        ax.set_aspect("equal")
        ax.set_title(f"Step {step} | {info.get('termination', '')}")

        plt.pause(0.01)
        step += 1

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="perpendicular")
    args = parser.parse_args()

    with open("config_env.yaml", "r") as f:
        cfg_env = yaml.safe_load(f)

    run_episode(cfg_env, scenario_name=args.scenario)