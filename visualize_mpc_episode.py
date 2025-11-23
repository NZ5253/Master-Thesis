import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon

from env.parking_env import ParkingEnv
from mpc.mpc_solver import MPCSolver


CAR_LENGTH = 0.36  # same as in config_mpc.yaml
CAR_WIDTH = 0.26


def draw_car(ax, state, length=CAR_LENGTH, width=CAR_WIDTH):
    """Draw the car as a rotated rectangle."""
    x, y, yaw, v = state

    # rectangle in car frame (center at 0,0)
    corners = np.array([
        [ length / 2,  width / 2],
        [ length / 2, -width / 2],
        [-length / 2, -width / 2],
        [-length / 2,  width / 2],
    ])

    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    world_corners = corners @ R.T + np.array([x, y])

    poly = Polygon(world_corners, fill=False)
    ax.add_patch(poly)

    # draw a small dot at the car center
    ax.plot(x, y, "ko", markersize=3)


def draw_obstacles(ax, env):
    """Draw rectangular obstacles from env.obstacles.obstacles."""
    for o in env.obstacles.obstacles:
        cx, cy, w, h = o["x"], o["y"], o["w"], o["h"]
        rect = Rectangle(
            (cx - w / 2.0, cy - h / 2.0),
            w,
            h,
            fill=True,
            alpha=0.2,
        )
        ax.add_patch(rect)


def draw_goal(ax, env):
    gx, gy, gyaw = env.goal
    ax.plot(gx, gy, "rx", markersize=8)
    ax.text(gx, gy, " goal", color="r")


def run_episode(cfg_env, cfg_mpc, scenario_name="parallel"):
    env = ParkingEnv(cfg_env)
    mpc = MPCSolver(cfg_env["vehicle"], cfg_mpc)

    obs = env.reset(randomize=True)
    done = False

    plt.ion()
    fig, ax = plt.subplots()

    step = 0
    while not done:
        state = env.state.copy()

        # ----- MPC control -----
        try:
            res = mpc.solve(state, cfg_env["goal"])
            steer, accel = mpc.extract_first_control(res)
            control = np.array([steer, accel], dtype=float)
        except Exception as e:
            print(f"MPC error at step {step}: {e}")
            control = np.zeros(2, dtype=float)

        # ----- step env -----
        obs, r, done, info = env.step(control)

        # ----- draw -----
        ax.clear()
        draw_obstacles(ax, env)
        draw_goal(ax, env)
        draw_car(ax, env.state)

        # world bounds: assume 4x4 centered at (0,0)
        ax.set_aspect("equal")
        ax.set_xlim(-2.2, 2.2)
        ax.set_ylim(-2.2, 2.2)
        ax.set_title(f"{scenario_name} | step {step} | termination: {info.get('termination', '')}")

        plt.pause(0.01)
        step += 1

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        type=str,
        default="parallel",
        choices=["parallel", "perpendicular"],
        help="Scenario (if using config_loader).",
    )
    args = parser.parse_args()

    # --- load env config ---
    try:
        # if you are using the unified config_env.yaml with scenarios + config_loader.py
        from config_loader import load_env_config

        cfg_env = load_env_config(args.scenario)
    except ImportError:
        # fallback: plain single-scenario config_env.yaml
        with open("config_env.yaml", "r") as f:
            cfg_env = yaml.safe_load(f)

    # --- load MPC config ---
    mpc_cfg_path = os.path.join("mpc", "config_mpc.yaml")
    with open(mpc_cfg_path, "r") as f:
        cfg_mpc = yaml.safe_load(f)

    run_episode(cfg_env, cfg_mpc, scenario_name=args.scenario)
