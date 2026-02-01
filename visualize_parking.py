import os
import pickle
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle

from env.parking_env import ParkingEnv
from mpc.teb_mpc import TEBMPC, VehicleState, ParkingGoal, Obstacle


# ============================================================
#  Car geometry & drawing helpers
# ============================================================

def _get_vehicle_dims(vehicle_cfg):
    """Helper to read length/width from config with sane defaults."""
    length = float(vehicle_cfg.get("length", 0.36))
    width = float(vehicle_cfg.get("width", 0.26))
    return length, width


def draw_car(ax, state, vehicle_cfg=None):
    """
    Draw the ego vehicle as a blue rectangle, using the same
    rear-axle → center logic as the env and MPC.
    """
    x_rear, y_rear, yaw = state[0], state[1], state[2]

    if vehicle_cfg is None:
        length, width = 0.36, 0.26
    else:
        length, width = _get_vehicle_dims(vehicle_cfg)

    # Rear axle to geometric center:
    #   center_offset = L/2 - rear_overhang
    # For CRC car: L = 0.36, rear_overhang ≈ 0.05 → 0.13 m
    dist_to_center = length / 2.0 - 0.05

    x_center = x_rear + dist_to_center * np.cos(yaw)
    y_center = y_rear + dist_to_center * np.sin(yaw)

    # Rectangle corners in local center frame
    corners = np.array([
        [length / 2, width / 2],
        [length / 2, -width / 2],
        [-length / 2, -width / 2],
        [-length / 2, width / 2],
    ])
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    world_corners = corners @ R.T + np.array([x_center, y_center])

    poly = Polygon(world_corners, fill=False, edgecolor='blue', linewidth=1.5)
    ax.add_patch(poly)

    # Rear axle heading arrow
    ax.arrow(
        x_rear, y_rear,
        0.2 * c, 0.2 * s,
        head_width=0.05, fc='blue', ec='blue'
    )


def draw_ego_circles(ax, state, vehicle_cfg):
    """
    Draw 4 internal circles (green) that approximate the car body.
    Updated to match TEBMPC logic (step = length/8).
    """
    x_rear, y_rear, yaw = state[0], state[1], state[2]
    length, width = _get_vehicle_dims(vehicle_cfg)

    # Rear axle → geometric center (must match env / collision model)
    dist_ra_to_center = length / 2.0 - 0.05

    # Match teb_mpc.py:
    # Use 4 spine circles, enlarged just enough so the union covers the rectangle corners.
    step = length / 8.0
    radius = float(np.hypot(width / 2.0, step))

    # Offsets from rear axle along the car axis (matching teb_mpc.py)
    offsets = [
        dist_ra_to_center + 3 * step,  # front center
        dist_ra_to_center + 1 * step,  # mid-front center
        dist_ra_to_center - 1 * step,  # mid-rear center
        dist_ra_to_center - 3 * step,  # rear center
    ]

    c_theta = np.cos(yaw)
    s_theta = np.sin(yaw)

    for off in offsets:
        cx = x_rear + off * c_theta
        cy = y_rear + off * s_theta

        circ = Circle((cx, cy), radius, color='#00FF00', alpha=0.3, zorder=25)
        ax.add_patch(circ)
        ax.plot(cx, cy, 'g.', markersize=2, zorder=26)


# ============================================================
#  Obstacles & goals
# ============================================================

def draw_obstacles(ax, env_or_obstacles):
    """
    Draw rectangular obstacles as rotated polygons.
    - Walls: darker / lower z-order
    - Parked cars: skyblue / higher z-order
    """
    if hasattr(env_or_obstacles, "obstacles"):
        obstacles = env_or_obstacles.obstacles.obstacles
    else:
        obstacles = env_or_obstacles

    for o in obstacles:
        cx, cy, w, h = o["x"], o["y"], o["w"], o["h"]
        theta = o.get("theta", 0.0)
        kind = o.get("kind", None)

        # Style walls vs cars vs curb
        if kind == "curb":
            # Soft curb: dark gray bar just under the parked cars
            color = 'dimgray'
            alpha = 0.7
            zorder = 4
        elif w > 1.5 or h > 1.5:
            # World walls
            color = 'black'
            alpha = 0.3
            zorder = 1
        else:
            # Parked cars / random boxes
            color = 'skyblue'
            alpha = 0.8
            zorder = 5

        # corners in local frame (centered)
        corners = np.array([
            [w / 2, h / 2],
            [w / 2, -h / 2],
            [-w / 2, -h / 2],
            [-w / 2, h / 2],
        ])

        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        rotated_corners = corners @ R.T + np.array([cx, cy])

        poly = Polygon(rotated_corners, closed=True,
                       facecolor=color, alpha=alpha, zorder=zorder)
        ax.add_patch(poly)

        border = Polygon(rotated_corners, closed=True,
                         fill=False, edgecolor='black', linewidth=1, zorder=zorder)
        ax.add_patch(border)


def draw_goal(ax, env_or_goal):
    """
    Draw the desired CAR CENTER as a red X.
    """
    # --- get rear-axle goal pose + vehicle length ---
    if hasattr(env_or_goal, "goal"):
        # live env
        gx, gy, gyaw = env_or_goal.goal
        L = float(env_or_goal.vehicle_params.get("length", 0.36))
    else:
        # numpy array / tuple (gx, gy, gyaw)
        gx, gy, gyaw = env_or_goal
        L = 0.36  # CRC default length

    # shift from rear axle to CAR CENTER (must match your env/obstacle code)
    dist_to_center = L / 2.0 - 0.05
    cx = gx + dist_to_center * np.cos(gyaw)
    cy = gy + dist_to_center * np.sin(gyaw)

    # Draw the CAR CENTER goal
    ax.plot(cx, cy, "rx", markersize=10, markeredgewidth=2, zorder=10)


# ============================================================
#  TEB obstacle conversion & debugging drawing
# ============================================================

def _env_obstacles_to_teb(env: ParkingEnv):
    """
    Convert env obstacles -> rectangle obstacles for MPC/TEB visualization.
    Matches the logic in generate_expert_data.py
    """
    obs_list = []
    for o in env.obstacles.obstacles:
        kind = o.get("kind", "")
        cx, cy = float(o["x"]), float(o["y"])
        w, h = float(o["w"]), float(o["h"])
        theta = float(o.get("theta", 0.0))

        # Skip curbs (env doesn't treat them as hard collisions)
        if kind == "curb":
            continue
        # Skip world walls (handled by boundary constraints)
        if max(w, h) > 1.5:
            continue

        # Pass full rectangle parameters
        obs_list.append(Obstacle(cx=cx, cy=cy, hx=w / 2.0, hy=h / 2.0, theta=theta))

    return obs_list


def draw_teb_obstacles(ax, env):
    """
    Draw MPC/TEB obstacles (rectangles) as red outlines.
    This shows exactly what the SDF collision checker sees.
    """
    teb_obstacles = _env_obstacles_to_teb(env)
    for o in teb_obstacles:
        # Get rotation matrix
        c, s = np.cos(o.theta), np.sin(o.theta)
        R = np.array([[c, -s], [s, c]])

        # Local corners from half-extents
        corners_local = np.array([
            [o.hx, o.hy],
            [o.hx, -o.hy],
            [-o.hx, -o.hy],
            [-o.hx, o.hy]
        ])

        # Transform to world
        corners_world = (corners_local @ R.T) + np.array([o.cx, o.cy])

        # Draw red outline
        poly = Polygon(corners_world, closed=True,
                       facecolor='red', alpha=0.25,
                       edgecolor='red', lw=1.5, zorder=20)
        ax.add_patch(poly)
        ax.plot(o.cx, o.cy, 'k.', markersize=2, zorder=21)


# ============================================================
#  MODE 1: Visualize MPC episode live
# ============================================================

def run_mpc_episode(full_cfg, scenario_name="perpendicular"):
    """
    Runs TEB-MPC in the chosen scenario and animates:
      - obstacles
      - goal
      - ego rectangle
      - ego 4-circle model
    """
    # Merge scenario-specific config into env config
    cfg_env = full_cfg.copy()
    if "scenarios" in full_cfg and scenario_name in full_cfg["scenarios"]:
        cfg_env.update(full_cfg["scenarios"][scenario_name])

    env = ParkingEnv(cfg_env)
    teb = TEBMPC(env_cfg=cfg_env)

    vehicle_cfg = full_cfg.get("vehicle", {})

    print(f"[MPC] Running scenario: {scenario_name}")
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
            # profile=scenario_name lets TEB pick different weights for parallel/perpendicular
            sol = teb.solve(state, goal, obstacles, profile=scenario_name)
            u = teb.first_control(sol)
            control = np.array(u, dtype=float)
        except Exception as e:
            print(f"[MPC] Solver error at step {step}: {e}")
            control = np.zeros(2)

        obs, r, done, info = env.step(control)

        ax.clear()
        draw_obstacles(ax, env)
        draw_goal(ax, env)
        draw_car(ax, env.state, vehicle_cfg=vehicle_cfg)
        draw_ego_circles(ax, env.state, vehicle_cfg)
        draw_teb_obstacles(ax, env)  # Added to visualize MPC obstacles live

        ax.set_xlim(-2.2, 2.2)
        ax.set_ylim(-2.2, 2.2)
        ax.set_aspect("equal")
        ax.set_title(f"Scenario: {scenario_name} | Step {step} | {info.get('termination', '')}")

        plt.pause(0.01)
        step += 1

    plt.ioff()
    plt.show()


# ============================================================
#  MODE 2: Visualize from expert data (.pkl)
# ============================================================

def visualize_episode_from_expert(pkl_path: str, cfg_full: dict, scenario_name: str = "parallel"):
    """
    Replays a stored expert episode:
      - traj: list[(obs, action, ...)]
      - goal: [x, y, yaw]
      - obstacles: env.obstacles.obstacles for that episode
    """
    cfg_env = cfg_full.copy()
    if "scenarios" in cfg_full and scenario_name in cfg_full["scenarios"]:
        cfg_env.update(cfg_full["scenarios"][scenario_name])

    env = ParkingEnv(cfg_env)
    vehicle_cfg = cfg_full.get("vehicle", {})

    print(f"[VIS] Loading: {pkl_path}")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        traj = data["traj"]
        goal = data["goal"]
        env.obstacles.obstacles = data["obstacles"]
        term = data.get("termination", "?")
        print(f"[VIS] Steps: {len(traj)}, Termination: {term}")
    else:
        traj = data
        goal = env.goal

    states = np.array([item[0][:4] for item in traj])

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))

    for k, state in enumerate(states):
        ax.clear()

        # 1. Real obstacles (blue rectangles)
        draw_obstacles(ax, env)

        # 2. TEB safety zones (red outlines)
        draw_teb_obstacles(ax, env)

        # 3. Goal
        draw_goal(ax, goal)

        # 4. Trajectory so far
        if k > 0:
            ax.plot(states[:k + 1, 0], states[:k + 1, 1], "b-", lw=2, alpha=0.6)

        # 5. Ego rectangle + 4 circles
        draw_car(ax, state, vehicle_cfg=vehicle_cfg)
        draw_ego_circles(ax, state, vehicle_cfg)

        ax.set_xlim(-2.2, 2.2)
        ax.set_ylim(-2.2, 2.2)
        ax.set_aspect("equal")
        ax.set_title(f"{os.path.basename(pkl_path)} | step {k + 1}/{len(states)}")

        plt.pause(0.02)

    plt.ioff()
    plt.show()


# ============================================================
#  CLI entrypoint
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to expert .pkl file (if set, replays expert instead of running MPC)",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="perpendicular",
        help="Scenario name: perpendicular or parallel",
    )
    args = parser.parse_args()

    with open("config_env.yaml", "r") as f:
        full_cfg = yaml.safe_load(f)

    if args.file is not None and os.path.exists(args.file):
        visualize_episode_from_expert(args.file, full_cfg, scenario_name=args.scenario)
    elif args.file is not None:
        print(f"[ERROR] File not found: {args.file}")
    else:
        run_mpc_episode(full_cfg, scenario_name=args.scenario)