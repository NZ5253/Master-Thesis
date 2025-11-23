# mpc/generate_expert_data.py
import os
import pickle
from typing import List

import numpy as np

from env.parking_env import ParkingEnv
from mpc.teb_mpc import TEBMPC, VehicleState, ParkingGoal, Obstacle


def _env_obstacles_to_teb(env) -> List[Obstacle]:
    """Convert env.obstacles.obstacles list-of-dicts to TEB Obstacle objects.

    We skip thin outer walls near the world border and only pass
    internal obstacles (e.g. parked neighbour cars, curb) to TEB.
    """
    obs_list: List[Obstacle] = []
    for o in env.obstacles.obstacles:
        cx = o["x"]
        cy = o["y"]
        w = o["w"]
        h = o["h"]

        # Heuristic: outer walls are long & thin and sit near |x| or |y| ≈ world border
        is_thin = (w < 0.1) or (h < 0.1)
        near_border = (abs(cx) > 1.8) or (abs(cy) > 1.8)
        is_wall = is_thin and near_border
        if is_wall:
            continue  # don't include walls in the TEB cost

        hx = w / 2.0
        hy = h / 2.0
        obs_list.append(Obstacle(cx=cx, cy=cy, hx=hx, hy=hy))

    return obs_list


def _next_episode_index(out_dir: str) -> int:
    """
    Return the next episode index so we don't overwrite existing data.

    If out_dir already contains episode_0000.pkl..episode_0041.pkl,
    this returns 42, and the next run will create episode_0042.pkl etc.
    """
    if not os.path.exists(out_dir):
        return 0
    max_idx = -1
    for fname in os.listdir(out_dir):
        if fname.startswith("episode_") and fname.endswith(".pkl"):
            try:
                idx = int(fname[len("episode_") : -len(".pkl")])
                max_idx = max(max_idx, idx)
            except ValueError:
                continue
    return max_idx + 1


def generate(
    cfg_env,
    n_episodes: int = 200,
    out_dir: str = "data/expert_trajectories",
) -> None:
    """
    Generate expert trajectories using the TEB-MPC controller (perpendicular only).

    - Uses env configuration `cfg_env`.
    - Does **not** overwrite existing episodes; it appends.
    - Only prints per-episode summary (no per-step logs).
    """
    os.makedirs(out_dir, exist_ok=True)

    env = ParkingEnv(cfg_env)
    teb = TEBMPC()  # loads mpc/config_mpc.yaml internally

    start_idx = _next_episode_index(out_dir)

    if start_idx == 0:
        print("[INFO] No existing episodes found, will start from 0")
    else:
        print(f"[INFO] I found last episode {start_idx-1}, will start from {start_idx}")

    for ep_offset in range(n_episodes):
        ep_idx = start_idx + ep_offset
        print(f"[EPISODE {ep_idx}] starting...")

        obs = env.reset(randomize=True)
        traj = []
        done = False
        step = 0

        while not done:
            state_arr = env.state.copy()
            state = VehicleState(
                x=state_arr[0],
                y=state_arr[1],
                yaw=state_arr[2],
                v=state_arr[3],
            )
            goal = ParkingGoal(
                x=env.goal[0],
                y=env.goal[1],
                yaw=env.goal[2],
            )
            obstacles = _env_obstacles_to_teb(env)

            try:
                sol = teb.solve(state, goal, obstacles, profile="perpendicular")
                steer, accel = sol.controls[0, 0], sol.controls[0, 1]
                control = np.array([steer, accel], dtype=float)
            except Exception as e:
                print(f"  TEB-MPC error at step {step}: {e}")
                control = np.zeros(2, dtype=float)

            traj.append((state_arr.copy(), control.copy()))
            obs, r, done, info = env.step(control)
            step += 1

        term = info.get("termination", None)

        # debug: how close did we end?
        last_state = state_arr
        x, y, yaw, v = last_state
        gx, gy, gyaw = env.goal
        pos_err = np.hypot(gx - x, gy - y)
        yaw_err = abs(((gyaw - yaw + np.pi) % (2 * np.pi)) - np.pi)

        print(
            f"[EPISODE {ep_idx}] finished in {step} steps, "
            f"termination={term}, pos_err={pos_err:.3f}, yaw_err={yaw_err:.3f}"
        )

        out_path = os.path.join(out_dir, f"episode_{ep_idx:04d}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(traj, f)


if __name__ == "__main__":
    import argparse
    import yaml

    # --------- device selection: try CUDA, else CPU ----------
    try:
        import torch

        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("[DEVICE] Using CUDA")
        else:
            device = torch.device("cpu")
            print("[DEVICE] CUDA not available, falling back to CPU")
    except Exception:
        device = "cpu"
        print("[DEVICE] PyTorch / CUDA not available, using CPU")

    parser = argparse.ArgumentParser(description="Generate expert trajectories with TEB-MPC (perpendicular only).")
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
        help="Output directory for expert trajectories.",
    )

    args = parser.parse_args()

    # ----- load env config (perpendicular only) -----
    with open("config_env.yaml", "r") as f:
        cfg_env = yaml.safe_load(f)

    # ----- run generation -----
    generate(cfg_env, n_episodes=args.episodes, out_dir=args.out_dir)
