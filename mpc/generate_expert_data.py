# mpc/generate_expert_data.py
import os
import pickle
import copy
import numpy as np
import yaml
import argparse
from typing import List

from env.parking_env import ParkingEnv
from mpc.teb_mpc import TEBMPC, VehicleState, ParkingGoal, Obstacle


def _env_obstacles_to_teb(env: ParkingEnv) -> List[Obstacle]:
    obs_list: List[Obstacle] = []

    for o in env.obstacles.obstacles:
        cx, cy, w, h = o["x"], o["y"], o["w"], o["h"]

        is_thin = (w < 0.1) or (h < 0.1)
        if is_thin:
            obs_list.append(Obstacle(cx=cx, cy=cy, hx=w / 2.0, hy=h / 2.0))
            continue

        # Neighbor Car Logic
        if w > 0.2 and h > 0.2:
            half_w = w / 2.0
            half_h = h / 2.0
            pin_radius = 0.05

            # 4 Precise Corner Pins
            # This tells the MPC: "Here are the exact edges. Pivot around them."
            obs_list.append(Obstacle(cx - half_w, cy + half_h, hx=pin_radius, hy=pin_radius))  # TL
            obs_list.append(Obstacle(cx + half_w, cy + half_h, hx=pin_radius, hy=pin_radius))  # TR
            obs_list.append(Obstacle(cx - half_w, cy - half_h, hx=pin_radius, hy=pin_radius))  # BL
            obs_list.append(Obstacle(cx + half_w, cy - half_h, hx=pin_radius, hy=pin_radius))  # BR

    return obs_list

def _next_episode_index(out_dir: str) -> int:
    if not os.path.exists(out_dir): return 0
    max_idx = -1
    for fname in os.listdir(out_dir):
        if fname.startswith("episode_") and fname.endswith(".pkl"):
            try:
                idx = int(fname[len("episode_"): -len(".pkl")])
                max_idx = max(max_idx, idx)
            except ValueError:
                continue
    return max_idx + 1


def generate(cfg_full: dict, scenario: str, n_episodes: int, out_dir: str = None) -> None:
    # 1. Config Merge
    if "scenarios" not in cfg_full or scenario not in cfg_full["scenarios"]:
        print(f"[ERROR] Scenario '{scenario}' not found")
        return

    cfg_env = copy.deepcopy(cfg_full)
    scenario_cfg = copy.deepcopy(cfg_full["scenarios"][scenario])
    for key, val in scenario_cfg.items():
        cfg_env[key] = val

    # 2. Output Dirs
    if out_dir is None:
        final_out_dir = f"data/expert_{scenario}"
    else:
        final_out_dir = out_dir

    # Create main dir and DEBUG dir
    debug_dir = final_out_dir + "_debug"
    os.makedirs(final_out_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)

    print(f"[INFO] Output Directory: {final_out_dir}")
    print(f"[INFO] Debug Directory:  {debug_dir} (Failures saved here)\n")

    env = ParkingEnv(cfg_env)
    # Increase max_obstacles to handle corner pins (4 pins * 2 cars = 8 + extras)
    teb = TEBMPC(max_obstacles=25)

    start_idx = _next_episode_index(final_out_dir)
    success_count = 0
    attempt_count = 0

    while success_count < n_episodes:
        attempt_count += 1
        obs = env.reset(randomize=True)
        episode_goal = env.goal.copy()
        episode_obstacles = [o.copy() for o in env.obstacles.obstacles]

        traj = []
        done = False
        step = 0

        # Debug trackers
        best_pos_err = float("inf")
        best_yaw_err = float("inf")
        best_step = 0

        while not done:
            state = VehicleState(x=env.state[0], y=env.state[1], yaw=env.state[2], v=env.state[3])
            goal = ParkingGoal(x=env.goal[0], y=env.goal[1], yaw=env.goal[2])
            obstacles = _env_obstacles_to_teb(env)

            try:
                sol = teb.solve(state, goal, obstacles, profile=scenario)
                action = np.array([float(sol.controls[0, 0]), float(sol.controls[0, 1])])
            except Exception:
                action = np.zeros(2)

            traj.append((obs.copy(), action.copy()))
            obs, reward, done, info = env.step(action)
            step += 1

            # --- DEBUG: track best distance / heading so far ---
            pos_err = info.get("pos_err", None)
            yaw_err = info.get("yaw_err", None)
            if pos_err is not None:
                if pos_err < best_pos_err:
                    best_pos_err = float(pos_err)
                    best_yaw_err = float(abs(yaw_err)) if yaw_err is not None else best_yaw_err
                    best_step = step

        term = info.get("termination", "unknown")
        # --- FINAL pose errors (using env state & goal) ---
        x, y, yaw, v = env.state
        gx, gy, gyaw = env.goal

        final_pos_err = float(np.hypot(gx - x, gy - y))
        final_yaw_err = float(abs(((gyaw - yaw + np.pi) % (2 * np.pi)) - np.pi))

        # Detailed debug line for this attempt
        print(
            f"[DETAIL] Attempt {attempt_count}: term={term}, steps={step}, "
            f"final_pos_err={final_pos_err:.3f}, final_yaw_err={final_yaw_err:.3f}, "
            f"best_pos_err={best_pos_err:.3f} at step {best_step}, "
            f"best_yaw_err={best_yaw_err:.3f}"
        )


        save_data = {
            "traj": traj,
            "goal": episode_goal,
            "obstacles": episode_obstacles,
            "termination": term
        }

        # --- SAVE LOGIC ---
        if term == "success":
            file_idx = start_idx + success_count
            with open(os.path.join(final_out_dir, f"episode_{file_idx:04d}.pkl"), "wb") as f:
                pickle.dump(save_data, f)

            print(f"[SUCCESS] Saved Episode {file_idx} (Steps: {step})")
            success_count += 1
        else:
            # Save FAILURE to debug folder
            fail_name = f"fail_{attempt_count:04d}_{term}.pkl"
            with open(os.path.join(debug_dir, fail_name), "wb") as f:
                pickle.dump(save_data, f)

            print(f"[FAIL] Attempt {attempt_count}: {term} (Steps: {step}) -> Saved to debug")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", "-n", type=int, default=50)
    parser.add_argument("--scenario", type=str, default="perpendicular")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    with open("config_env.yaml", "r") as f:
        cfg_full = yaml.safe_load(f)

    generate(cfg_full, args.scenario, args.episodes, args.out_dir)