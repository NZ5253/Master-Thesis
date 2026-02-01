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
from mpc.staged_controller import StagedAtoBtoCController


def _env_obstacles_to_teb(env: ParkingEnv) -> List[Obstacle]:
    """Convert env obstacles -> MPC/TEB rectangle obstacles.

    - Uses full rectangles (no pins), so parked cars are fully covered *without gaps*.
    - Skips curbs (env doesn't treat them as collisions).
    - Skips world walls (MPC boundary term handles them).
    """
    obs_list: List[Obstacle] = []
    for o in env.obstacles.obstacles:
        kind = o.get("kind", "")
        cx, cy = float(o["x"]), float(o["y"])
        w, h = float(o["w"]), float(o["h"])
        theta = float(o.get("theta", 0.0))

        if kind == "curb":
            continue
        if max(w, h) > 1.5:  # big world walls
            continue

        obs_list.append(Obstacle(cx=cx, cy=cy, hx=w / 2.0, hy=h / 2.0, theta=theta))
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


def _compute_pose_errors(env: ParkingEnv, state_vec: np.ndarray) -> dict:
    """Compute pose error metrics consistent with env success logic.

    - Parallel: car-center error to slot center (along/lateral + euclidean)
    - Perpendicular: rear-axle error to rear-axle goal
    """
    x_ra, y_ra, yaw, v = [float(x) for x in state_vec]
    gx, gy, gyaw = [float(x) for x in env.goal]

    yaw_err = float(abs(((gyaw - yaw + np.pi) % (2 * np.pi)) - np.pi))

    bay_cfg = (env.parking_cfg.get("bay", {}) or {})
    bay_yaw = float(bay_cfg.get("yaw", 0.0))

    # vehicle geometry (must match env + obstacle geometry)
    L = float((env.vehicle_params or {}).get("length", 0.36))
    dist_to_center = L / 2.0 - 0.05

    if abs(bay_yaw) < 0.3:
        # PARALLEL: evaluate using car center vs slot center
        cx = x_ra + dist_to_center * float(np.cos(yaw))
        cy = y_ra + dist_to_center * float(np.sin(yaw))

        slot = getattr(env, "bay_center", np.array([0.0, float(bay_cfg.get("center_y", 0.13)), bay_yaw], dtype=float))
        slot_cx = float(slot[0])
        slot_cy = float(slot[1])

        along_err = abs(cx - slot_cx)
        lateral_err = abs(cy - slot_cy)
        pos_err = float(np.hypot(cx - slot_cx, cy - slot_cy))
        return {
            "pos_err": pos_err,
            "yaw_err": yaw_err,
            "along_err": float(along_err),
            "lateral_err": float(lateral_err),
        }

    # PERPENDICULAR (or any non-parallel): rear-axle error to goal
    pos_err = float(np.hypot(gx - x_ra, gy - y_ra))
    return {"pos_err": pos_err, "yaw_err": yaw_err}


def generate(cfg_full: dict, args) -> None:
    scenario = args.scenario
    n_episodes = args.episodes
    out_dir = args.out_dir
    use_hybrid = args.hybrid

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

    debug_dir = final_out_dir + "_debug"
    os.makedirs(final_out_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)

    print(f"[INFO] Output Directory: {final_out_dir}")
    print(f"[INFO] Debug Directory:  {debug_dir} (Failures saved here)\n")

    # --- Regression Seeds Setup ---
    seeds_to_run = None
    if args.regression_seeds and os.path.exists(args.regression_seeds):
        with open(args.regression_seeds, 'r') as f:
            seeds_to_run = [int(line.strip()) for line in f if line.strip().isdigit()]
        n_episodes = len(seeds_to_run)
        print(f"[INFO] Regression mode: running {n_episodes} seeds from {args.regression_seeds}")

    # --- IMPORTANT: env + MPC share the SAME env cfg now ---
    env = ParkingEnv(cfg_env)

    # Choose controller based on mode
    if use_hybrid:
        print(f"[INFO] Using HYBRID TEB+MPC controller (plan once, track)")
        print(
            f"[INFO] Using STAGED controller: A->B (receding MPC) -> WAIT -> B->C ({'HYBRID' if use_hybrid else 'baseline'})")
        controller = StagedAtoBtoCController(
            env_cfg=cfg_env,
            dt=cfg_env["dt"],
            config_path="mpc/config_mpc.yaml",
            max_obstacles=25,
            use_hybrid_for_parking=use_hybrid,
            wait_time_s=0.5,
            B_pos_tol=args.B_pos_tol,
            B_yaw_tol=args.B_yaw_tol,
            B_hold_steps=args.B_hold_steps,
            B_vel_tol=args.B_vel_tol,
        )

    else:
        print(f"[INFO] Using BASELINE MPC controller (re-plan every step)")
        teb = TEBMPC(max_obstacles=25, env_cfg=cfg_env)

    start_idx = _next_episode_index(final_out_dir)
    success_count = 0
    attempt_count = 0

    while True:
        # Loop termination conditions
        if seeds_to_run is not None:
            if attempt_count >= len(seeds_to_run):
                break
        elif success_count >= n_episodes:
            break

        attempt_count += 1

        # --- Seed Handling ---
        current_seed = None
        if seeds_to_run is not None:
            current_seed = seeds_to_run[attempt_count - 1]
            print(f"--- Attempt {attempt_count} | Seed {current_seed} ---")
        elif args.seed_base is not None:
            current_seed = int(args.seed_base) + attempt_count

        if current_seed is not None:
            np.random.seed(current_seed)

        obs = env.reset(randomize=True)
        goal_C = ParkingGoal(x=env.goal[0], y=env.goal[1], yaw=env.goal[2])

        if use_hybrid:
            # Phase 4: Pass bay_center for dynamic B calculation
            controller.reset_episode(goal_C, profile=scenario, bay_center=env.bay_center)

        episode_goal = env.goal.copy()
        episode_obstacles = [o.copy() for o in env.obstacles.obstacles]

        traj = []
        done = False
        step = 0

        # Debug trackers
        best_pos_err = float("inf")
        best_yaw_err = float("inf")
        best_step = 0
        phase_history = []

        while not done:
            state = VehicleState(x=env.state[0], y=env.state[1], yaw=env.state[2], v=env.state[3])
            goal_C = ParkingGoal(x=env.goal[0], y=env.goal[1], yaw=env.goal[2])
            obstacles = _env_obstacles_to_teb(env)

            if use_hybrid:
                action = controller.get_control(state, goal_C, obstacles, profile=scenario)
            else:
                sol = teb.solve(state, goal_C, obstacles, profile=scenario)
                action = np.array([float(sol.controls[0, 0]), float(sol.controls[0, 1])], dtype=float)

            if use_hybrid:
                ctrl_info = controller.get_info()
                mode_name = ctrl_info.get("mode", "unknown")
                if len(phase_history) == 0 or phase_history[-1] != mode_name:
                    phase_history.append(mode_name)
                    if len(phase_history) > 1:
                        print(f"  [Step {step}] Mode transition: {phase_history[-2]} -> {mode_name}", flush=True)

                # Log hold mode activation for debugging
                status = ctrl_info.get("status", "")
                if "hold" in status:
                    print(
                        f"  [Step {step}] HOLD MODE: {status}, v={env.state[3]:.4f}, pos_err={metrics.get('pos_err', 0):.4f}",
                        flush=True)

            traj.append((obs.copy(), action.copy()))
            obs, reward, done, info = env.step(action)
            step += 1

            # Compute consistent metrics
            metrics = _compute_pose_errors(env, env.state)

            # Print progress every 10 steps with velocity info
            if step % 10 == 0:
                print(f"  [Step {step}] pos_err={metrics['pos_err']:.3f}, v={env.state[3]:.4f}", flush=True)

            # --- DEBUG: track best distance / heading so far (consistent with env success) ---
            pos_err = float(metrics.get('pos_err', 0.0))
            yaw_err = float(metrics.get('yaw_err', 0.0))
            if pos_err < best_pos_err:
                best_pos_err = pos_err
                best_yaw_err = abs(yaw_err)
                best_step = step

        term = info.get("termination", "unknown")
        # --- FINAL pose errors (consistent with env success) ---
        final_metrics = _compute_pose_errors(env, env.state)
        final_pos_err = float(final_metrics.get('pos_err', 0.0))
        final_yaw_err = float(final_metrics.get('yaw_err', 0.0))

        # Detailed debug line for this attempt
        phase_str = " -> ".join(phase_history) if phase_history else "N/A"
        print(
            f"[DETAIL] Attempt {attempt_count}: term={term}, steps={step}, "
            f"final_pos_err={final_pos_err:.3f}, final_yaw_err={final_yaw_err:.3f}, "
            f"best_pos_err={best_pos_err:.3f} at step {best_step}, "
            f"best_yaw_err={best_yaw_err:.3f}"
        )

        # --- STEP 3 LOGGING: B-Stage Handoff Details (Safe Format) ---
        if use_hybrid:
            ci = controller.get_info()

            def safe_fmt(val):
                return f"{val:.3f}" if isinstance(val, (int, float)) else str(val)

            print(f"  [B] first_enter={ci.get('B_first_enter_step')}, "
                  f"wait_trigger={ci.get('B_wait_trigger_step')}, "
                  f"trigger(pos={safe_fmt(ci.get('B_pos_err_at_trigger'))}, yaw={safe_fmt(ci.get('B_yaw_err_at_trigger'))}, v={safe_fmt(ci.get('B_v_at_trigger'))}), "
                  f"profile={ci.get('B_profile_at_trigger')}")

        print(f"  Phases: {phase_str}")

        save_data = {
            "traj": traj,
            "goal": episode_goal,
            "obstacles": episode_obstacles,
            "termination": term,
            "seed": current_seed,
        }

        if use_hybrid:
            ci = controller.get_info()
            save_data["B_handoff"] = {
                "goal_B": ci.get("goal_B"),
                "B_pos_tol": args.B_pos_tol,
                "B_yaw_tol": args.B_yaw_tol,
                "B_hold_steps": args.B_hold_steps,
                "B_vel_tol": args.B_vel_tol,
                "first_enter_step": ci.get("B_first_enter_step"),
                "wait_trigger_step": ci.get("B_wait_trigger_step"),
                "pos_err_at_trigger": ci.get("B_pos_err_at_trigger"),
                "yaw_err_at_trigger": ci.get("B_yaw_err_at_trigger"),
                "v_at_trigger": ci.get("B_v_at_trigger"),
                "profile_at_trigger": ci.get("B_profile_at_trigger"),
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
    parser.add_argument("--hybrid", action="store_true",
                        help="Use hybrid TEB+MPC controller (plan once, track) instead of baseline MPC")

    # B-stage tuning knobs
    parser.add_argument("--B-pos-tol", type=float, default=0.15)
    parser.add_argument("--B-yaw-tol", type=float, default=0.30)
    parser.add_argument("--B-hold-steps", type=int, default=7)
    parser.add_argument("--B-vel-tol", type=float, default=0.05)

    # Regression harness knobs (Phase 2.4)
    parser.add_argument("--seed-base", type=int, default=None,
                        help="If set, makes env.reset reproducible per attempt: seed = seed_base + attempt_idx")
    parser.add_argument("--regression-seeds", type=str, default=None,
                        help="Path to a txt file with one integer seed per line; runs exactly those attempts and prints summary.")

    args = parser.parse_args()

    with open("config_env.yaml", "r") as f:
        cfg_full = yaml.safe_load(f)

    generate(cfg_full, args)