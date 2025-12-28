#!/usr/bin/env python3
"""Test different fixed spawn positions to find optimal ones."""

import numpy as np
import yaml
import copy
import os
from env.parking_env import ParkingEnv
from mpc.teb_mpc import TEBMPC, VehicleState, ParkingGoal
from mpc.generate_expert_data import _env_obstacles_to_teb


def test_spawn_position(x_offset, y_offset, n_episodes=3):
    """
    Test a specific spawn position.

    Args:
        x_offset: X offset from goal (meters)
        y_offset: Y offset from goal (meters)
        n_episodes: Number of test episodes

    Returns:
        dict with success_rate, avg_depth, avg_steering_changes
    """
    # Load config
    with open("config_env.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # Modify spawn position
    cfg["scenarios"]["parallel"]["spawn_lane"]["x_min_offset"] = x_offset
    cfg["scenarios"]["parallel"]["spawn_lane"]["x_max_offset"] = x_offset
    cfg["scenarios"]["parallel"]["spawn_lane"]["y_min"] = cfg["scenarios"]["parallel"]["parking"]["bay"]["center_y"] + y_offset
    cfg["scenarios"]["parallel"]["spawn_lane"]["y_max"] = cfg["scenarios"]["parallel"]["parking"]["bay"]["center_y"] + y_offset

    # Merge scenario config
    scenario_cfg = copy.deepcopy(cfg["scenarios"]["parallel"])
    env_cfg = copy.deepcopy(cfg)
    for key, val in scenario_cfg.items():
        env_cfg[key] = val

    # Create env and MPC
    env = ParkingEnv(env_cfg)
    teb = TEBMPC(max_obstacles=25, env_cfg=env_cfg)

    successes = 0
    depths = []
    steering_changes_list = []

    for ep in range(n_episodes):
        obs = env.reset(randomize=True)

        step = 0
        done = False
        prev_steer = 0
        steer_changes = 0
        in_final_phase = False

        while not done and step < 150:  # Cap at 150 steps
            x, y, yaw, v = env.state
            state = VehicleState(x=x, y=y, yaw=yaw, v=v)
            goal = ParkingGoal(x=env.goal[0], y=env.goal[1], yaw=env.goal[2])
            obstacles = _env_obstacles_to_teb(env)

            try:
                sol = teb.solve(state, goal, obstacles, profile="parallel")
                action = np.array([float(sol.controls[0, 0]), float(sol.controls[0, 1])])

                # Track final phase steering
                depth = abs(y - env.goal[1])
                if depth < 0.10 and not in_final_phase:
                    in_final_phase = True

                if in_final_phase:
                    if abs(action[0]) > 0.01 and abs(prev_steer) > 0.01:
                        if np.sign(action[0]) != np.sign(prev_steer):
                            steer_changes += 1
                    prev_steer = action[0]

            except:
                action = np.zeros(2)

            obs, reward, done, info = env.step(action)
            step += 1

        term = info.get("termination", "unknown")
        if term == "success":
            successes += 1
            # Calculate final depth
            final_depth = abs(env.state[1] - env.goal[1])
            depths.append(final_depth * 100)  # cm
            steering_changes_list.append(steer_changes)

    success_rate = successes / n_episodes if n_episodes > 0 else 0
    avg_depth = np.mean(depths) if depths else np.inf
    avg_steering = np.mean(steering_changes_list) if steering_changes_list else np.inf

    return {
        "success_rate": success_rate,
        "avg_depth_cm": avg_depth,
        "avg_steering_changes": avg_steering,
        "n_success": successes,
        "n_total": n_episodes
    }


def main():
    """Sweep different spawn positions."""
    print("=" * 70)
    print("SPAWN POSITION SWEEP")
    print("=" * 70)

    # Test positions around the mean from successful episodes
    # Mean from analysis: x_offset=1.007, y_offset=0.834

    x_offsets = [0.80, 0.90, 1.00, 1.10, 1.20]  # From 0.8m to 1.2m ahead
    y_offsets = [0.75, 0.80, 0.85, 0.90]  # From 75cm to 90cm lateral

    print(f"\nTesting {len(x_offsets)} x {len(y_offsets)} = {len(x_offsets) * len(y_offsets)} positions")
    print("(3 episodes per position)\n")

    results = []

    for x_off in x_offsets:
        for y_off in y_offsets:
            print(f"Testing x_offset={x_off:.2f}, y_offset={y_off:.2f}... ", end="", flush=True)

            result = test_spawn_position(x_off, y_off, n_episodes=3)
            result["x_offset"] = x_off
            result["y_offset"] = y_off
            results.append(result)

            print(f"Success: {result['n_success']}/{result['n_total']}, "
                  f"Depth: {result['avg_depth_cm']:.2f}cm, "
                  f"Steering: {result['avg_steering_changes']:.1f}")

    # Find best positions
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    # Sort by success rate, then by depth
    results.sort(key=lambda r: (-r["success_rate"], r["avg_depth_cm"]))

    print("\nTop 10 Positions (by success rate, then depth):")
    print("-" * 70)
    print(f"{'X offset':<10} {'Y offset':<10} {'Success':<10} {'Depth (cm)':<12} {'Steering':<10}")
    print("-" * 70)

    for r in results[:10]:
        print(f"{r['x_offset']:<10.2f} {r['y_offset']:<10.2f} "
              f"{r['n_success']}/{r['n_total']:<8} "
              f"{r['avg_depth_cm']:<12.2f} {r['avg_steering_changes']:<10.1f}")

    # Identify 100% success positions
    perfect_positions = [r for r in results if r["success_rate"] == 1.0]

    if perfect_positions:
        print(f"\n{len(perfect_positions)} positions with 100% success rate:")
        print("-" * 70)
        print(f"{'X offset':<10} {'Y offset':<10} {'Depth (cm)':<12} {'Steering':<10}")
        print("-" * 70)
        for r in perfect_positions:
            print(f"{r['x_offset']:<10.2f} {r['y_offset']:<10.2f} "
                  f"{r['avg_depth_cm']:<12.2f} {r['avg_steering_changes']:<10.1f}")

        # Best among perfect (lowest depth, then steering)
        best = min(perfect_positions, key=lambda r: (r["avg_depth_cm"], r["avg_steering_changes"]))
        print(f"\nRECOMMENDED POSITION:")
        print(f"  x_offset: {best['x_offset']:.2f} m")
        print(f"  y_offset: {best['y_offset']:.2f} m")
        print(f"  Expected depth: {best['avg_depth_cm']:.2f} cm")
        print(f"  Expected steering changes: {best['avg_steering_changes']:.1f}")

    print("=" * 70)


if __name__ == "__main__":
    main()
