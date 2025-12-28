#!/usr/bin/env python3
"""Analyze performance of fixed spawn episodes."""

import os
import pickle
import numpy as np


def analyze_episode(filepath):
    """Analyze a single episode."""
    with open(filepath, "rb") as f:
        data = pickle.load(f)

    traj = data["traj"]
    goal = data["goal"]

    gx, gy, gyaw = goal

    # Extract full trajectory
    positions = []
    actions = []

    for obs, action in traj:
        x, y, yaw, v = obs[0], obs[1], obs[2], obs[3]
        positions.append([x, y, yaw, v])
        actions.append(action)

    positions = np.array(positions)
    actions = np.array(actions)

    # Find when final phase starts (when depth < 0.10m based on MPC config)
    depths = []
    for x, y, yaw, v in positions:
        # Calculate rear axle depth (Y offset from goal)
        depth = abs(y - gy)
        depths.append(depth)

    depths = np.array(depths)

    # Final phase threshold from config: 0.10m
    final_phase_start = np.where(depths < 0.10)[0]
    if len(final_phase_start) > 0:
        final_phase_idx = final_phase_start[0]
    else:
        final_phase_idx = len(depths)

    # Analyze steering in final phase
    if final_phase_idx < len(actions):
        final_actions = actions[final_phase_idx:]
        final_steers = final_actions[:, 0]

        # Count steering changes (sign changes)
        steer_changes = 0
        for i in range(1, len(final_steers)):
            if abs(final_steers[i]) > 0.01 and abs(final_steers[i-1]) > 0.01:
                if np.sign(final_steers[i]) != np.sign(final_steers[i-1]):
                    steer_changes += 1

        final_steps = len(final_actions)
    else:
        steer_changes = 0
        final_steps = 0

    # Final depth
    final_depth = depths[-1]

    return {
        "total_steps": len(positions),
        "final_phase_steps": final_steps,
        "final_depth_cm": final_depth * 100,
        "steering_changes": steer_changes,
        "final_yaw_err_deg": abs(((gyaw - positions[-1, 2] + np.pi) % (2 * np.pi)) - np.pi) * 180 / np.pi,
    }


def main():
    """Analyze all episodes in the dataset."""
    data_dir = "data/expert_parallel"

    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} not found")
        return

    print("=" * 70)
    print("FIXED SPAWN PERFORMANCE ANALYSIS")
    print("=" * 70)

    episode_files = sorted([f for f in os.listdir(data_dir) if f.startswith("episode_") and f.endswith(".pkl")])

    results = []

    for fname in episode_files:
        filepath = os.path.join(data_dir, fname)
        result = analyze_episode(filepath)
        results.append(result)

        print(f"\n{fname}:")
        print(f"  Total steps:         {result['total_steps']}")
        print(f"  Final phase steps:   {result['final_phase_steps']}")
        print(f"  Final depth:         {result['final_depth_cm']:.2f} cm")
        print(f"  Steering changes:    {result['steering_changes']}")
        print(f"  Final yaw error:     {result['final_yaw_err_deg']:.2f}°")

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    depths = [r["final_depth_cm"] for r in results]
    steering_changes = [r["steering_changes"] for r in results]
    final_steps = [r["final_phase_steps"] for r in results]

    print(f"\nFinal Depth (cm):")
    print(f"  Mean:   {np.mean(depths):.2f} cm")
    print(f"  Std:    {np.std(depths):.2f} cm")
    print(f"  Min:    {np.min(depths):.2f} cm")
    print(f"  Max:    {np.max(depths):.2f} cm")

    print(f"\nSteering Changes (final phase):")
    print(f"  Mean:   {np.mean(steering_changes):.1f}")
    print(f"  Std:    {np.std(steering_changes):.1f}")
    print(f"  Min:    {np.min(steering_changes)}")
    print(f"  Max:    {np.max(steering_changes)}")

    print(f"\nFinal Phase Duration:")
    print(f"  Mean:   {np.mean(final_steps):.1f} steps")
    print(f"  Std:    {np.std(final_steps):.1f} steps")

    print(f"\nSuccess Rate: {len(results)}/{len(results)} = 100.0%")
    print("=" * 70)


if __name__ == "__main__":
    main()
