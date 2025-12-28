#!/usr/bin/env python3
"""Analyze spawn positions from successful episodes."""

import os
import pickle
import numpy as np


def analyze_successful_episodes():
    """Load successful episodes and analyze their spawn positions."""

    data_dir = "data/expert_parallel_collision_improved"

    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} not found")
        return

    print("=" * 70)
    print("SUCCESSFUL EPISODE SPAWN ANALYSIS")
    print("=" * 70)

    spawn_positions = []
    goal_positions = []

    episode_files = sorted([f for f in os.listdir(data_dir) if f.startswith("episode_") and f.endswith(".pkl")])

    for fname in episode_files[:20]:  # Analyze first 20 successes
        filepath = os.path.join(data_dir, fname)

        with open(filepath, "rb") as f:
            data = pickle.load(f)

        traj = data["traj"]
        goal = data["goal"]

        # First observation is the spawn state
        obs0, action0 = traj[0]
        # obs = [x, y, yaw, v, dx, dy, dtheta, dist_front, dist_left, dist_right]
        x0, y0, yaw0, v0 = obs0[0], obs0[1], obs0[2], obs0[3]

        gx, gy, gyaw = goal

        spawn_positions.append([x0, y0, yaw0])
        goal_positions.append([gx, gy, gyaw])

        # Calculate initial offset from goal
        dx = x0 - gx
        dy = y0 - gy
        dyaw = abs(((gyaw - yaw0 + np.pi) % (2 * np.pi)) - np.pi)

        print(f"\n{fname}:")
        print(f"  Spawn: x={x0:.3f}, y={y0:.3f}, yaw={np.degrees(yaw0):.1f}°")
        print(f"  Goal:  x={gx:.3f}, y={gy:.3f}, yaw={np.degrees(gyaw):.1f}°")
        print(f"  Offset: Δx={dx:.3f}, Δy={dy:.3f}, Δyaw={np.degrees(dyaw):.1f}°")
        print(f"  Distance: {np.hypot(dx, dy):.3f} m")

    # Statistical summary
    spawn_positions = np.array(spawn_positions)
    goal_positions = np.array(goal_positions)

    offsets_x = spawn_positions[:, 0] - goal_positions[:, 0]
    offsets_y = spawn_positions[:, 1] - goal_positions[:, 1]

    print("\n" + "=" * 70)
    print("STATISTICAL SUMMARY (Spawn relative to Goal)")
    print("=" * 70)
    print(f"\nX Offset (spawn - goal):")
    print(f"  Mean:   {np.mean(offsets_x):.3f} m")
    print(f"  Std:    {np.std(offsets_x):.3f} m")
    print(f"  Min:    {np.min(offsets_x):.3f} m")
    print(f"  Max:    {np.max(offsets_x):.3f} m")

    print(f"\nY Offset (spawn - goal):")
    print(f"  Mean:   {np.mean(offsets_y):.3f} m")
    print(f"  Std:    {np.std(offsets_y):.3f} m")
    print(f"  Min:    {np.min(offsets_y):.3f} m")
    print(f"  Max:    {np.max(offsets_y):.3f} m")

    # Spawn yaw (absolute)
    print(f"\nSpawn Yaw:")
    print(f"  Mean:   {np.degrees(np.mean(spawn_positions[:, 2])):.1f}°")
    print(f"  Std:    {np.degrees(np.std(spawn_positions[:, 2])):.1f}°")

    # Goal yaw (absolute)
    print(f"\nGoal Yaw:")
    print(f"  Mean:   {np.degrees(np.mean(goal_positions[:, 2])):.1f}°")

    print("\n" + "=" * 70)
    print("RECOMMENDED FIXED SPAWN OFFSET:")
    print("=" * 70)
    print(f"  x_offset: {np.mean(offsets_x):.3f} m (use as x_min_offset/x_max_offset)")
    print(f"  y: goal_y + {np.mean(offsets_y):.3f} m")
    print(f"  yaw: {np.degrees(np.mean(spawn_positions[:, 2])):.1f}°")
    print("=" * 70)


if __name__ == "__main__":
    analyze_successful_episodes()
