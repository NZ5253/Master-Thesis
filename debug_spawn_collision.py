#!/usr/bin/env python3
"""Debug script to check if fixed spawn position causes initial collision."""

import numpy as np
import yaml
from env.parking_env import ParkingEnv


def check_spawn_collision():
    """Test if the current fixed spawn configuration creates an initial collision."""

    # Load config
    with open("config_env.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # Get parallel scenario config
    scenario_cfg = cfg["scenarios"]["parallel"]

    # Merge scenario into env config
    env_cfg = cfg.copy()
    for key, val in scenario_cfg.items():
        env_cfg[key] = val

    # Create environment
    env = ParkingEnv(env_cfg)

    print("=" * 70)
    print("SPAWN COLLISION DEBUG")
    print("=" * 70)

    # Test 10 resets to see initial conditions
    for i in range(10):
        obs = env.reset(randomize=True)

        # Get spawn and goal info
        x, y, yaw, v = env.state
        gx, gy, gyaw = env.goal

        # Check collision
        collision = env.obstacles.check_collision(env.state, env.model)

        # Calculate initial distance to goal
        dist_to_goal = np.hypot(gx - x, gy - y)
        yaw_diff = abs(((gyaw - yaw + np.pi) % (2 * np.pi)) - np.pi)

        print(f"\nReset {i+1}:")
        print(f"  Spawn:     x={x:.3f}, y={y:.3f}, yaw={np.degrees(yaw):.1f}°")
        print(f"  Goal:      x={gx:.3f}, y={gy:.3f}, yaw={np.degrees(gyaw):.1f}°")
        print(f"  Distance:  {dist_to_goal:.3f} m")
        print(f"  Yaw diff:  {np.degrees(yaw_diff):.1f}°")
        print(f"  COLLISION: {collision}")

        if collision:
            print("  ⚠️  INITIAL COLLISION DETECTED!")
            print(f"  Obstacles ({len(env.obstacles.obstacles)}):")
            for j, o in enumerate(env.obstacles.obstacles):
                kind = o.get("kind", "obstacle")
                print(f"    {j}: {kind} at ({o['x']:.3f}, {o['y']:.3f}), "
                      f"w={o['w']:.3f}, h={o['h']:.3f}, theta={np.degrees(o.get('theta', 0)):.1f}°")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    check_spawn_collision()
