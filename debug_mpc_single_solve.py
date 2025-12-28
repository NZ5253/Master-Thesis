#!/usr/bin/env python3
"""Debug script to test a single MPC solve from fixed spawn position."""

import numpy as np
import yaml
from env.parking_env import ParkingEnv
from mpc.teb_mpc import TEBMPC, VehicleState, ParkingGoal, Obstacle
from mpc.generate_expert_data import _env_obstacles_to_teb


def test_single_solve():
    """Test a single MPC solve from the fixed spawn position."""

    # Load config
    with open("config_env.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # Get parallel scenario config
    scenario_cfg = cfg["scenarios"]["parallel"]

    # Merge scenario into env config
    env_cfg = cfg.copy()
    for key, val in scenario_cfg.items():
        env_cfg[key] = val

    # Create environment and MPC
    env = ParkingEnv(env_cfg)
    teb = TEBMPC(max_obstacles=25, env_cfg=env_cfg)

    print("=" * 70)
    print("SINGLE MPC SOLVE DEBUG")
    print("=" * 70)

    # Reset environment
    obs = env.reset(randomize=True)

    # Get state and goal
    x, y, yaw, v = env.state
    gx, gy, gyaw = env.goal

    print(f"\nInitial State:")
    print(f"  Position: ({x:.3f}, {y:.3f})")
    print(f"  Yaw:      {np.degrees(yaw):.1f}°")
    print(f"  Velocity: {v:.3f} m/s")

    print(f"\nGoal:")
    print(f"  Position: ({gx:.3f}, {gy:.3f})")
    print(f"  Yaw:      {np.degrees(gyaw):.1f}°")

    print(f"\nDistance to goal: {np.hypot(gx-x, gy-y):.3f} m")

    print(f"\nObstacles: {len(env.obstacles.obstacles)}")
    for i, o in enumerate(env.obstacles.obstacles):
        kind = o.get("kind", "obstacle")
        print(f"  {i}: {kind} at ({o['x']:.3f}, {o['y']:.3f}), "
              f"w={o['w']:.3f}, h={o['h']:.3f}")

    # Convert to MPC format
    state = VehicleState(x=x, y=y, yaw=yaw, v=v)
    goal = ParkingGoal(x=gx, y=gy, yaw=gyaw)
    obstacles = _env_obstacles_to_teb(env)

    print(f"\nMPC Obstacles (converted): {len(obstacles)}")

    print("\nAttempting MPC solve...")
    print("(This should complete in <1 second if working correctly)")
    print("-" * 70)

    import time
    start_time = time.time()

    try:
        sol = teb.solve(state, goal, obstacles, profile="parallel")
        elapsed = time.time() - start_time

        print(f"\n✓ SOLVE SUCCEEDED in {elapsed:.2f} seconds")
        print(f"  Controls: steer={sol.controls[0,0]:.3f}, accel={sol.controls[0,1]:.3f}")
        print(f"  Horizon: {sol.controls.shape[0]} steps")
        if sol.phase is not None:
            print(f"  Phase: {sol.phase.value}")

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n✗ SOLVE FAILED after {elapsed:.2f} seconds")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()

    # Test a few more steps to see if subsequent solves are faster
    print("\n" + "-" * 70)
    print("Testing subsequent solves...")
    print("-" * 70)

    for step in range(1, 6):
        # Apply the action
        action = np.array([float(sol.controls[0, 0]), float(sol.controls[0, 1])])
        obs, reward, done, info = env.step(action)

        if done:
            print(f"\nStep {step}: Episode terminated ({info.get('termination', 'unknown')})")
            break

        # Solve again
        x, y, yaw, v = env.state
        state = VehicleState(x=x, y=y, yaw=yaw, v=v)
        obstacles = _env_obstacles_to_teb(env)

        start_time = time.time()
        sol = teb.solve(state, goal, obstacles, profile="parallel")
        elapsed = time.time() - start_time

        print(f"Step {step}: solve in {elapsed:.3f}s, steer={sol.controls[0,0]:.3f}, accel={sol.controls[0,1]:.3f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    test_single_solve()
