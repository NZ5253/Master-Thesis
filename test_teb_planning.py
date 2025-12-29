"""
Test script for TEB planning mode.

This tests Phase 2, Milestone 2.2: TEB Planner Implementation
"""

import numpy as np
from mpc.teb_mpc import TEBMPC, VehicleState, ParkingGoal, Obstacle
from env.parking_env import ParkingEnv
import yaml


def test_teb_planning():
    """Test TEB planning mode creates reference trajectory."""
    print("="*70)
    print("Testing TEB Planning Mode (Phase 2, Milestone 2.2)")
    print("="*70)

    # Load environment config
    with open("config_env.yaml", "r") as f:
        cfg_full = yaml.safe_load(f)

    # Merge parallel scenario into config
    import copy
    scenario = "parallel"
    env_cfg = copy.deepcopy(cfg_full)
    scenario_cfg = copy.deepcopy(cfg_full["scenarios"][scenario])
    for key, val in scenario_cfg.items():
        env_cfg[key] = val

    # Create environment to get obstacles and goal
    env = ParkingEnv(config=env_cfg)
    env.reset()

    # Create TEBMPC solver
    mpc = TEBMPC(
        config_path="mpc/config_mpc.yaml",
        env_cfg=env_cfg,
        dt=env_cfg["dt"]
    )

    # Get initial state and goal
    initial_state = VehicleState(
        x=env.state[0],
        y=env.state[1],
        yaw=env.state[2],
        v=env.state[3]
    )

    goal = ParkingGoal(
        x=env.goal[0],
        y=env.goal[1],
        yaw=env.goal[2]
    )

    # Convert obstacles
    obstacles = []
    for o in env.obstacles.obstacles:
        cx, cy, w, h = o["x"], o["y"], o["w"], o["h"]
        obstacles.append(Obstacle(cx=cx, cy=cy, hx=w/2, hy=h/2))

    print(f"\n[SETUP] Initial State: x={initial_state.x:.2f}, y={initial_state.y:.2f}, "
          f"yaw={initial_state.yaw:.2f}, v={initial_state.v:.2f}")
    print(f"[SETUP] Goal: x={goal.x:.2f}, y={goal.y:.2f}, yaw={goal.yaw:.2f}")
    print(f"[SETUP] Obstacles: {len(obstacles)}")

    # ============================================================================
    # TEST: Plan trajectory using TEB
    # ============================================================================
    print("\n" + "="*70)
    print("TEST: TEB Planning Mode")
    print("="*70)

    ref_traj = mpc.plan_trajectory(
        state=initial_state,
        goal=goal,
        obstacles=obstacles,
        profile="parallel"
    )

    # ============================================================================
    # VERIFY: Reference trajectory
    # ============================================================================
    print("\n" + "="*70)
    print("VERIFICATION")
    print("="*70)

    assert ref_traj.success, "❌ Planning failed!"
    print("✓ Planning succeeded")

    # 1.3m distance, expect 25-50 steps @ 0.1-0.25s avg dt
    assert 20 <= ref_traj.n_steps <= 60, f"❌ Steps out of range: {ref_traj.n_steps} not in [20, 60]"
    print(f"✓ Reasonable number of steps: {ref_traj.n_steps}")

    assert ref_traj.total_time > 0, "❌ Zero duration!"
    print(f"✓ Positive duration: {ref_traj.total_time:.2f}s")

    # Check dt variability (TEB should create variable dt)
    dt_std = np.std(ref_traj.dt_array)
    dt_range = np.max(ref_traj.dt_array) - np.min(ref_traj.dt_array)
    print(f"✓ dt variability: std={dt_std:.4f}, range={dt_range:.4f}")

    if dt_range > 0.05:
        print("  → TEB created variable time intervals (good for commitment)")
    else:
        print("  ⚠ dt is nearly constant (TEB may not be optimizing time)")

    # Check maneuvers
    maneuvers = ref_traj.analyze_maneuvers()
    print(f"✓ Identified {len(maneuvers)} committed maneuvers")

    # Expected: 3-5 committed maneuvers for parallel parking
    if 3 <= len(maneuvers) <= 8:
        print("  → Maneuver count is reasonable for parking")
    else:
        print(f"  ⚠ Unusual maneuver count: {len(maneuvers)}")

    # ============================================================================
    # SUMMARY
    # ============================================================================
    print("\n" + "="*70)
    print("REFERENCE TRAJECTORY SUMMARY")
    print("="*70)
    print(ref_traj.summary())

    # ============================================================================
    # SUCCESS CRITERIA (Phase 2, Milestone 2.2)
    # ============================================================================
    print("\n" + "="*70)
    print("SUCCESS CRITERIA CHECK")
    print("="*70)

    checks = {
        "Planning succeeds": ref_traj.success,
        "Steps >= 40": ref_traj.n_steps >= 40,
        "Duration > 0": ref_traj.total_time > 0,
        "Variable dt (range > 0.02)": dt_range > 0.02,
        "3-8 maneuvers": 3 <= len(maneuvers) <= 8
    }

    all_passed = True
    for check, passed in checks.items():
        status = "✓" if passed else "❌"
        print(f"  {status} {check}")
        if not passed:
            all_passed = False

    print("\n" + "="*70)
    if all_passed:
        print("✅ MILESTONE 2.2: TEB PLANNER - PASSED")
    else:
        print("⚠ MILESTONE 2.2: TEB PLANNER - PARTIAL (some checks failed)")
    print("="*70)

    return ref_traj


if __name__ == "__main__":
    test_teb_planning()
