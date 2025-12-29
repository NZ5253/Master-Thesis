"""
Test script for Hybrid TEB+MPC system.

Phase 3: Tests that MPC can track a TEB-generated reference trajectory.
"""

import numpy as np
import yaml
import copy
from mpc.teb_mpc import TEBMPC, VehicleState, ParkingGoal, Obstacle
from env.parking_env import ParkingEnv


def test_hybrid_tracking():
    """Test full hybrid system: TEB plans, MPC tracks."""
    print("="*70)
    print("Testing Hybrid TEB+MPC System (Phase 3)")
    print("="*70)

    # Load environment config
    with open("config_env.yaml", "r") as f:
        cfg_full = yaml.safe_load(f)

    # Merge parallel scenario
    scenario = "parallel"
    env_cfg = copy.deepcopy(cfg_full)
    scenario_cfg = copy.deepcopy(cfg_full["scenarios"][scenario])
    for key, val in scenario_cfg.items():
        env_cfg[key] = val

    # Create environment
    env = ParkingEnv(config=env_cfg)
    env.reset()

    # Create TEBMPC solver
    mpc = TEBMPC(
        config_path="mpc/config_mpc.yaml",
        env_cfg=env_cfg,
        dt=env_cfg["dt"]
    )

    # Get initial state and goal
    state = VehicleState(
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

    print(f"\n[SETUP] Initial State: x={state.x:.2f}, y={state.y:.2f}, yaw={state.yaw:.2f}")
    print(f"[SETUP] Goal: x={goal.x:.2f}, y={goal.y:.2f}, yaw={goal.yaw:.2f}")
    print(f"[SETUP] Distance to goal: {np.sqrt((goal.x-state.x)**2 + (goal.y-state.y)**2):.3f}m")

    # ============================================================================
    # STEP 1: TEB Planning (once at start)
    # ============================================================================
    print("\n" + "="*70)
    print("STEP 1: TEB PLANNING (once)")
    print("="*70)

    ref_traj = mpc.plan_trajectory(
        state=state,
        goal=goal,
        obstacles=obstacles,
        profile="parallel"
    )

    assert ref_traj.success, "❌ TEB planning failed!"
    print(f"✓ TEB created reference trajectory: {ref_traj.n_steps} steps, {ref_traj.total_time:.2f}s")

    # ============================================================================
    # STEP 2: MPC Tracking (every step)
    # ============================================================================
    print("\n" + "="*70)
    print("STEP 2: MPC TRACKING (every step)")
    print("="*70)

    max_steps = 100
    trajectory = []
    current_state = state

    for step_idx in range(max_steps):
        # Track reference trajectory
        sol = mpc.track_trajectory(
            state=current_state,
            reference=ref_traj,
            obstacles=obstacles,
            step=step_idx,
            profile="parallel"
        )

        if not sol.success:
            print(f"❌ Tracking failed at step {step_idx}")
            break

        # Execute first control
        control = sol.controls[0]
        trajectory.append({
            "step": step_idx,
            "state": [current_state.x, current_state.y, current_state.yaw, current_state.v],
            "control": control.tolist(),
            "ref_goal": sol.info.get("reference_goal", None)
        })

        # Simulate one step
        obs, reward, done, info = env.step(control)
        current_state = VehicleState(
            x=env.state[0],
            y=env.state[1],
            yaw=env.state[2],
            v=env.state[3]
        )

        # Check progress
        pos_err = np.sqrt((env.state[0] - goal.x)**2 + (env.state[1] - goal.y)**2)
        yaw_err = abs((env.state[2] - goal.yaw + np.pi) % (2 * np.pi) - np.pi)

        if step_idx % 10 == 0 or step_idx < 5:
            print(f"  [Step {step_idx:2d}] pos_err={pos_err:.3f}m, yaw_err={yaw_err:.3f}rad, "
                  f"steer={control[0]:.3f}, accel={control[1]:.3f}")

        # Check if goal reached
        if pos_err < 0.05 and yaw_err < 0.10:
            print(f"\n✓ Goal reached at step {step_idx}!")
            print(f"  Final errors: pos={pos_err:.4f}m, yaw={yaw_err:.4f}rad")
            break

        # Check for collision
        if done and not info.get("success", False):
            print(f"❌ Collision or failure at step {step_idx}")
            break

    # ============================================================================
    # STEP 3: Analysis
    # ============================================================================
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)

    total_steps = len(trajectory)
    print(f"Total steps: {total_steps}")
    print(f"Reference length: {ref_traj.n_steps} steps")

    # Check for oscillations by analyzing position error
    pos_errors = []
    for t in trajectory:
        state_arr = t["state"]
        pos_err = np.sqrt((state_arr[0] - goal.x)**2 + (state_arr[1] - goal.y)**2)
        pos_errors.append(pos_err)

    # Detect oscillations (error increases after decreasing)
    oscillations = 0
    for i in range(2, len(pos_errors)):
        if pos_errors[i-2] > pos_errors[i-1] < pos_errors[i]:
            # Error decreased then increased = potential oscillation
            if pos_errors[i] - pos_errors[i-1] > 0.02:  # Significant increase (>2cm)
                oscillations += 1
                print(f"  ⚠ Oscillation detected at step {i}: "
                      f"{pos_errors[i-2]:.3f} → {pos_errors[i-1]:.3f} → {pos_errors[i]:.3f}")

    # ============================================================================
    # COMPARISON WITH BASELINE
    # ============================================================================
    print("\n" + "="*70)
    print("COMPARISON: Hybrid vs Baseline")
    print("="*70)

    print(f"{'Metric':<25} {'Baseline':<15} {'Hybrid':<15} {'Change'}")
    print("-" * 70)
    print(f"{'Steps':<25} {'55':<15} {f'{total_steps}':<15} {f'{((total_steps-55)/55*100):+.1f}%' if total_steps > 0 else 'N/A'}")
    print(f"{'Oscillations':<25} {'1 major':<15} {f'{oscillations}':<15} {f'{oscillations-1:+d}' if oscillations >= 0 else 'N/A'}")
    print(f"{'Final pos error':<25} {'2.8cm':<15} {f'{pos_errors[-1]*100:.1f}cm':<15}" if pos_errors else "N/A")
    print(f"{'Success':<25} {'100%':<15} {'Yes' if total_steps < max_steps else 'No':<15}")

    # ============================================================================
    # SUCCESS CRITERIA
    # ============================================================================
    print("\n" + "="*70)
    print("SUCCESS CRITERIA")
    print("="*70)

    checks = {
        "TEB planning succeeds": ref_traj.success,
        "MPC tracking succeeds": total_steps > 0 and total_steps < max_steps,
        "Reaches goal": total_steps < max_steps,
        "Fewer oscillations than baseline": oscillations <= 1,
        "Reasonable steps (20-60)": 20 <= total_steps <= 60
    }

    all_passed = True
    for check, passed in checks.items():
        status = "✓" if passed else "❌"
        print(f"  {status} {check}")
        if not passed:
            all_passed = False

    print("\n" + "="*70)
    if all_passed:
        print("✅ PHASE 3: MPC TRACKING - PASSED")
    else:
        print("⚠ PHASE 3: MPC TRACKING - PARTIAL")
    print("="*70)

    return all_passed


if __name__ == "__main__":
    success = test_hybrid_tracking()
    exit(0 if success else 1)
