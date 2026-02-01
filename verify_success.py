"""
Verify that reported successes are real by checking final position errors.

This script:
1. Loads a checkpoint
2. Runs episodes
3. For each episode, shows:
   - Final position error (cm)
   - Final velocity (cm/s)
   - Final heading error (degrees)
   - Whether it meets success criteria
4. Compares reported success vs actual measurements

Usage:
    python verify_success.py checkpoints/curriculum/curriculum_20260121_152111/phase1_foundation/best_checkpoint
"""

import sys
import numpy as np
import ray
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env

from rl.curriculum_env import create_env_for_rllib, make_curriculum_env


def verify_checkpoint(checkpoint_path, num_episodes=20):
    """Run episodes and verify success criteria are actually met."""

    # Initialize Ray
    ray.init(ignore_reinit_error=True, log_to_driver=False, logging_level="ERROR")
    register_env("curriculum_parking_env", create_env_for_rllib)

    print("=" * 80)
    print("SUCCESS VERIFICATION")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Episodes: {num_episodes}")
    print()

    # Load checkpoint
    print("Loading checkpoint...")
    algo = PPO.from_checkpoint(checkpoint_path)
    print("Checkpoint loaded!\n")

    # Create environment
    env = make_curriculum_env(scenario="parallel", phase_name="phase1_foundation")

    # Get success tolerances from config
    env.reset()
    # Access parking config from the wrapped env
    parking_cfg = env.env.parking_cfg
    success_cfg = parking_cfg.get("success", {})
    along_tol = float(success_cfg.get("parallel_along_tol", 0.15))  # meters (X direction)
    lateral_tol = float(success_cfg.get("parallel_lateral_tol", 0.15))  # meters (Y direction)
    vel_tol = float(success_cfg.get("v_tol", 0.15))  # m/s
    yaw_tol = float(success_cfg.get("yaw_tol", 0.15))  # radians

    # Vehicle geometry for car center calculation
    vehicle_params = env.env.vehicle_params
    L = float(vehicle_params.get("length", 0.36))
    dist_to_center = L / 2.0 - 0.05

    print("SUCCESS CRITERIA:")
    print(f"  Along tolerance (X):    {along_tol*100:.0f}cm")
    print(f"  Lateral tolerance (Y):  {lateral_tol*100:.0f}cm")
    print(f"  Velocity tolerance:     {vel_tol*100:.0f}cm/s")
    print(f"  Heading tolerance:      {np.degrees(yaw_tol):.1f}°")
    print("=" * 80)
    print()

    # Track statistics
    reported_successes = 0
    actual_successes = 0
    false_positives = 0
    false_negatives = 0

    success_along_errors = []
    success_lateral_errors = []
    success_velocity_errors = []
    success_heading_errors = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        step = 0

        goal = info["goal"]
        bay_center = env.env.bay_center  # Get actual bay center for this episode

        while not done:
            action = algo.compute_single_action(obs, explore=False, policy_id="default_policy")
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

        # Get final state (rear axle position)
        state = env.env.state
        x_ra, y_ra, yaw, v = state[0], state[1], state[2], state[3]
        gx, gy, gyaw = goal[0], goal[1], goal[2]

        # Calculate CAR CENTER position (what success checks)
        cx = x_ra + dist_to_center * np.cos(yaw)
        cy = y_ra + dist_to_center * np.sin(yaw)

        # Calculate errors using car center vs bay center
        slot_cx, slot_cy = bay_center[0], bay_center[1]
        along_error = abs(cx - slot_cx)  # X direction
        lateral_error = abs(cy - slot_cy)  # Y direction

        vel_error = abs(v)
        yaw_error = abs(((gyaw - yaw + np.pi) % (2 * np.pi)) - np.pi)

        # Check if it actually meets criteria
        meets_along = along_error < along_tol
        meets_lateral = lateral_error < lateral_tol
        meets_velocity = vel_error < vel_tol
        meets_heading = yaw_error < yaw_tol

        actual_success = meets_along and meets_lateral and meets_velocity and meets_heading
        reported_success = info.get("success", False)

        # Track stats
        if reported_success:
            reported_successes += 1
        if actual_success:
            actual_successes += 1
            success_along_errors.append(along_error * 100)  # cm
            success_lateral_errors.append(lateral_error * 100)  # cm
            success_velocity_errors.append(vel_error * 100)  # cm/s
            success_heading_errors.append(np.degrees(yaw_error))  # degrees

        if reported_success and not actual_success:
            false_positives += 1
        if not reported_success and actual_success:
            false_negatives += 1

        # Print episode details
        status_icon = "✓" if actual_success else "✗"
        mismatch = ""
        if reported_success != actual_success:
            mismatch = " ⚠️ MISMATCH!"

        print(f"Ep {ep+1:2d}: {status_icon} | "
              f"Along: {along_error*100:5.1f}cm {'✓' if meets_along else '✗'} | "
              f"Lat: {lateral_error*100:5.1f}cm {'✓' if meets_lateral else '✗'} | "
              f"Vel: {vel_error*100:5.1f}cm/s {'✓' if meets_velocity else '✗'} | "
              f"Yaw: {np.degrees(yaw_error):5.1f}° {'✓' if meets_heading else '✗'} | "
              f"Steps: {step:3d}{mismatch}")

    # Summary statistics
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    print(f"Total episodes: {num_episodes}")
    print()
    print(f"Reported successes: {reported_successes} ({reported_successes/num_episodes*100:.1f}%)")
    print(f"Actual successes:   {actual_successes} ({actual_successes/num_episodes*100:.1f}%)")
    print()

    if false_positives > 0:
        print(f"⚠️  FALSE POSITIVES: {false_positives} episodes marked success but didn't meet criteria!")
    else:
        print("✓ No false positives (all reported successes are real)")

    if false_negatives > 0:
        print(f"⚠️  FALSE NEGATIVES: {false_negatives} episodes met criteria but not marked success!")
    else:
        print("✓ No false negatives (all actual successes were detected)")

    print()

    if actual_successes > 0:
        print("SUCCESSFUL EPISODES - FINAL STATE ANALYSIS:")
        print(f"  Along errors (X): {np.mean(success_along_errors):.1f}cm ± {np.std(success_along_errors):.1f}cm")
        print(f"                    (Range: {np.min(success_along_errors):.1f}cm - {np.max(success_along_errors):.1f}cm)")
        print(f"  Lateral errors (Y): {np.mean(success_lateral_errors):.1f}cm ± {np.std(success_lateral_errors):.1f}cm")
        print(f"                    (Range: {np.min(success_lateral_errors):.1f}cm - {np.max(success_lateral_errors):.1f}cm)")
        print(f"  Velocity errors:  {np.mean(success_velocity_errors):.1f}cm/s ± {np.std(success_velocity_errors):.1f}cm/s")
        print(f"                    (Range: {np.min(success_velocity_errors):.1f}cm/s - {np.max(success_velocity_errors):.1f}cm/s)")
        print(f"  Heading errors:   {np.mean(success_heading_errors):.1f}° ± {np.std(success_heading_errors):.1f}°")
        print(f"                    (Range: {np.min(success_heading_errors):.1f}° - {np.max(success_heading_errors):.1f}°)")
        print()
        print("✓ All values are WELL within tolerance! These are REAL parking successes.")
    else:
        print("No successful episodes to analyze.")

    print("=" * 80)

    # Cleanup
    ray.shutdown()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_success.py <checkpoint_path> [num_episodes]")
        print()
        print("Example:")
        print("  python verify_success.py checkpoints/curriculum/curriculum_20260121_152111/phase1_foundation/best_checkpoint 50")
        sys.exit(1)

    checkpoint = sys.argv[1]
    num_eps = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    verify_checkpoint(checkpoint, num_episodes=num_eps)
