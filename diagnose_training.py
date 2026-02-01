"""
Diagnostic script to understand why agent isn't discovering success.

Runs the trained agent and checks:
1. How close does it get to the goal?
2. What's preventing success?
3. Are rewards actually working as expected?
"""

import numpy as np
from rl.curriculum_env import make_curriculum_env
from ray.rllib.algorithms.ppo import PPO

def diagnose_checkpoint(checkpoint_path, num_episodes=10):
    """Diagnose what's preventing success."""

    # Load checkpoint
    print("Loading checkpoint...")
    algo = PPO.from_checkpoint(checkpoint_path)

    # Create environment
    env = make_curriculum_env(scenario="parallel", phase_name="phase1_foundation")

    print("\n" + "=" * 80)
    print("DIAGNOSTIC ANALYSIS")
    print("=" * 80)

    closest_positions = []
    closest_velocities = []
    closest_yaws = []
    final_rewards = []

    for ep in range(num_episodes):
        obs, info = env.reset(seed=42 + ep)
        done = False
        episode_reward = 0.0
        step = 0

        goal = info["goal"]
        closest_dist = float('inf')
        closest_state = None

        while not done:
            action = algo.compute_single_action(obs, explore=False, policy_id="default_policy")
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            step += 1

            # Track closest approach to goal
            state = env.env.state
            x, y, yaw, v = state[0], state[1], state[2], state[3]
            dist_to_goal = np.sqrt((x - goal[0])**2 + (y - goal[1])**2)

            if dist_to_goal < closest_dist:
                closest_dist = dist_to_goal
                closest_state = state.copy()

        # Analyze closest state
        if closest_state is not None:
            x, y, yaw, v = closest_state[0], closest_state[1], closest_state[2], closest_state[3]
            pos_err = np.sqrt((x - goal[0])**2 + (y - goal[1])**2)
            yaw_err = abs(yaw - goal[2])

            closest_positions.append(pos_err)
            closest_velocities.append(abs(v))
            closest_yaws.append(yaw_err)

        final_rewards.append(episode_reward)

        # Episode summary
        success = info.get("success", False)
        term = info.get("termination", "?")
        print(f"Ep {ep+1}: {'SUCCESS' if success else term:12s} | "
              f"Reward: {episode_reward:6.2f} | Steps: {step:3d} | "
              f"Closest: {closest_dist:.3f}m")

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY - Why No Success?")
    print("=" * 80)
    print(f"Success tolerances:")
    print(f"  Position: 0.15m (15cm)")
    print(f"  Velocity: 0.15 m/s")
    print(f"  Yaw: 0.15 rad (~8.5°)")
    print()
    print(f"Agent's closest approaches:")
    print(f"  Position error: {np.mean(closest_positions):.3f}m ± {np.std(closest_positions):.3f}m")
    print(f"  Velocity: {np.mean(closest_velocities):.3f} m/s ± {np.std(closest_velocities):.3f} m/s")
    print(f"  Yaw error: {np.mean(closest_yaws):.3f} rad ± {np.std(closest_yaws):.3f} rad")
    print()

    # Diagnosis
    print("DIAGNOSIS:")
    avg_pos_err = np.mean(closest_positions)
    avg_vel = np.mean(closest_velocities)
    avg_yaw_err = np.mean(closest_yaws)

    if avg_pos_err > 0.50:
        print(f"  ❌ MAJOR ISSUE: Agent not getting close to goal (avg {avg_pos_err:.2f}m away)")
        print(f"     → Agent hasn't learned to approach the parking bay")
        print(f"     → Need stronger reward gradient or easier initial position")
    elif avg_pos_err > 0.15:
        print(f"  ⚠️  Position: Getting close ({avg_pos_err:.2f}m) but not close enough")
        print(f"     → Agent knows where to go but can't nail the final position")
    else:
        print(f"  ✓ Position: Good ({avg_pos_err:.2f}m)")

    if avg_vel > 0.30:
        print(f"  ❌ Velocity: Moving too fast ({avg_vel:.2f} m/s)")
        print(f"     → Agent hasn't learned to stop")
    elif avg_vel > 0.15:
        print(f"  ⚠️  Velocity: Close but too fast ({avg_vel:.2f} m/s)")
    else:
        print(f"  ✓ Velocity: Good ({avg_vel:.2f} m/s)")

    if avg_yaw_err > 0.30:
        print(f"  ❌ Heading: Way off ({avg_yaw_err:.2f} rad = {np.degrees(avg_yaw_err):.1f}°)")
        print(f"     → Agent hasn't learned correct orientation")
    elif avg_yaw_err > 0.15:
        print(f"  ⚠️  Heading: Close but not aligned ({avg_yaw_err:.2f} rad = {np.degrees(avg_yaw_err):.1f}°)")
    else:
        print(f"  ✓ Heading: Good ({avg_yaw_err:.2f} rad)")

    print(f"\nAverage reward per episode: {np.mean(final_rewards):.2f}")
    print("=" * 80)


if __name__ == "__main__":
    import sys
    import ray
    from ray.tune.registry import register_env
    from rl.curriculum_env import create_env_for_rllib

    ray.init(ignore_reinit_error=True, log_to_driver=False, logging_level="ERROR")
    register_env("curriculum_parking_env", create_env_for_rllib)

    checkpoint = "checkpoints/curriculum/curriculum_20260121_123203/phase1_foundation/best_checkpoint"

    if len(sys.argv) > 1:
        checkpoint = sys.argv[1]

    diagnose_checkpoint(checkpoint, num_episodes=20)

    ray.shutdown()
