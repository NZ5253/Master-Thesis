"""
Quick checkpoint testing without visualization.

Usage:
    python -m rl.test_checkpoint --checkpoint <path> --num-episodes 20

Example:
    python -m rl.test_checkpoint \
        --checkpoint checkpoints/curriculum/curriculum_20260121_114300/phase1_foundation/best_checkpoint \
        --num-episodes 50
"""

import argparse
import os
from pathlib import Path

import ray
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env
import numpy as np

from rl.curriculum_env import create_env_for_rllib, make_curriculum_env


def test_checkpoint(args):
    """
    Load a checkpoint and test without visualization.

    Args:
        args: Command line arguments
    """
    # Initialize Ray (minimal logging)
    ray.init(ignore_reinit_error=True, log_to_driver=False, logging_level="ERROR")

    # Register environment
    register_env("curriculum_parking_env", create_env_for_rllib)

    print("=" * 80)
    print("CHECKPOINT TESTING")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Episodes: {args.num_episodes}")
    print(f"Deterministic: {args.deterministic}")
    print("=" * 80)

    # Load algorithm
    print("\nLoading checkpoint...")
    algo = PPO.from_checkpoint(args.checkpoint)
    print("Checkpoint loaded!")

    # Create environment
    if args.scenario == "parallel":
        from rl.gym_parking_env import GymParkingEnv
        # GymParkingEnv will load config_env.yaml automatically
        env = GymParkingEnv(config=None, scenario="parallel")
    else:
        env = make_curriculum_env(
            scenario=args.scenario,
            phase_name=args.phase_name,
            curriculum_config_path=args.curriculum_config,
        )

    # Statistics
    successes = 0
    collisions = 0
    timeouts = 0
    rewards = []
    lengths = []

    print(f"\nRunning {args.num_episodes} episodes...")

    for ep in range(args.num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        step = 0

        while not done:
            # Get action from policy
            action = algo.compute_single_action(
                obs,
                explore=not args.deterministic,
                policy_id="default_policy"
            )

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            step += 1

        # Track stats
        rewards.append(episode_reward)
        lengths.append(step)

        if info.get("success", False):
            successes += 1
        elif info.get("termination") == "collision":
            collisions += 1
        else:
            timeouts += 1

        # Progress indicator
        if (ep + 1) % 10 == 0:
            print(f"  Progress: {ep + 1}/{args.num_episodes} episodes")

    # Calculate statistics
    success_rate = successes / args.num_episodes
    collision_rate = collisions / args.num_episodes
    timeout_rate = timeouts / args.num_episodes
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    mean_length = np.mean(lengths)
    std_length = np.std(lengths)

    # Display results
    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    print(f"Total Episodes: {args.num_episodes}")
    print()
    print("OUTCOMES:")
    print(f"  Success Rate:   {success_rate:.1%} ({successes} episodes)")
    print(f"  Collision Rate: {collision_rate:.1%} ({collisions} episodes)")
    print(f"  Timeout Rate:   {timeout_rate:.1%} ({timeouts} episodes)")
    print()
    print("PERFORMANCE:")
    print(f"  Mean Reward:  {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"  Mean Length:  {mean_length:.1f} ± {std_length:.1f} steps")
    print()

    # Grade the policy
    if success_rate >= 0.9:
        grade = "EXCELLENT ★★★★★"
    elif success_rate >= 0.7:
        grade = "VERY GOOD ★★★★"
    elif success_rate >= 0.5:
        grade = "GOOD ★★★"
    elif success_rate >= 0.3:
        grade = "FAIR ★★"
    else:
        grade = "NEEDS IMPROVEMENT ★"

    print(f"Grade: {grade}")
    print("=" * 80)

    # Cleanup
    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a trained checkpoint without visualization")

    # Checkpoint
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint directory")

    # Testing
    parser.add_argument("--num-episodes", type=int, default=20,
                        help="Number of episodes to test")
    parser.add_argument("--deterministic", action="store_true",
                        help="Use deterministic policy (no exploration)")

    # Environment
    parser.add_argument("--scenario", type=str, default="parallel",
                        help="Scenario to use")
    parser.add_argument("--phase-name", type=str, default="phase1_foundation",
                        help="Phase name for curriculum environments")
    parser.add_argument("--curriculum-config", type=str, default="rl/curriculum_config.yaml",
                        help="Path to curriculum config")

    args = parser.parse_args()

    test_checkpoint(args)
