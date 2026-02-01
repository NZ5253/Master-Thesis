"""
PPO training script for parallel parking using RLlib.

Inspired by RL_APS repository with optimized hyperparameters for parking task.
"""

import os
import argparse
import yaml
from datetime import datetime

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

from rl.gym_parking_env import make_parking_env


def create_env(env_config):
    """Environment factory for RLlib."""
    return make_parking_env(
        scenario=env_config.get("scenario", "parallel"),
        config=env_config.get("parking_config", None)
    )


def train_ppo(args):
    """
    Train PPO agent on parallel parking task.

    Args:
        args: Command line arguments
    """
    # Initialize Ray
    ray.init(
        num_cpus=args.num_cpus,
        num_gpus=args.num_gpus,
        ignore_reinit_error=True
    )

    # Register environment
    register_env("parking_env", create_env)

    # Load parking environment config
    with open("config_env.yaml", "r") as f:
        env_config_full = yaml.safe_load(f)

    env_config = {
        "scenario": args.scenario,
        "parking_config": env_config_full
    }

    # ========== PPO Configuration ==========
    # Optimized based on RL_APS and parking task characteristics
    config = (
        PPOConfig()
        .environment(
            env="parking_env",
            env_config=env_config,
            # Clip rewards for stability
            clip_rewards=False,  # We handle reward scaling in reward function
        )
        .framework("torch")
        .resources(
            num_gpus=args.num_gpus,
            num_cpus_per_worker=1,
        )
        .rollouts(
            num_rollout_workers=args.num_workers,
            num_envs_per_worker=1,
            rollout_fragment_length=200,  # Episode length ~150-300 steps
            batch_mode="truncate_episodes",
        )
        .training(
            # Learning rate schedule (start high, decay)
            lr=args.lr,
            lr_schedule=[
                [0, args.lr],
                [args.total_timesteps, args.lr * 0.1]
            ],
            # PPO hyperparameters (close to defaults, tuned for continuous control)
            gamma=0.99,  # Discount factor
            lambda_=0.95,  # GAE lambda
            clip_param=0.2,  # PPO clip epsilon
            vf_clip_param=10.0,  # Value function clip
            entropy_coeff=args.entropy_coeff,  # Entropy bonus (exploration)
            entropy_coeff_schedule=[
                [0, args.entropy_coeff],
                [args.total_timesteps, args.entropy_coeff * 0.1]
            ],
            kl_coeff=0.2,  # KL penalty coefficient
            kl_target=0.01,  # Target KL divergence
            # Training batch size and epochs
            train_batch_size=args.train_batch_size,
            sgd_minibatch_size=args.sgd_minibatch_size,
            num_sgd_iter=args.num_sgd_iter,
            # Value function
            vf_loss_coeff=0.5,
            use_critic=True,
            use_gae=True,
            # Gradient clipping
            grad_clip=0.5,
        )
        .exploration(
            explore=True,
            exploration_config={
                "type": "StochasticSampling",  # Sample from policy distribution
            }
        )
        .evaluation(
            evaluation_interval=args.eval_interval,
            evaluation_duration=10,  # 10 episodes per evaluation
            evaluation_duration_unit="episodes",
            evaluation_num_workers=1,
            evaluation_config={
                "explore": False,  # Deterministic evaluation
            }
        )
        .reporting(
            min_sample_timesteps_per_iteration=args.train_batch_size,
            min_train_timesteps_per_iteration=args.train_batch_size,
        )
        .debugging(
            log_level="INFO",
            seed=args.seed,
        )
    )

    # ========== Model Configuration ==========
    # Custom network architecture inspired by RL_APS
    config.model = {
        "fcnet_hiddens": [256, 256],  # 2 hidden layers, 256 units each
        "fcnet_activation": "tanh",  # Tanh activation (better for continuous control)
        "vf_share_layers": False,  # Separate value function network
        "use_lstm": False,
        "max_seq_len": 20,
        "free_log_std": True,  # Learn log_std independently
        "no_final_linear": False,
        # Note: fcnet_bias/weights_initializer not fully supported in Ray 2.9
        # Network will use default initialization
    }

    # Build algorithm
    algo = config.build()

    # ========== Training Loop ==========
    print("=" * 80)
    print(f"Starting PPO training for {args.scenario} parking")
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Workers: {args.num_workers}")
    print(f"GPUs: {args.num_gpus}")
    print("=" * 80)

    # Create checkpoint directory
    checkpoint_dir = os.path.join(
        args.checkpoint_dir,
        f"{args.scenario}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save training config
    config_save_path = os.path.join(checkpoint_dir, "training_config.yaml")
    with open(config_save_path, "w") as f:
        yaml.dump(vars(args), f)

    print(f"Checkpoints will be saved to: {checkpoint_dir}\n")

    best_reward = -float("inf")
    timesteps_elapsed = 0

    try:
        iteration = 0
        while timesteps_elapsed < args.total_timesteps:
            iteration += 1

            # Train for one iteration
            result = algo.train()

            timesteps_elapsed = result["timesteps_total"]
            episode_reward_mean = result["episode_reward_mean"]

            # Print progress
            print(f"\n{'='*80}")
            print(f"Iteration {iteration} | Timesteps: {timesteps_elapsed:,}/{args.total_timesteps:,}")
            print(f"{'='*80}")
            print(f"Episode reward mean: {episode_reward_mean:.2f}")
            print(f"Episode length mean: {result['episode_len_mean']:.1f}")

            # Print evaluation results if available
            if "evaluation" in result and result["evaluation"]:
                eval_reward = result["evaluation"]["episode_reward_mean"]
                print(f"Evaluation reward: {eval_reward:.2f}")

                # Save best model
                if eval_reward > best_reward:
                    best_reward = eval_reward
                    best_path = os.path.join(checkpoint_dir, "best_checkpoint")
                    algo.save(best_path)
                    print(f"New best model saved! Reward: {eval_reward:.2f}")

            # Periodic checkpoint
            if iteration % args.save_interval == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{iteration}")
                algo.save(checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path}")

        print("\n" + "=" * 80)
        print("Training completed!")
        print("=" * 80)

        # Final evaluation
        print("\nRunning final evaluation...")
        final_eval = algo.evaluate()
        print(f"Final evaluation reward: {final_eval['evaluation']['episode_reward_mean']:.2f}")

        # Save final model
        final_path = os.path.join(checkpoint_dir, "final_checkpoint")
        algo.save(final_path)
        print(f"Final model saved to: {final_path}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        interrupt_path = os.path.join(checkpoint_dir, "interrupted_checkpoint")
        algo.save(interrupt_path)
        print(f"Model saved to: {interrupt_path}")

    finally:
        algo.stop()
        ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO agent for parallel parking")

    # Environment
    parser.add_argument("--scenario", type=str, default="parallel",
                        choices=["parallel", "perpendicular"],
                        help="Parking scenario")

    # Training
    parser.add_argument("--total-timesteps", type=int, default=500000,
                        help="Total training timesteps (default: 500k)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate (default: 3e-4)")
    parser.add_argument("--train-batch-size", type=int, default=4000,
                        help="Training batch size (default: 4000)")
    parser.add_argument("--sgd-minibatch-size", type=int, default=128,
                        help="SGD minibatch size (default: 128)")
    parser.add_argument("--num-sgd-iter", type=int, default=10,
                        help="Number of SGD iterations per update (default: 10)")
    parser.add_argument("--entropy-coeff", type=float, default=0.01,
                        help="Entropy coefficient for exploration (default: 0.01)")

    # Resources
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of parallel workers (default: 4)")
    parser.add_argument("--num-cpus", type=int, default=8,
                        help="Number of CPUs (default: 8)")
    parser.add_argument("--num-gpus", type=int, default=0,
                        help="Number of GPUs (default: 0)")

    # Evaluation & Checkpointing
    parser.add_argument("--eval-interval", type=int, default=10,
                        help="Evaluation interval in iterations (default: 10)")
    parser.add_argument("--save-interval", type=int, default=50,
                        help="Checkpoint save interval (default: 50)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/ppo",
                        help="Checkpoint directory")

    # Misc
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    train_ppo(args)
