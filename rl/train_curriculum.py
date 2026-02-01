"""
Curriculum-based PPO training for parallel parking.

Progressively trains through 6 phases of increasing difficulty:
  Phase 1: Fixed spawn, fixed bay (foundation)
  Phase 2: Random spawn, fixed bay
  Phase 3: Random spawn, random bay X
  Phase 4: Random spawn, random bay X+Y (full)
  Phase 5: + Neighbor jitter
  Phase 6: + Random obstacles (maximum difficulty)

Each phase loads the previous phase's weights and continues training.
"""

import os
import argparse
import yaml
import numpy as np
from datetime import datetime
from pathlib import Path

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

from rl.curriculum_env import CurriculumManager, create_env_for_rllib


def evaluate_phase(algo, env_config, num_episodes=20):
    """
    Evaluate current policy on a phase with detailed metrics.

    Args:
        algo: Trained RLlib algorithm
        env_config: Environment configuration
        num_episodes: Number of episodes to evaluate

    Returns:
        Dict with evaluation metrics including termination breakdown,
        motion quality metrics (gear switches, steering oscillations),
        and safety metrics (min obstacle clearance).
    """
    from rl.curriculum_env import make_curriculum_env

    env = make_curriculum_env(
        scenario=env_config["scenario"],
        phase_name=env_config["phase_name"],
        curriculum_config_path=env_config["curriculum_config_path"],
    )

    # Counters
    successes = 0
    collisions = 0
    timeouts = 0

    # Metrics
    rewards = []
    lengths = []
    final_pos_errors = []
    final_yaw_errors = []
    
    # Detailed final state breakdown (for debugging Phase 1 failure modes)
    final_along_errors = []
    final_lat_errors = []
    final_velocities = []

    # Motion quality metrics (per checklist requirements)
    gear_switch_counts = []
    steering_oscillation_counts = []
    min_obstacle_clearances = []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=42 + ep)
        done = False
        episode_reward = 0.0
        step = 0

        # Per-episode tracking for motion quality
        prev_velocity = 0.0
        prev_steer = 0.0
        gear_switches = 0
        steer_sign_changes = 0
        episode_min_clearance = float('inf')

        while not done:
            action = algo.compute_single_action(obs, explore=False, policy_id="default_policy")
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            step += 1

            # Track gear switches (velocity sign flip)
            curr_velocity = obs[3]  # velocity is obs[3]
            if (prev_velocity > 0.05 and curr_velocity < -0.05) or \
               (prev_velocity < -0.05 and curr_velocity > 0.05):
                gear_switches += 1
            prev_velocity = curr_velocity

            # Track steering oscillations (steering sign changes)
            curr_steer = float(action[0])
            if step > 1 and ((prev_steer > 0.1 and curr_steer < -0.1) or
                             (prev_steer < -0.1 and curr_steer > 0.1)):
                steer_sign_changes += 1
            prev_steer = curr_steer

            # Track min obstacle clearance
            dist_front = obs[4]
            dist_left = obs[5]
            dist_right = obs[6]
            min_dist = min(dist_front, dist_left, dist_right)
            if min_dist < episode_min_clearance:
                episode_min_clearance = min_dist

        # Track termination reasons (from env, not reward wrapper)
        termination = info.get("termination", "unknown")
        if termination == "success":
            successes += 1
        elif termination == "collision":
            collisions += 1
        elif termination == "max_steps":
            timeouts += 1

        rewards.append(episode_reward)
        lengths.append(step)
        final_pos_errors.append(info.get("pos_err", 0.0))
        final_yaw_errors.append(info.get("yaw_err", 0.0))
        
        # Capture precise failure mode (from final observation)
        # obs: [along, lateral, yaw_err, v, ...]
        final_along_errors.append(abs(obs[0]))
        final_lat_errors.append(abs(obs[1]))
        final_velocities.append(abs(obs[3]))

        # Motion quality
        gear_switch_counts.append(gear_switches)
        steering_oscillation_counts.append(steer_sign_changes)
        min_obstacle_clearances.append(episode_min_clearance if episode_min_clearance < float('inf') else 0.0)

    success_rate = successes / num_episodes

    return {
        "success_rate": success_rate,
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "mean_length": np.mean(lengths),
        "std_length": np.std(lengths),
        # Termination breakdown
        "successes": successes,
        "collisions": collisions,
        "timeouts": timeouts,
        # Error metrics
        "mean_pos_err": np.mean(final_pos_errors),
        "mean_yaw_err": np.mean(final_yaw_errors),
        "min_pos_err": np.min(final_pos_errors) if final_pos_errors else 0.0,
        # Detailed failure analysis
        "mean_along_err": np.mean(final_along_errors),
        "mean_lat_err": np.mean(final_lat_errors),
        "mean_final_v": np.mean(final_velocities),
        # Motion quality metrics (for debugging training issues)
        "mean_gear_switches": np.mean(gear_switch_counts),
        "max_gear_switches": np.max(gear_switch_counts) if gear_switch_counts else 0,
        "mean_steer_oscillations": np.mean(steering_oscillation_counts),
        "max_steer_oscillations": np.max(steering_oscillation_counts) if steering_oscillation_counts else 0,
        # Safety metrics
        "mean_min_clearance": np.mean(min_obstacle_clearances),
        "min_clearance": np.min(min_obstacle_clearances) if min_obstacle_clearances else 0.0,
    }


def load_policy_weights_only(algo, checkpoint_path):
    """
    Load only policy weights from checkpoint, not optimizer state.
    This allows resuming between phases with different training configs.
    Skips output layer if dimensions don't match (architecture differences).
    
    Args:
        algo: The algorithm to load weights into
        checkpoint_path: Path to the checkpoint directory
    
    Returns:
        True if successful, False otherwise
    """
    try:
        from pathlib import Path
        import pickle
        import torch
        
        checkpoint_path = Path(checkpoint_path)
        policy_state_path = checkpoint_path / "policies" / "default_policy" / "policy_state.pkl"
        
        if not policy_state_path.exists():
            print(f"Warning: Policy state not found at {policy_state_path}")
            return False
        
        with open(policy_state_path, "rb") as f:
            policy_state = pickle.load(f)
        
        # Get the policy
        policy = algo.workers.local_worker().get_policy("default_policy")
        current_state = policy.model.state_dict()
        
        # Load model weights (RLlib stores them under "weights" key as numpy arrays)
        if "weights" in policy_state:
            weights = policy_state["weights"]
            
            # Convert numpy arrays to torch tensors and filter by shape match
            weights_torch = {}
            skipped_keys = []
            
            for key, value in weights.items():
                if key not in current_state:
                    skipped_keys.append(f"{key} (not in current model)")
                    continue
                    
                # Convert numpy to torch
                if isinstance(value, np.ndarray):
                    tensor_value = torch.from_numpy(value).float()
                else:
                    tensor_value = value
                
                # Check shape match
                if tensor_value.shape != current_state[key].shape:
                    skipped_keys.append(f"{key} (shape {tensor_value.shape} -> {current_state[key].shape})")
                    continue
                
                weights_torch[key] = tensor_value
            
            if weights_torch:
                policy.model.load_state_dict(weights_torch, strict=False)
                print(f"✓ Loaded {len(weights_torch)} weight tensors from checkpoint")
                if skipped_keys:
                    print(f"  (Skipped {len(skipped_keys)} incompatible/missing weights)")
                return True
            else:
                print("Warning: No compatible weights found in checkpoint")
                if skipped_keys:
                    print(f"  Skipped all {len(skipped_keys)} weights due to shape/type mismatches")
                return False
                
        else:
            print("Warning: 'weights' key not found in policy state")
            return False
            
    except Exception as e:
        print(f"Error loading policy weights: {e}")
        return False


def train_curriculum(args):
    """
    Train PPO agent with curriculum learning.

    Args:
        args: Command line arguments
    """
    # Validate resource allocation
    min_cpus_needed = args.num_workers + 1  # workers + driver
    if args.num_cpus < min_cpus_needed:
        print(f"WARNING: num_cpus ({args.num_cpus}) < required ({min_cpus_needed})")
        print(f"Setting num_cpus to {min_cpus_needed + 1} to avoid resource contention")
        args.num_cpus = min_cpus_needed + 1
    
    # Initialize Ray with reduced logging and better resource management
    ray.init(
        num_cpus=args.num_cpus,
        num_gpus=args.num_gpus,
        ignore_reinit_error=True,
        log_to_driver=False,  # Reduce driver logs
        logging_level="ERROR",  # Only show errors
        object_store_memory=500_000_000,  # 500MB object store to reduce memory usage
    )

    # Load curriculum
    curriculum_manager = CurriculumManager(args.curriculum_config)

    # Load progression settings from curriculum config
    with open(args.curriculum_config, "r") as f:
        full_curriculum_config = yaml.safe_load(f)
    progression_config = full_curriculum_config.get("progression", {})
    consecutive_successes_required = progression_config.get("consecutive_successes", 3)
    evaluation_episodes = progression_config.get("evaluation_episodes", 20)

    # Register environment
    register_env("curriculum_parking_env", create_env_for_rllib)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.checkpoint_dir) / f"curriculum_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("CURRICULUM TRAINING FOR PARALLEL PARKING")
    print("=" * 80)
    print(f"Total phases: {len(curriculum_manager.phase_order)}")
    print(f"Output directory: {output_dir}")
    print(f"Workers: {args.num_workers}")
    print(f"GPUs: {args.num_gpus}")
    print("=" * 80)

    # Save curriculum config to output
    curriculum_save_path = output_dir / "curriculum_config.yaml"
    with open(args.curriculum_config, "r") as f_in:
        with open(curriculum_save_path, "w") as f_out:
            f_out.write(f_in.read())

    # Training log
    training_log = []

    # Initialize algorithm (will be built for first phase)
    algo = None
    previous_checkpoint = None

    # Handle resume from specific phase
    start_phase_idx = 0
    resume_phase_checkpoint = None
    if args.resume_from_phase:
        if args.resume_from_phase not in curriculum_manager.phase_order:
            print(f"Error: Phase '{args.resume_from_phase}' not found in curriculum")
            print(f"Available phases: {curriculum_manager.phase_order}")
            return

        # Debug output
        print(f"Resume checkpoint arg: {args.resume_checkpoint}")
        if args.resume_checkpoint:
            checkpoint_path = Path(args.resume_checkpoint)
            print(f"Checkpoint path: {checkpoint_path}")
            print(f"Checkpoint exists: {checkpoint_path.exists()}")
            if checkpoint_path.exists():
                print(f"Checkpoint contents: {list(checkpoint_path.iterdir())}")
        
        if not args.resume_checkpoint or not Path(args.resume_checkpoint).exists():
            print(f"Error: Must provide valid --resume-checkpoint when using --resume-from-phase")
            if args.resume_checkpoint:
                print(f"Tried path: {args.resume_checkpoint}")
            return

        start_phase_idx = curriculum_manager.phase_order.index(args.resume_from_phase)
        # Store the checkpoint for the phase being resumed
        resume_phase_checkpoint = args.resume_checkpoint
        # For continuing to next phases, use this as the previous checkpoint
        previous_checkpoint = args.resume_checkpoint

        print("=" * 80)
        print("RESUMING TRAINING FROM CHECKPOINT")
        print("=" * 80)
        print(f"Resume phase: {args.resume_from_phase} (phase {start_phase_idx + 1})")
        print(f"Checkpoint: {args.resume_checkpoint}")
        print(f"Remaining phases: {len(curriculum_manager.phase_order) - start_phase_idx}")
        print("=" * 80)
    
    # Handle start from specific phase (fresh start)
    elif args.start_phase:
        if args.start_phase not in curriculum_manager.phase_order:
            print(f"Error: Phase '{args.start_phase}' not found in curriculum")
            print(f"Available phases: {curriculum_manager.phase_order}")
            return
        
        start_phase_idx = curriculum_manager.phase_order.index(args.start_phase)
        
        # If not phase 1, try to load weights from previous phase
        if start_phase_idx > 0:
            prev_phase_name = curriculum_manager.phase_order[start_phase_idx - 1]
            print("=" * 80)
            print(f"STARTING FROM PHASE {start_phase_idx + 1}: {args.start_phase} (FRESH)")
            print("=" * 80)
            print(f"Will load weights from previous phase: {prev_phase_name}")
            print("=" * 80)
        else:
            print("=" * 80)
            print(f"STARTING FROM PHASE {start_phase_idx + 1}: {args.start_phase} (COMPLETELY FRESH)")
            print("=" * 80)

    # ========== TRAIN EACH PHASE ==========
    for phase_idx, phase_name in enumerate(curriculum_manager.phase_order):
        # Skip phases before resume point
        if phase_idx < start_phase_idx:
            print(f"Skipping Phase {phase_idx + 1}: {phase_name} (already completed)")
            continue

        phase_config = curriculum_manager.get_phase(phase_name)
        train_config = curriculum_manager.get_training_config(phase_name)

        # Ensure numeric types are correct (YAML may parse them as strings in some cases)
        train_config = {
            "lr": float(train_config["lr"]),
            "entropy_coeff": float(train_config["entropy_coeff"]),
            "train_batch_size": int(train_config["train_batch_size"]),
            "num_sgd_iter": int(train_config["num_sgd_iter"]),
            "eval_interval": int(train_config["eval_interval"]),
        }

        print("\n" + "=" * 80)
        print(f"PHASE {phase_idx + 1}/{len(curriculum_manager.phase_order)}: {phase_config['name']}")
        print("=" * 80)
        print(curriculum_manager.get_phase_summary(phase_name))
        print()

        # Environment config for this phase
        env_config = {
            "scenario": "parallel",
            "phase_name": phase_name,
            "curriculum_config_path": args.curriculum_config,
        }

        # ========== Build/Update PPO Config ==========
        config = (
            PPOConfig()
            .environment(
                env="curriculum_parking_env",
                env_config=env_config,
                clip_rewards=False,
            )
            .framework("torch")
            .resources(
                num_gpus=args.num_gpus,
                num_cpus_per_worker=1,
            )
            .rollouts(
                num_rollout_workers=args.num_workers,
                num_envs_per_worker=1,
                rollout_fragment_length=200,
                batch_mode="truncate_episodes",
            )
            .training(
                lr=train_config["lr"],
                gamma=0.99,
                lambda_=0.95,
                clip_param=0.2,
                vf_clip_param=10.0,
                entropy_coeff=train_config["entropy_coeff"],
                kl_coeff=0.2,
                kl_target=0.01,
                train_batch_size=train_config["train_batch_size"],
                sgd_minibatch_size=128,
                num_sgd_iter=train_config["num_sgd_iter"],
                vf_loss_coeff=0.5,
                use_critic=True,
                use_gae=True,
                grad_clip=0.5,
            )
            .exploration(
                explore=True,
                exploration_config={"type": "StochasticSampling"},
            )
            .evaluation(
                evaluation_interval=train_config["eval_interval"],
                evaluation_duration=10,
                evaluation_duration_unit="episodes",
                evaluation_num_workers=1,
                evaluation_config={"explore": False},
            )
            .debugging(log_level="ERROR", seed=args.seed)  # Reduced logging
        )

        # Model config
        config.model = {
            "fcnet_hiddens": [256, 256],
            "fcnet_activation": "tanh",
            "vf_share_layers": False,
            "use_lstm": False,
            "max_seq_len": 20,
            "free_log_std": True,
            # Note: fcnet_weights_initializer_config not supported in Ray 2.9
        }

        # Build or restore algorithm
        if algo is None:
            # First phase or resume phase: build fresh then restore
            algo = config.build()
            
            # If resuming, restore ONLY policy weights (not optimizer state)
            if resume_phase_checkpoint is not None and phase_idx == start_phase_idx:
                print(f"Restoring policy weights from: {resume_phase_checkpoint}")
                try:
                    if load_policy_weights_only(algo, resume_phase_checkpoint):
                        print("Policy weights loaded successfully (warm start)")
                    else:
                        print("Warning: Could not load policy weights, continuing with fresh weights")
                    resume_phase_checkpoint = None  # Clear it after first use
                except Exception as e:
                    print(f"Error loading checkpoint: {e}")
                    print("Continuing with fresh weights")
            else:
                print("Created new PPO algorithm")
        else:
            # Subsequent phases (non-resume): restore from previous checkpoint and update config
            algo.stop()  # Stop current algorithm
            algo = config.build()  # Build new one with updated config

            if previous_checkpoint is not None:
                # Load weights from previous phase (policy only, not optimizer)
                print(f"Loading policy weights from: {previous_checkpoint}")
                try:
                    if load_policy_weights_only(algo, previous_checkpoint):
                        print("Policy weights loaded successfully (warm start)")
                    else:
                        print("Warning: Could not load policy weights, continuing with fresh weights")
                except Exception as e:
                    print(f"Warning: Could not load previous phase checkpoint: {e}")
                    print("Continuing with fresh weights for this phase")

        # ========== TRAIN THIS PHASE ==========
        phase_dir = output_dir / phase_name
        phase_dir.mkdir(exist_ok=True)

        target_timesteps = phase_config["timesteps"]
        success_threshold = phase_config["success_threshold"]

        # Track timesteps for THIS phase only (not global)
        # Get initial timestep count before this phase starts
        initial_result = algo.train()
        phase_start_timesteps = initial_result["timesteps_total"]
        phase_timesteps_elapsed = 0
        iteration = 1  # Already did one iteration above
        best_eval_reward = -float("inf")
        best_success_rate = 0.0
        consecutive_passes = 0
        no_improvement_count = 0

        print(f"\n{'='*80}")
        print(f"TRAINING PHASE {phase_idx + 1}: {phase_config['name']}")
        print(f"{'='*80}")
        print(f"Success threshold: {success_threshold:.1%} (must pass {consecutive_successes_required} consecutive evaluations)")
        print(f"Target timesteps: ~{target_timesteps:,}")
        print(f"Max timesteps: {target_timesteps * 10:,}")
        print(f"Training config: lr={train_config['lr']}, entropy={train_config['entropy_coeff']}")
        print(f"Starting from global timestep: {phase_start_timesteps:,}")
        print(f"{'='*80}\n")

        # Training loop - continue until success threshold AND max timesteps
        max_timesteps = target_timesteps * 10 if args.train_until_success else target_timesteps

        # Process the initial training result
        result = initial_result

        while phase_timesteps_elapsed < max_timesteps:
            # Calculate phase-specific timesteps from result
            current_total_timesteps = result["timesteps_total"]
            phase_timesteps_elapsed = current_total_timesteps - phase_start_timesteps

            # Print progress every 5 iterations
            if iteration % 5 == 0:
                print(f"[Iter {iteration}] Phase Steps: {phase_timesteps_elapsed:,}/{max_timesteps:,} | "
                      f"Reward: {result['episode_reward_mean']:.2f} | "
                      f"Length: {result['episode_len_mean']:.1f} | "
                      f"Best Success: {best_success_rate:.1%}")

            # Evaluation
            if "evaluation" in result and result["evaluation"]:
                eval_reward = result["evaluation"]["episode_reward_mean"]

                # Custom evaluation for success rate
                eval_metrics = evaluate_phase(algo, env_config, num_episodes=evaluation_episodes)
                success_rate = eval_metrics["success_rate"]

                print(f"\n[EVAL] Iter {iteration} | Phase Steps: {phase_timesteps_elapsed:,}")
                print(f"  Success Rate: {success_rate:.1%} (Threshold: {success_threshold:.1%})")
                print(f"  Terminations: {eval_metrics['successes']} success, "
                      f"{eval_metrics['collisions']} collision, "
                      f"{eval_metrics['timeouts']} timeout")
                print(f"  Errors: pos={eval_metrics['mean_pos_err']:.3f}m "
                      f"(min:{eval_metrics['min_pos_err']:.3f}m), "
                      f"yaw={np.degrees(eval_metrics['mean_yaw_err']):.1f}deg")
                
                # NEW: Detailed final state logging
                print(f"  Final State: along={eval_metrics['mean_along_err']:.3f}m, "
                      f"lat={eval_metrics['mean_lat_err']:.3f}m, "
                      f"v={eval_metrics['mean_final_v']:.3f}m/s")

                print(f"  Motion: gear_switches={eval_metrics['mean_gear_switches']:.1f} (max:{eval_metrics['max_gear_switches']}), "
                      f"steer_osc={eval_metrics['mean_steer_oscillations']:.1f}")
                print(f"  Safety: min_clearance={eval_metrics['min_clearance']:.3f}m")
                print(f"  Reward: {eval_reward:.2f} (mean)")

                # Check if passing threshold
                if success_rate >= success_threshold:
                    consecutive_passes += 1
                    print(f"  ✓ PASSED threshold! ({consecutive_passes}/{consecutive_successes_required} consecutive passes needed)")
                    no_improvement_count = 0  # Reset no-improvement counter
                else:
                    consecutive_passes = 0
                    print(f"  ✗ Below threshold (need {success_threshold:.1%}, got {success_rate:.1%})")

                # Save best model for this phase
                if eval_reward > best_eval_reward or success_rate > best_success_rate:
                    best_eval_reward = max(eval_reward, best_eval_reward)
                    best_success_rate = max(success_rate, best_success_rate)
                    best_path = phase_dir / "best_checkpoint"
                    algo.save(str(best_path))
                    print(f"  💾 Best model saved (reward: {eval_reward:.2f}, success: {success_rate:.1%})")
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                    if no_improvement_count >= 5:
                        print(f"  ⚠️  No improvement for {no_improvement_count} evaluations")

                # Advancement ONLY if passing threshold consecutively
                if consecutive_passes >= consecutive_successes_required:
                    print(f"\n{'='*80}")
                    print(f"✓✓✓ PHASE {phase_idx + 1} COMPLETE - SUCCESS THRESHOLD ACHIEVED!")
                    print(f"{'='*80}")
                    print(f"Phase timesteps: {phase_timesteps_elapsed:,}")
                    print(f"Success rate: {success_rate:.1%} (threshold: {success_threshold:.1%})")
                    print(f"Best success rate: {best_success_rate:.1%}")
                    print(f"Best eval reward: {best_eval_reward:.2f}")
                    print(f"{'='*80}\n")
                    break

                print()

            # Periodic checkpoint (less frequent to save disk space)
            if iteration % 100 == 0:
                checkpoint_path = phase_dir / f"checkpoint_{iteration}"
                algo.save(str(checkpoint_path))

            # Train next iteration (for next loop)
            iteration += 1
            if phase_timesteps_elapsed < max_timesteps:
                result = algo.train()

        # ========== PHASE COMPLETE ==========
        # Check if phase actually succeeded
        if best_success_rate < success_threshold:
            print("\n" + "!" * 80)
            print(f"⚠️  WARNING: PHASE {phase_idx + 1} DID NOT REACH SUCCESS THRESHOLD")
            print("!" * 80)
            print(f"  Required: {success_threshold:.1%}")
            print(f"  Achieved: {best_success_rate:.1%}")
            print(f"  Phase timesteps: {phase_timesteps_elapsed:,}")
            print(f"  Best eval reward: {best_eval_reward:.2f}")
            print("!" * 80)
            print("\nCannot proceed to next phase without meeting threshold.")
            print("Please review the training or adjust the threshold.")
            print("!" * 80 + "\n")

            # Save what we have and stop
            final_checkpoint = phase_dir / "final_checkpoint"
            algo.save(str(final_checkpoint))

            # Log phase results
            training_log.append({
                "phase": phase_name,
                "timesteps": phase_timesteps_elapsed,
                "best_reward": float(best_eval_reward),
                "best_success_rate": float(best_success_rate),
                "checkpoint": str(final_checkpoint),
                "status": "FAILED - Did not reach threshold"
            })

            # Save training log
            log_path = output_dir / "training_log.yaml"
            with open(log_path, "w") as f:
                yaml.dump(training_log, f)

            print(f"Training stopped. Logs saved to: {log_path}")
            algo.stop()
            ray.shutdown()
            return

        print("\n" + "=" * 80)
        print(f"PHASE {phase_idx + 1} SUMMARY")
        print("=" * 80)
        print(f"  Phase timesteps: {phase_timesteps_elapsed:,}")
        print(f"  Best eval reward: {best_eval_reward:.2f}")
        print(f"  Best success rate: {best_success_rate:.1%}")
        print(f"  Status: ✓ SUCCESS (threshold {success_threshold:.1%} achieved)")
        print("=" * 80)

        # Save final checkpoint for this phase
        final_checkpoint = phase_dir / "final_checkpoint"
        algo.save(str(final_checkpoint))
        previous_checkpoint = str(final_checkpoint)  # Use for next phase

        # Log phase results
        training_log.append({
            "phase": phase_name,
            "timesteps": phase_timesteps_elapsed,
            "best_reward": float(best_eval_reward),
            "best_success_rate": float(best_success_rate),
            "checkpoint": str(final_checkpoint),
            "status": "SUCCESS"
        })

        # Save training log
        log_path = output_dir / "training_log.yaml"
        with open(log_path, "w") as f:
            yaml.dump(training_log, f)

    # ========== FINAL EVALUATION ==========
    print("\n" + "=" * 80)
    print("CURRICULUM TRAINING COMPLETE")
    print("=" * 80)

    print("\nPhase Summary:")
    for i, log_entry in enumerate(training_log):
        print(f"  Phase {i+1} ({log_entry['phase']}): "
              f"Success {log_entry['best_success_rate']:.1%}, "
              f"Reward {log_entry['best_reward']:.2f}")

    print(f"\nAll checkpoints saved to: {output_dir}")
    print(f"Training log: {output_dir / 'training_log.yaml'}")

    # Cleanup
    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Curriculum training for parallel parking")

    # Curriculum
    parser.add_argument("--curriculum-config", type=str, default="rl/curriculum_config.yaml",
                        help="Path to curriculum config")
    parser.add_argument("--train-until-success", action="store_true",
                        help="Train each phase until success threshold is met (ignores timestep limits)")
    parser.add_argument("--start-phase", type=str, default=None,
                        help="Start training from a specific phase (e.g., 'phase2_random_spawn')")
    parser.add_argument("--resume-from-phase", type=str, default=None,
                        help="Resume training from a specific phase (e.g., 'phase3_random_bay_x')")
    parser.add_argument("--resume-checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from (must match --resume-from-phase)")

    # Resources
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of parallel workers")
    parser.add_argument("--num-cpus", type=int, default=8,
                        help="Number of CPUs")
    parser.add_argument("--num-gpus", type=int, default=0,
                        help="Number of GPUs")

    # Output
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/curriculum",
                        help="Checkpoint directory")

    # Misc
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    train_curriculum(args)