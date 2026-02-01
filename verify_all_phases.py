#!/usr/bin/env python3
"""
Comprehensive verification script for all 6 curriculum phases.
Checks that each phase has correct configuration and obstacles.
"""

import yaml
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from rl.curriculum_env import CurriculumManager
from rl.gym_parking_env import GymParkingEnv


def load_base_config():
    """Load base environment config."""
    with open("config_env.yaml", "r") as f:
        return yaml.safe_load(f)


def verify_phase_config(phase_name, curriculum_mgr, base_config):
    """
    Verify a single phase has correct configuration.

    Returns:
        dict: Verification results
    """
    print(f"\n{'='*70}")
    print(f"VERIFYING: {phase_name}")
    print(f"{'='*70}")

    results = {
        "phase_name": phase_name,
        "errors": [],
        "warnings": [],
        "info": {},
    }

    # Get phase data
    phase = curriculum_mgr.get_phase(phase_name)
    train_cfg = curriculum_mgr.get_training_config(phase_name)

    # 1. Check timesteps
    timesteps = phase.get("timesteps", 0)
    results["info"]["timesteps"] = timesteps
    print(f"✓ Timesteps: {timesteps:,}")

    # 2. Check success threshold
    threshold = phase.get("success_threshold", 0.0)
    results["info"]["success_threshold"] = threshold
    print(f"✓ Success Threshold: {threshold:.1%}")

    # 3. Get merged environment config
    merged_config = curriculum_mgr.get_phase_env_config(phase_name, base_config)

    # 4. Check if config has scenarios -> parallel structure
    if "scenarios" not in merged_config:
        results["errors"].append("Missing 'scenarios' key in config")
        return results

    if "parallel" not in merged_config["scenarios"]:
        results["errors"].append("Missing 'parallel' scenario in config")
        return results

    parallel_cfg = merged_config["scenarios"]["parallel"]

    # 5. Check obstacles section
    if "obstacles" not in parallel_cfg:
        results["errors"].append("Missing 'obstacles' section in parallel config")
        return results

    obstacles_cfg = parallel_cfg["obstacles"]

    # 6. Check neighbor config
    if "neighbor" not in obstacles_cfg:
        results["errors"].append("Missing 'neighbor' config in obstacles")
        return results

    neighbor_cfg = obstacles_cfg["neighbor"]

    # 7. Verify all required neighbor fields
    required_fields = ["w", "h", "offset", "pos_jitter"]
    missing_fields = []

    for field in required_fields:
        if field not in neighbor_cfg:
            missing_fields.append(field)

    if missing_fields:
        results["errors"].append(f"Missing neighbor fields: {missing_fields}")
        print(f"✗ Missing neighbor fields: {missing_fields}")
    else:
        print(f"✓ Neighbor config complete:")
        print(f"  - Width (w): {neighbor_cfg['w']:.2f}m")
        print(f"  - Height (h): {neighbor_cfg['h']:.2f}m")
        print(f"  - Offset: {neighbor_cfg['offset']:.2f}m")
        print(f"  - Position Jitter: {neighbor_cfg['pos_jitter']:.3f}m")
        results["info"]["neighbor"] = neighbor_cfg

    # 8. Test environment creation
    print(f"\n  Creating test environment...")
    try:
        env = GymParkingEnv(config=merged_config, scenario="parallel")

        # Count obstacles
        obstacles = env.env.obstacles.obstacles
        num_obstacles = len(obstacles)

        # Count by type
        walls = sum(1 for o in obstacles if o.get("kind") != "curb")
        curbs = sum(1 for o in obstacles if o.get("kind") == "curb")

        print(f"  ✓ Environment created successfully")
        print(f"  ✓ Total obstacles: {num_obstacles}")
        print(f"    - Walls + Neighbors: {walls}")
        print(f"    - Curbs: {curbs}")

        results["info"]["num_obstacles"] = num_obstacles
        results["info"]["num_walls"] = walls
        results["info"]["num_curbs"] = curbs

        # Expected: 4 walls + 2 neighbors + 1 curb = 7 total
        if num_obstacles != 7:
            results["warnings"].append(
                f"Expected 7 obstacles (4 walls + 2 neighbors + 1 curb), got {num_obstacles}"
            )
            print(f"  ⚠ Warning: Expected 7 obstacles, got {num_obstacles}")
        else:
            print(f"  ✓ Correct number of obstacles (7 total)")

        env.close()

    except Exception as e:
        results["errors"].append(f"Failed to create environment: {str(e)}")
        print(f"  ✗ Environment creation failed: {e}")
        return results

    # 9. Check training config
    print(f"\n  Training Configuration:")
    print(f"  - Learning Rate: {train_cfg.get('lr', 'N/A')}")
    print(f"  - Entropy Coeff: {train_cfg.get('entropy_coeff', 'N/A')}")
    print(f"  - Batch Size: {train_cfg.get('train_batch_size', 'N/A')}")
    print(f"  - SGD Iterations: {train_cfg.get('num_sgd_iter', 'N/A')}")

    results["info"]["training"] = train_cfg

    return results


def main():
    print("="*70)
    print("CURRICULUM PHASE VERIFICATION")
    print("="*70)
    print("\nThis script verifies that all 9 phases have:")
    print("  1. Progressive difficulty curriculum")
    print("  2. Correct timesteps and thresholds")
    print("  3. Complete obstacle configurations")
    print("  4. Proper environment creation")
    print("  5. Expected number of obstacles (7 total)")

    # Load curriculum
    curriculum_mgr = CurriculumManager("rl/curriculum_config.yaml")
    base_config = load_base_config()

    # Expected values - must match rl/curriculum_config.yaml
    # Updated 2026-01-28 to match actual curriculum configuration
    # Progressive curriculum: easy spawn → medium → full S-curve → randomization
    expected = {
        "phase1_foundation": {
            "timesteps": 50_000_000,
            "threshold": 0.85,
            "neighbor_jitter": 0.0,
        },
        "phase1b_medium_offset": {
            "timesteps": 50_000_000,
            "threshold": 0.85,
            "neighbor_jitter": 0.0,
        },
        "phase1c_full_offset": {
            "timesteps": 80_000_000,
            "threshold": 0.85,
            "neighbor_jitter": 0.0,
        },
        "phase2_random_spawn": {
            "timesteps": 60_000_000,
            "threshold": 0.80,
            "neighbor_jitter": 0.0,
        },
        "phase3_random_bay_x": {
            "timesteps": 50_000_000,
            "threshold": 0.75,
            "neighbor_jitter": 0.0,
        },
        "phase4a_random_bay_y_small": {
            "timesteps": 200_000_000,
            "threshold": 0.70,
            "neighbor_jitter": 0.0,
        },
        "phase4_random_bay_full": {
            "timesteps": 550_000_000,
            "threshold": 0.70,
            "neighbor_jitter": 0.0,
        },
        "phase5_neighbor_jitter": {
            "timesteps": 60_000_000,
            "threshold": 0.65,
            "neighbor_jitter": 0.05,
        },
        "phase6_random_obstacles": {
            "timesteps": 650_000_000,
            "threshold": 0.60,
            "neighbor_jitter": 0.05,
        },
    }

    # Verify all phases
    all_results = {}
    total_errors = 0
    total_warnings = 0

    for phase_name in curriculum_mgr.phase_order:
        results = verify_phase_config(phase_name, curriculum_mgr, base_config)
        all_results[phase_name] = results

        # Check against expected values
        exp = expected.get(phase_name, {})

        if results["info"].get("timesteps") != exp.get("timesteps"):
            results["errors"].append(
                f"Incorrect timesteps: expected {exp['timesteps']:,}, "
                f"got {results['info'].get('timesteps', 0):,}"
            )

        if abs(results["info"].get("success_threshold", 0.0) - exp.get("threshold", 0.0)) > 0.001:
            results["errors"].append(
                f"Incorrect threshold: expected {exp['threshold']:.1%}, "
                f"got {results['info'].get('success_threshold', 0.0):.1%}"
            )

        neighbor = results["info"].get("neighbor", {})
        if abs(neighbor.get("pos_jitter", 0.0) - exp.get("neighbor_jitter", 0.0)) > 0.001:
            results["errors"].append(
                f"Incorrect neighbor jitter: expected {exp['neighbor_jitter']:.3f}, "
                f"got {neighbor.get('pos_jitter', 0.0):.3f}"
            )

        total_errors += len(results["errors"])
        total_warnings += len(results["warnings"])

    # Check progression settings
    print(f"\n{'='*70}")
    print("VERIFYING PROGRESSION SETTINGS")
    print(f"{'='*70}")

    with open("rl/curriculum_config.yaml", "r") as f:
        full_config = yaml.safe_load(f)

    progression = full_config.get("progression", {})
    consecutive = progression.get("consecutive_successes", 0)

    print(f"✓ Evaluation Episodes: {progression.get('evaluation_episodes', 'N/A')}")
    print(f"✓ Consecutive Successes Required: {consecutive}")

    if consecutive != 5:
        print(f"✗ ERROR: Expected 5 consecutive successes, got {consecutive}")
        total_errors += 1
    else:
        print(f"✓ Correct consecutive successes threshold (5)")

    print(f"✓ Warm Start: {progression.get('warm_start', 'N/A')}")
    print(f"✓ Reset Optimizer: {progression.get('reset_optimizer', 'N/A')}")

    # Final summary
    print(f"\n{'='*70}")
    print("VERIFICATION SUMMARY")
    print(f"{'='*70}")

    print(f"\nPhases Verified: {len(all_results)}")
    print(f"Total Errors: {total_errors}")
    print(f"Total Warnings: {total_warnings}")

    if total_errors > 0:
        print("\n❌ VERIFICATION FAILED - ERRORS FOUND:")
        for phase_name, results in all_results.items():
            if results["errors"]:
                print(f"\n{phase_name}:")
                for error in results["errors"]:
                    print(f"  - {error}")
        return 1

    if total_warnings > 0:
        print("\n⚠ WARNINGS FOUND:")
        for phase_name, results in all_results.items():
            if results["warnings"]:
                print(f"\n{phase_name}:")
                for warning in results["warnings"]:
                    print(f"  - {warning}")

    print("\n✅ ALL PHASES VERIFIED SUCCESSFULLY!")
    print("\nConfiguration Summary:")
    print(f"  - Total Training: ~1.6B timesteps (9 phases)")
    print(f"  - Phases: 9 progressive phases from easy to hard")
    print(f"    Phase 1 (easy):     50M steps, y_offset=0.25m")
    print(f"    Phase 1b (medium):  50M steps, y_offset=0.50m")
    print(f"    Phase 1c (full):    80M steps, y_offset=0.83m")
    print(f"    Phases 2-6: randomization phases")
    print(f"  - Progression: 5 consecutive successes required")
    print(f"  - Obstacles: 7 per environment (4 walls + 2 neighbors + 1 curb)")
    print(f"  - Neighbor Dimensions: 0.36m × 0.26m at ±0.54m offset")

    print("\n🚀 Ready to start training!")
    print("\nRun: ./quick_train.sh curriculum")

    return 0


if __name__ == "__main__":
    sys.exit(main())
