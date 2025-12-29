"""
Analyze hybrid vs baseline expert data to compare performance.

This script compares the performance of the hybrid TEB+MPC controller
against the baseline MPC controller.
"""

import pickle
import os
import numpy as np
from typing import List, Dict


def analyze_episode(filepath: str) -> Dict:
    """
    Analyze a single episode file.

    Returns:
        Dict with episode statistics
    """
    with open(filepath, "rb") as f:
        data = pickle.load(f)

    traj = data["traj"]
    termination = data["termination"]

    n_steps = len(traj)
    final_obs, final_action = traj[-1]

    # Extract final position error (if available in observation)
    # The observation format should include goal distance
    final_pos_err = None
    if hasattr(final_obs, "__len__") and len(final_obs) > 6:
        # Typical obs includes [x, y, yaw, v, goal_rel_x, goal_rel_y, ...]
        # final_pos_err = sqrt(goal_rel_x^2 + goal_rel_y^2)
        # This is approximate - actual calculation depends on observation format
        pass

    return {
        "n_steps": n_steps,
        "termination": termination,
        "success": termination == "success"
    }


def analyze_directory(dir_path: str) -> Dict:
    """
    Analyze all episodes in a directory.

    Returns:
        Dict with aggregate statistics
    """
    if not os.path.exists(dir_path):
        return None

    episode_files = sorted([
        os.path.join(dir_path, f)
        for f in os.listdir(dir_path)
        if f.startswith("episode_") and f.endswith(".pkl")
    ])

    if not episode_files:
        return None

    steps_list = []
    success_count = 0

    for filepath in episode_files:
        ep_data = analyze_episode(filepath)
        steps_list.append(ep_data["n_steps"])
        if ep_data["success"]:
            success_count += 1

    return {
        "n_episodes": len(episode_files),
        "success_count": success_count,
        "success_rate": success_count / len(episode_files) if episode_files else 0,
        "steps_mean": np.mean(steps_list),
        "steps_std": np.std(steps_list),
        "steps_min": np.min(steps_list),
        "steps_max": np.max(steps_list),
        "steps_all": steps_list
    }


def compare_results(baseline_dir: str, hybrid_dir: str):
    """
    Compare baseline and hybrid results.
    """
    print("="*70)
    print("HYBRID vs BASELINE COMPARISON")
    print("="*70)

    baseline_stats = analyze_directory(baseline_dir)
    hybrid_stats = analyze_directory(hybrid_dir)

    if baseline_stats is None:
        print(f"❌ No baseline data found in {baseline_dir}")
        return

    if hybrid_stats is None:
        print(f"❌ No hybrid data found in {hybrid_dir}")
        return

    print(f"\n{'Metric':<25} {'Baseline':<15} {'Hybrid':<15} {'Change'}")
    print("-" * 70)

    print(f"{'Episodes':<25} {baseline_stats['n_episodes']:<15} "
          f"{hybrid_stats['n_episodes']:<15} "
          f"{hybrid_stats['n_episodes'] - baseline_stats['n_episodes']:+d}")

    print(f"{'Success Rate':<25} {baseline_stats['success_rate']*100:.1f}%{'':<10} "
          f"{hybrid_stats['success_rate']*100:.1f}%{'':<10} "
          f"{(hybrid_stats['success_rate'] - baseline_stats['success_rate'])*100:+.1f}%")

    print(f"{'Steps (mean ± std)':<25} "
          f"{baseline_stats['steps_mean']:.1f} ± {baseline_stats['steps_std']:.1f}{'':<4} "
          f"{hybrid_stats['steps_mean']:.1f} ± {hybrid_stats['steps_std']:.1f}{'':<4} "
          f"{hybrid_stats['steps_mean'] - baseline_stats['steps_mean']:+.1f}")

    print(f"{'Steps (range)':<25} "
          f"[{baseline_stats['steps_min']}, {baseline_stats['steps_max']}]{'':<10} "
          f"[{hybrid_stats['steps_min']}, {hybrid_stats['steps_max']}]")

    # Detailed step distribution
    print("\n" + "="*70)
    print("STEP DISTRIBUTION")
    print("="*70)

    print(f"\nBaseline ({baseline_stats['n_episodes']} episodes):")
    for i, steps in enumerate(baseline_stats['steps_all'][:10]):
        print(f"  Episode {i}: {steps} steps")
    if baseline_stats['n_episodes'] > 10:
        print(f"  ... and {baseline_stats['n_episodes'] - 10} more")

    print(f"\nHybrid ({hybrid_stats['n_episodes']} episodes):")
    for i, steps in enumerate(hybrid_stats['steps_all'][:10]):
        print(f"  Episode {i}: {steps} steps")
    if hybrid_stats['n_episodes'] > 10:
        print(f"  ... and {hybrid_stats['n_episodes'] - 10} more")

    # Comparison verdict
    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)

    step_improvement = baseline_stats['steps_mean'] - hybrid_stats['steps_mean']
    step_pct = (step_improvement / baseline_stats['steps_mean']) * 100

    if step_improvement > 0:
        print(f"✓ Hybrid is FASTER by {step_improvement:.1f} steps ({step_pct:.1f}%)")
    elif step_improvement < -1:
        print(f"⚠ Hybrid is SLOWER by {-step_improvement:.1f} steps ({-step_pct:.1f}%)")
    else:
        print(f"≈ Hybrid and baseline are SIMILAR (~{step_improvement:.1f} steps difference)")

    # Key metric: Oscillation analysis
    # This would require more detailed trajectory analysis
    # For now, just note that hybrid should have fewer oscillations
    print("\nKey expected improvements:")
    print("  - Fewer oscillations (zig-zag patterns)")
    print("  - More committed maneuvers (smooth steering)")
    print("  - Better trajectory quality (needs visual inspection)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare hybrid vs baseline expert data")
    parser.add_argument("--baseline", type=str, default="data/expert_parallel",
                       help="Directory with baseline episodes")
    parser.add_argument("--hybrid", type=str, default="data/expert_parallel_hybrid",
                       help="Directory with hybrid episodes")
    args = parser.parse_args()

    compare_results(args.baseline, args.hybrid)
