"""
generate_eval_plots.py
======================
Standalone per-system evaluation plots for thesis.

Generates individual deep-dive plots for each system:
  - Original RL (reference car)
  - ChronosCar (deployed)
  - New Car (in training)
  - MPC Baseline (run live)

Each system gets its own figure set showing:
  1. Learning curve (reward + episode length vs cumulative timesteps, phase boundaries)
  2. PPO training health (entropy, KL divergence, policy loss, value loss)
  3. Phase progression (success rate + reward per phase, tolerance tightening)

MPC gets:
  1. Trajectory examples (3 scenarios: easy / medium / hard spawn)
  2. Final position error distribution (pos_err, yaw_err, along, lateral)
  3. Phase transition analysis (steps per phase)

Usage:
  source venv/bin/activate
  python generate_eval_plots.py             # all systems
  python generate_eval_plots.py --skip-mpc  # skip MPC (fast)
  python generate_eval_plots.py --mpc-only  # only MPC
  python generate_eval_plots.py --mpc-episodes 50  # more MPC eval episodes
"""

import os
import json
import argparse
import warnings
import numpy as np
import yaml
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, FancyArrowPatch
from pathlib import Path

warnings.filterwarnings("ignore")

# ============================================================
# Publication style
# ============================================================
plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

OUT_DIR = Path("thesis_figures/eval")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RAY_BASE = Path(os.path.expanduser("~/ray_results"))

# ============================================================
# Data sources for each system
# ============================================================

SYSTEMS = {
    "original_rl": {
        "label": "Original RL (Reference Car)",
        "color": "#2166ac",
        "ray_dirs": [
            "PPO_curriculum_parking_env_2026-02-03_09-00-23t5tokuoq",
        ],
        "training_logs": [
            "checkpoints/curriculum/curriculum_20260203_090023/training_log.yaml",
        ],
        "phase_names": ["P1 Foundation", "P2 Random Spawn", "P3 Bay X",
                        "P4a Bay Y (sm)", "P4 Full Random", "P5 Neighbors", "P6 Polish"],
    },
    "chronos_car": {
        "label": "ChronosCar (Deployed)",
        "color": "#d6604d",
        "ray_dirs": [
            "PPO_curriculum_parking_env_2026-02-08_22-34-0277fvokhy",
            "PPO_curriculum_parking_env_2026-02-09_23-05-26gwm0y3rg",
            "PPO_curriculum_parking_env_2026-02-10_11-38-105apksp9r",
        ],
        "training_logs": [
            "checkpoints/chronos/curriculum_20260208_223402/training_log.yaml",
            "checkpoints/chronos/curriculum_20260209_230526/training_log.yaml",
            "checkpoints/chronos/curriculum_20260210_113810/training_log.yaml",
        ],
        "phase_names": ["P1 Foundation", "P2 Random Spawn", "P3 Bay X",
                        "P4a Bay Y (sm)", "P4 Full Random", "P5 Neighbors",
                        "P6 Obstacles", "P7 Polish"],
    },
    "new_car": {
        "label": "New Car (Larger Scale)",
        "color": "#4dac26",
        "ray_dirs": [
            "PPO_curriculum_parking_env_2026-02-20_21-11-522uwqu9_1",
            "PPO_curriculum_parking_env_2026-02-25_13-43-171982rik_",
            "PPO_curriculum_parking_env_2026-02-27_14-48-43fy6gohrk",
        ],
        "training_logs": [
            "checkpoints/newcar/curriculum_20260220_211152/training_log.yaml",
            "checkpoints/newcar/curriculum_20260225_134317/training_log.yaml",
            "checkpoints/newcar/curriculum_20260227_144843/training_log.yaml",
        ],
        "phase_names": ["P1 Foundation", "P2 Random Spawn", "P3 Bay X",
                        "P4a Bay Y (sm)", "P4 Full Random", "P5 Neighbors",
                        "P6 Obstacles", "P7 Polish"],
    },
}

# Success tolerances per phase (along_tol in cm) for each system
TOLERANCES = {
    "original_rl": [12.5, 12.5, 10.0, 8.0, 7.0, 6.0, 5.0],
    "chronos_car": [4.5, 4.5, 4.5, 4.0, 3.5, 3.0, 3.0, 3.0],
    "new_car":     [5.4, 5.4, 4.5, 3.6, 3.1, 2.7, 2.7, 2.7],
}


# ============================================================
# Data loading helpers
# ============================================================

def load_training_logs(log_paths):
    """Load and concatenate per-phase training_log.yaml entries."""
    phases = []
    for path in log_paths:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            entries = yaml.safe_load(f)
        if entries:
            phases.extend(entries)
    return phases


def load_progress_csv(ray_dirs):
    """Load and concatenate progress.csv from ray_results directories.
    Returns DataFrame with cumulative timesteps offset applied."""
    dfs = []
    ts_offset = 0
    for rdir in ray_dirs:
        path = RAY_BASE / rdir / "progress.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "timesteps_total" not in df.columns:
            continue
        df = df.dropna(subset=["timesteps_total", "episode_reward_mean"])
        df["timesteps_cumul"] = df["timesteps_total"] + ts_offset
        ts_offset = df["timesteps_cumul"].max()
        dfs.append(df)
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)


def load_result_json(ray_dirs):
    """Load result.json (NDJSON) from ray_results to get PPO stats."""
    records = []
    ts_offset = 0
    for rdir in ray_dirs:
        path = RAY_BASE / rdir / "result.json"
        if not path.exists():
            continue
        local_records = []
        with open(path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    ls = (d.get("info", {})
                            .get("learner", {})
                            .get("default_policy", {})
                            .get("learner_stats", {}))
                    if not ls:
                        continue
                    local_records.append({
                        "timesteps_total": d.get("timesteps_total", 0),
                        "entropy": ls.get("entropy"),
                        "kl": ls.get("kl"),
                        "total_loss": ls.get("total_loss"),
                        "policy_loss": ls.get("policy_loss"),
                        "vf_loss": ls.get("vf_loss"),
                        "vf_explained_var": ls.get("vf_explained_var"),
                        "cur_lr": ls.get("cur_lr"),
                        "entropy_coeff": ls.get("entropy_coeff"),
                    })
                except Exception:
                    continue
        if local_records:
            df = pd.DataFrame(local_records)
            df["timesteps_cumul"] = df["timesteps_total"] + ts_offset
            ts_offset = df["timesteps_cumul"].max()
            records.append(df)
    if not records:
        return None
    return pd.concat(records, ignore_index=True)


def smooth(series, window=15):
    """Rolling mean smoothing."""
    return series.rolling(window=window, min_periods=1, center=True).mean()


def compute_phase_boundaries(phases):
    """Return cumulative timestep at each phase boundary."""
    boundaries = [0]
    cumul = 0
    for p in phases:
        cumul += p["timesteps"]
        boundaries.append(cumul)
    return boundaries


# ============================================================
# Plot 1: Learning curve (reward + ep_len)
# ============================================================

def plot_learning_curve(skey, sinfo, phases, df_progress, save=True):
    if df_progress is None:
        print(f"  [SKIP] No progress.csv for {skey}")
        return

    color = sinfo["color"]
    label = sinfo["label"]
    phase_names = sinfo["phase_names"]

    # Phase boundaries from training_log
    boundaries = compute_phase_boundaries(phases)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle(f"{label}\nLearning Curve — Reward & Episode Length", fontsize=12, fontweight="bold")

    ts = df_progress["timesteps_cumul"] / 1e6  # millions

    # --- Reward ---
    raw_reward = df_progress["episode_reward_mean"]
    smoothed_reward = smooth(raw_reward)
    ax1.plot(ts, raw_reward, color=color, alpha=0.2, linewidth=0.8)
    ax1.plot(ts, smoothed_reward, color=color, linewidth=2.0, label="Reward (smoothed)")

    # Mark best reward per phase
    for i, p in enumerate(phases):
        ax1.axhline(p["best_reward"], color=color, linestyle=":", alpha=0.4, linewidth=0.8)

    ax1.set_ylabel("Episode Reward")
    ax1.legend(loc="upper left")

    # Phase boundary shading
    for i, b in enumerate(boundaries[:-1]):
        b_m = b / 1e6
        e_m = boundaries[i + 1] / 1e6
        if i < len(phase_names):
            name = phase_names[i] if i < len(phase_names) else f"P{i+1}"
        else:
            name = f"P{i+1}"
        shade = 0.06 if i % 2 == 0 else 0.0
        ax1.axvspan(b_m, e_m, alpha=shade, color="gray")
        mid = (b_m + e_m) / 2
        ax1.text(mid, ax1.get_ylim()[1] if ax1.get_ylim()[1] != 1.0 else 0,
                 name, ha="center", va="top", fontsize=7, color="gray",
                 rotation=90 if (e_m - b_m) < 1.0 else 0)

    for b in boundaries[1:-1]:
        ax1.axvline(b / 1e6, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

    # --- Episode Length ---
    if "episode_len_mean" in df_progress.columns:
        raw_len = df_progress["episode_len_mean"]
        smoothed_len = smooth(raw_len)
        ax2.plot(ts, raw_len, color=color, alpha=0.2, linewidth=0.8)
        ax2.plot(ts, smoothed_len, color=color, linewidth=2.0, label="Ep. Length (smoothed)")
        ax2.set_ylabel("Episode Length (steps)")
        ax2.legend(loc="upper right")

        for b in boundaries[1:-1]:
            ax2.axvline(b / 1e6, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        for i, b in enumerate(boundaries[:-1]):
            b_m, e_m = b / 1e6, boundaries[i + 1] / 1e6
            shade = 0.06 if i % 2 == 0 else 0.0
            ax2.axvspan(b_m, e_m, alpha=shade, color="gray")

    ax2.set_xlabel("Cumulative Timesteps (millions)")

    plt.tight_layout()
    if save:
        stem = f"{skey}_learning_curve"
        fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(OUT_DIR / f"{stem}.png", bbox_inches="tight")
        print(f"  Saved {stem}.pdf")
    plt.close(fig)


# ============================================================
# Plot 2: PPO training health
# ============================================================

def plot_ppo_health(skey, sinfo, phases, df_ppo, save=True):
    if df_ppo is None:
        print(f"  [SKIP] No result.json for {skey}")
        return

    color = sinfo["color"]
    label = sinfo["label"]
    boundaries = compute_phase_boundaries(phases)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.suptitle(f"{label}\nPPO Training Health Metrics", fontsize=12, fontweight="bold")

    ts = df_ppo["timesteps_cumul"] / 1e6

    def _add_phase_lines(ax):
        for b in boundaries[1:-1]:
            ax.axvline(b / 1e6, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        for i, b in enumerate(boundaries[:-1]):
            b_m, e_m = b / 1e6, boundaries[i + 1] / 1e6
            ax.axvspan(b_m, e_m, alpha=0.06 if i % 2 == 0 else 0, color="gray")
        ax.set_xlabel("Cumulative Timesteps (M)")

    panels = [
        ("entropy", "Entropy", "Policy entropy (bits)", "Higher = more exploration"),
        ("kl", "KL Divergence", "Approx. KL from old policy", "Should stay < 0.01"),
        ("policy_loss", "Policy Loss", "PPO clipped surrogate loss", "Should decrease then stabilize"),
        ("vf_explained_var", "Value Function Explained Var.", "VF accuracy (1.0 = perfect)", "Should approach 1.0"),
    ]

    for ax, (col, title, ylabel, note) in zip(axes.flat, panels):
        if col not in df_ppo.columns:
            ax.set_visible(False)
            continue
        raw = df_ppo[col].dropna()
        if raw.empty:
            ax.set_visible(False)
            continue
        ts_col = df_ppo.loc[raw.index, "timesteps_cumul"] / 1e6
        ax.plot(ts_col, raw, color=color, alpha=0.15, linewidth=0.6)
        ax.plot(ts_col, smooth(raw, window=20), color=color, linewidth=1.8)
        ax.set_title(f"{title}\n({note})", fontsize=9)
        ax.set_ylabel(ylabel)
        _add_phase_lines(ax)

    plt.tight_layout()
    if save:
        stem = f"{skey}_ppo_health"
        fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(OUT_DIR / f"{stem}.png", bbox_inches="tight")
        print(f"  Saved {stem}.pdf")
    plt.close(fig)


# ============================================================
# Plot 3: Phase progression (success rate + reward + tolerances)
# ============================================================

def plot_phase_progression(skey, sinfo, phases, save=True):
    if not phases:
        print(f"  [SKIP] No phase data for {skey}")
        return

    color = sinfo["color"]
    label = sinfo["label"]
    phase_names = sinfo["phase_names"]
    tolerances = TOLERANCES.get(skey, [])

    n = len(phases)
    names = phase_names[:n] if phase_names else [p["phase"] for p in phases]
    success_rates = [p["best_success_rate"] * 100 for p in phases]
    rewards = [p["best_reward"] for p in phases]
    timesteps_M = [p["timesteps"] / 1e6 for p in phases]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(f"{label}\nPer-Phase Curriculum Progression", fontsize=12, fontweight="bold")

    x = np.arange(n)
    bar_w = 0.65

    # --- Success rate ---
    ax = axes[0]
    bars = ax.bar(x, success_rates, bar_w, color=color, alpha=0.8, edgecolor="white")
    ax.axhline(80, color="red", linestyle="--", linewidth=1.2, label="80% threshold", alpha=0.7)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Best Success Rate (%)")
    ax.set_title("Success Rate per Phase")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    for bar, val in zip(bars, success_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.legend(fontsize=8)

    # --- Reward ---
    ax = axes[1]
    bars = ax.bar(x, rewards, bar_w, color=color, alpha=0.8, edgecolor="white")
    ax.set_ylabel("Best Eval Reward")
    ax.set_title("Reward per Phase")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    for bar, val in zip(bars, rewards):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                f"{val:.0f}", ha="center", va="bottom", fontsize=7)

    # --- Training cost + tolerance ---
    ax = axes[2]
    bars = ax.bar(x, timesteps_M, bar_w, color=color, alpha=0.8, edgecolor="white",
                  label="Training steps (M)")
    ax.set_ylabel("Training Steps (M)")
    ax.set_title("Training Cost & Tolerance")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    for bar, val in zip(bars, timesteps_M):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                f"{val:.1f}M", ha="center", va="bottom", fontsize=7)

    # Overlay tolerance line if available
    if tolerances:
        tols = tolerances[:n]
        ax2 = ax.twinx()
        ax2.spines["right"].set_visible(True)
        ax2.plot(x, tols, "s--", color="darkorange", linewidth=1.5, markersize=5,
                 label="Along tol (cm)")
        ax2.set_ylabel("Along Tolerance (cm)", color="darkorange")
        ax2.tick_params(axis="y", labelcolor="darkorange")
        ax2.legend(loc="upper right", fontsize=8)

    # Total annotation
    total_M = sum(timesteps_M)
    final_sr = success_rates[-1] if success_rates else 0
    fig.text(0.5, 0.01,
             f"Total training: {total_M:.1f}M steps  |  Final success: {final_sr:.0f}%",
             ha="center", fontsize=9, color="gray")

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    if save:
        stem = f"{skey}_phase_progression"
        fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(OUT_DIR / f"{stem}.png", bbox_inches="tight")
        print(f"  Saved {stem}.pdf")
    plt.close(fig)


# ============================================================
# Plot 4: Combined learning curve (reward coloured by phase, full view)
# ============================================================

def plot_reward_by_phase(skey, sinfo, phases, df_progress, save=True):
    """Reward coloured differently per phase — shows curriculum transition points."""
    if df_progress is None or not phases:
        return

    label = sinfo["label"]
    phase_names = sinfo["phase_names"]

    PHASE_COLORS = plt.cm.tab10(np.linspace(0, 1, max(len(phases), 10)))
    boundaries = compute_phase_boundaries(phases)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    fig.suptitle(f"{label}\nReward Learning Curve — Coloured by Curriculum Phase",
                 fontsize=12, fontweight="bold")

    ts_all = df_progress["timesteps_cumul"] / 1e6
    rew_all = df_progress["episode_reward_mean"]

    # Draw raw (light gray)
    ax.plot(ts_all, rew_all, color="lightgray", linewidth=0.5, zorder=1)

    # Draw smoothed per-phase segments
    for i, (b_start, b_end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        mask = (df_progress["timesteps_cumul"] >= b_start) & \
               (df_progress["timesteps_cumul"] <= b_end)
        seg = df_progress[mask]
        if seg.empty:
            continue
        ts_seg = seg["timesteps_cumul"] / 1e6
        rew_seg = smooth(seg["episode_reward_mean"], window=10)
        name = phase_names[i] if i < len(phase_names) else f"P{i+1}"
        ax.plot(ts_seg, rew_seg, color=PHASE_COLORS[i], linewidth=2.2,
                label=name, zorder=2)
        ax.axvline(b_end / 1e6, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

        # Phase label at top
        mid = (b_start + b_end) / 2e6
        sr = phases[i]["best_success_rate"] * 100
        ax.text(mid, ax.get_ylim()[1] if ax.get_ylim()[1] != 1.0 else 1000,
                f"{name}\n({sr:.0f}%)", ha="center", va="top",
                fontsize=7, color=PHASE_COLORS[i])

    ax.set_xlabel("Cumulative Timesteps (millions)")
    ax.set_ylabel("Episode Reward Mean")
    ax.legend(loc="upper left", ncol=4, fontsize=8)

    plt.tight_layout()
    if save:
        stem = f"{skey}_reward_by_phase"
        fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(OUT_DIR / f"{stem}.png", bbox_inches="tight")
        print(f"  Saved {stem}.pdf")
    plt.close(fig)


# ============================================================
# MPC Evaluation
# ============================================================

def run_mpc_evaluation(n_episodes=30, seed=42):
    """Run MPC on the parking environment and collect metrics."""
    print(f"\n  Running MPC evaluation ({n_episodes} episodes)...")

    try:
        import sys
        sys.path.insert(0, ".")
        from env.parking_env import ParkingEnv
        from mpc.teb_mpc import TEBMPC, VehicleState, ParkingGoal, Obstacle
        from mpc.staged_controller import StagedAtoBtoCController
        import yaml

        with open("config_env.yaml") as f:
            env_cfg = yaml.safe_load(f)

        rng = np.random.RandomState(seed)
        dt = float(env_cfg.get("dt", env_cfg.get("simulation", {}).get("dt", 0.1)))

        # Build controller once (IPOPT compilation is expensive: ~14s)
        # use_hybrid_for_parking=False: single IPOPT instance instead of two
        print("  Building MPC controller (one-time IPOPT compilation)...")
        controller = StagedAtoBtoCController(
            env_cfg=env_cfg,
            dt=dt,
            config_path="mpc/config_mpc.yaml",
            use_hybrid_for_parking=False,
        )
        print("  Controller ready.")

        results = []

        for ep in range(n_episodes):
            env = ParkingEnv(env_cfg)
            obs, info = env.reset()

            def env_obs_to_teb(env):
                obs_list = []
                for o in env.obstacles.obstacles:
                    kind = o.get("kind", "")
                    if kind == "curb":
                        continue
                    if max(float(o["w"]), float(o["h"])) > 1.5:
                        continue
                    obs_list.append(Obstacle(
                        cx=float(o["x"]), cy=float(o["y"]),
                        hx=float(o["w"]) / 2.0, hy=float(o["h"]) / 2.0,
                        theta=float(o.get("theta", 0.0))
                    ))
                return obs_list

            traj_x, traj_y, traj_yaw = [], [], []
            terminated = False
            truncated = False
            steps = 0
            max_steps = 200  # cap for eval speed (IPOPT is slow)

            while not terminated and not truncated and steps < max_steps:
                state = VehicleState(
                    x=float(env.state[0]), y=float(env.state[1]),
                    yaw=float(env.state[2]), v=float(env.state[3])
                )
                gx, gy, gyaw = [float(x) for x in env.goal]
                goal = ParkingGoal(x=gx, y=gy, yaw=gyaw)
                obstacles = env_obs_to_teb(env)

                action = controller.get_control(
                    state, goal, obstacles, profile="parallel"
                )

                # Map [steer, accel] to env action space
                steer_cmd = float(np.clip(action[0], -1, 1))
                vel_cmd = float(np.clip(action[1], -1, 1))

                step_ret = env.step(np.array([steer_cmd, vel_cmd]))
                if len(step_ret) == 5:
                    obs, reward, terminated, truncated, info = step_ret
                else:
                    obs, reward, terminated, info = step_ret
                    truncated = False
                traj_x.append(float(env.state[0]))
                traj_y.append(float(env.state[1]))
                traj_yaw.append(float(env.state[2]))
                steps += 1

            # Final metrics
            x_ra, y_ra, yaw, v = [float(x) for x in env.state]
            gx, gy, gyaw = [float(x) for x in env.goal]

            L = float(env_cfg.get("vehicle", {}).get("length", 0.36))
            dtc = L / 2.0 - 0.05
            cx = x_ra + dtc * np.cos(yaw)
            cy = y_ra + dtc * np.sin(yaw)

            bay_cfg = env_cfg.get("parking", {}).get("bay", {})
            bay_yaw = float(bay_cfg.get("yaw", 0.0))
            slot = getattr(env, "bay_center",
                           np.array([0.0, float(bay_cfg.get("center_y", 0.13)), bay_yaw]))
            slot_cx, slot_cy = float(slot[0]), float(slot[1])

            along_err = cx - slot_cx
            lateral_err = cy - slot_cy
            pos_err = float(np.hypot(cx - slot_cx, cy - slot_cy))
            yaw_err = float(abs(((gyaw - yaw + np.pi) % (2 * np.pi)) - np.pi))
            term = info.get("termination", "timeout") if info else "timeout"

            results.append({
                "episode": ep,
                "steps": steps,
                "termination": term,
                "success": term == "success",
                "pos_err": pos_err,
                "yaw_err_deg": np.degrees(yaw_err),
                "along_err": along_err,
                "lateral_err": lateral_err,
                "traj_x": traj_x,
                "traj_y": traj_y,
                "traj_yaw": traj_yaw,
            })

            status = "✓" if term == "success" else ("✗col" if "collision" in term else "✗to")
            if (ep + 1) % 5 == 0:
                sr = sum(r["success"] for r in results) / len(results) * 100
                print(f"    Ep {ep+1:3d}/{n_episodes} | {status} | "
                      f"steps={steps:3d} | pos={pos_err:.3f}m "
                      f"yaw={np.degrees(yaw_err):.1f}° | SR={sr:.0f}%")

        success_rate = sum(r["success"] for r in results) / len(results)
        print(f"\n  MPC evaluation complete: {success_rate*100:.1f}% success ({n_episodes} eps)")
        return results

    except Exception as e:
        import traceback
        print(f"  [ERROR] MPC evaluation failed: {e}")
        traceback.print_exc()
        return None


def plot_mpc_results(results, save=True):
    """Plot MPC evaluation results."""
    if not results:
        print("  [SKIP] No MPC results to plot")
        return

    successes = [r for r in results if r["success"]]
    failures = [r for r in results if not r["success"]]
    success_rate = len(successes) / len(results) * 100

    # ---- Figure A: Trajectory examples ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"MPC Baseline — Example Trajectories\n"
                 f"Success rate: {success_rate:.0f}% ({len(successes)}/{len(results)} episodes)",
                 fontsize=12, fontweight="bold")

    # Pick 3 representative trajectories
    def pick_examples():
        examples = []
        # Best success (fewest steps)
        if successes:
            best = min(successes, key=lambda r: r["steps"])
            examples.append(("Best Success", best, "#2166ac"))
        # Typical success (median steps)
        if len(successes) > 2:
            sorted_suc = sorted(successes, key=lambda r: r["steps"])
            mid = sorted_suc[len(sorted_suc) // 2]
            examples.append(("Typical Success", mid, "#4dac26"))
        # A failure
        if failures:
            examples.append(("Failure", failures[0], "#d6604d"))
        # Pad with None
        while len(examples) < 3:
            examples.append(None)
        return examples

    examples = pick_examples()

    for ax, ex in zip(axes, examples):
        if ex is None:
            ax.set_visible(False)
            continue
        title, r, col = ex

        ax.set_aspect("equal")
        ax.set_xlim(-1.4, 1.4)
        ax.set_ylim(-1.4, 1.4)

        # Draw trajectory
        tx, ty = r["traj_x"], r["traj_y"]
        ax.plot(tx, ty, color=col, linewidth=1.5, alpha=0.8)
        ax.plot(tx[0], ty[0], "o", color=col, markersize=7, label="Start")
        ax.plot(tx[-1], ty[-1], "s", color=col, markersize=7, label="End")

        # Draw parking bay (approximate)
        bay_rect = Rectangle((-0.125, 0.05), 0.25, 0.15,
                              fill=True, facecolor="lightgray",
                              edgecolor="black", linewidth=1.5, label="Bay")
        ax.add_patch(bay_rect)

        result_str = "✓ SUCCESS" if r["success"] else f"✗ {r['termination'].upper()}"
        ax.set_title(f"{title}\n{result_str}\n"
                     f"Steps: {r['steps']} | pos_err: {r['pos_err']:.3f}m | "
                     f"yaw: {r['yaw_err_deg']:.1f}°",
                     fontsize=8)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.legend(fontsize=7)

    plt.tight_layout()
    if save:
        fig.savefig(OUT_DIR / "mpc_trajectories.pdf", bbox_inches="tight")
        fig.savefig(OUT_DIR / "mpc_trajectories.png", bbox_inches="tight")
        print("  Saved mpc_trajectories.pdf")
    plt.close(fig)

    # ---- Figure B: Error distributions ----
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(f"MPC Baseline — Final Position Error Distributions\n"
                 f"(n={len(results)}, success={success_rate:.0f}%)",
                 fontsize=12, fontweight="bold")

    cols_labels = [
        ("pos_err", "Position Error (m)", "#2166ac", [0, 0]),
        ("yaw_err_deg", "Yaw Error (°)", "#d6604d", [0, 1]),
        ("along_err", "Along-Bay Error (m)", "#4dac26", [1, 0]),
        ("lateral_err", "Lateral Error (m)", "#762a83", [1, 1]),
    ]

    for col, xlabel, color, (ri, ci) in cols_labels:
        ax = axes[ri][ci]
        vals_s = [r[col] for r in successes]
        vals_f = [r[col] for r in failures]
        if vals_s:
            ax.hist(vals_s, bins=15, color=color, alpha=0.7, label=f"Success (n={len(vals_s)})")
        if vals_f:
            ax.hist(vals_f, bins=15, color="gray", alpha=0.5, label=f"Fail (n={len(vals_f)})")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
        if vals_s:
            ax.axvline(np.mean(vals_s), color=color, linestyle="--", linewidth=1.2,
                       label=f"μ={np.mean(vals_s):.3f}")

    plt.tight_layout()
    if save:
        fig.savefig(OUT_DIR / "mpc_error_distributions.pdf", bbox_inches="tight")
        fig.savefig(OUT_DIR / "mpc_error_distributions.png", bbox_inches="tight")
        print("  Saved mpc_error_distributions.pdf")
    plt.close(fig)

    # ---- Figure C: Steps to park distribution ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("MPC Baseline — Efficiency & Outcome Breakdown",
                 fontsize=12, fontweight="bold")

    # Steps histogram
    ax = axes[0]
    all_steps = [r["steps"] for r in results]
    suc_steps = [r["steps"] for r in successes]
    ax.hist(all_steps, bins=20, color="lightgray", edgecolor="white", label="All")
    ax.hist(suc_steps, bins=20, color="#2166ac", alpha=0.7, label="Success")
    ax.set_xlabel("Steps to Terminate")
    ax.set_ylabel("Count")
    ax.set_title("Steps Distribution")
    ax.legend()
    if suc_steps:
        ax.axvline(np.mean(suc_steps), color="#2166ac", linestyle="--",
                   label=f"μ={np.mean(suc_steps):.0f}")

    # Outcome pie
    ax = axes[1]
    term_counts = {}
    for r in results:
        t = r["termination"]
        term_counts[t] = term_counts.get(t, 0) + 1
    pie_colors = {"success": "#4dac26", "collision": "#d6604d", "timeout": "#f4a742"}
    labels = list(term_counts.keys())
    sizes = [term_counts[l] for l in labels]
    colors = [pie_colors.get(l, "gray") for l in labels]
    ax.pie(sizes, labels=[f"{l}\n({v})" for l, v in term_counts.items()],
           colors=colors, autopct="%1.0f%%", startangle=90)
    ax.set_title(f"Outcome Breakdown\n(n={len(results)})")

    plt.tight_layout()
    if save:
        fig.savefig(OUT_DIR / "mpc_outcomes.pdf", bbox_inches="tight")
        fig.savefig(OUT_DIR / "mpc_outcomes.png", bbox_inches="tight")
        print("  Saved mpc_outcomes.pdf")
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Generate per-system evaluation plots")
    parser.add_argument("--skip-mpc", action="store_true", help="Skip MPC evaluation")
    parser.add_argument("--mpc-only", action="store_true", help="Only run MPC evaluation")
    parser.add_argument("--mpc-episodes", type=int, default=30, help="MPC evaluation episodes")
    args = parser.parse_args()

    if not args.mpc_only:
        for skey, sinfo in SYSTEMS.items():
            print(f"\n{'='*60}")
            print(f"  {sinfo['label']}")
            print(f"{'='*60}")

            # Load data
            phases = load_training_logs(sinfo["training_logs"])
            df_progress = load_progress_csv(sinfo["ray_dirs"])
            df_ppo = load_result_json(sinfo["ray_dirs"])

            print(f"  Phases: {len(phases)} | Progress rows: "
                  f"{len(df_progress) if df_progress is not None else 0} | "
                  f"PPO records: {len(df_ppo) if df_ppo is not None else 0}")

            # Generate all 4 plots
            plot_learning_curve(skey, sinfo, phases, df_progress)
            plot_ppo_health(skey, sinfo, phases, df_ppo)
            plot_phase_progression(skey, sinfo, phases)
            plot_reward_by_phase(skey, sinfo, phases, df_progress)

    if not args.skip_mpc:
        print(f"\n{'='*60}")
        print("  MPC Baseline Evaluation")
        print(f"{'='*60}")
        mpc_results = run_mpc_evaluation(n_episodes=args.mpc_episodes)
        plot_mpc_results(mpc_results)

    print(f"\n{'='*60}")
    print(f"All evaluation figures saved to: {OUT_DIR}/")
    files = sorted(OUT_DIR.glob("*"))
    for f in files:
        size_kb = f.stat().st_size // 1024
        print(f"  {f.name}  ({size_kb}KB)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
