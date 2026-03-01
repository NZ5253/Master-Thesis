#!/usr/bin/env python3
"""
generate_thesis_plots.py
========================
Generates publication-quality figures for thesis from training logs.

Usage:
    source venv/bin/activate
    python generate_thesis_plots.py

Output: thesis_figures/
    fig1_success_progression.pdf   - Success rate across curriculum phases (all 3 systems)
    fig2_timesteps_per_phase.pdf   - Training cost per phase (grouped bars)
    fig3_reward_curves.pdf         - Learning curves from progress.csv (reward over time)
    fig4_episode_length.pdf        - Episode length over training (shows efficiency)
    fig5_final_comparison.pdf      - Side-by-side system comparison summary
    thesis_data_table.txt          - LaTeX table of all training results
"""

import os
import glob
import yaml
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
from collections import defaultdict

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
RAY_DIR  = os.path.expanduser("~/ray_results")
OUT_DIR  = os.path.join(BASE_DIR, "thesis_figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# Style
# ──────────────────────────────────────────────

plt.rcParams.update({
    'font.family':       'DejaVu Serif',
    'font.size':         11,
    'axes.titlesize':    12,
    'axes.labelsize':    11,
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
    'legend.fontsize':   9,
    'figure.dpi':        150,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
    'axes.grid':         True,
    'grid.alpha':        0.3,
    'axes.spines.top':   False,
    'axes.spines.right': False,
})

COLORS = {
    'original_rl': '#2196F3',   # blue
    'chronoscar':  '#FF5722',   # orange-red
    'newcar':      '#4CAF50',   # green
}

SYSTEM_LABELS = {
    'original_rl': 'Original RL (reference)',
    'chronoscar':  'ChronosCar (1/28 scale)',
    'newcar':      'New Car (encoder)',
}

# ──────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────

def load_training_log(path):
    """Load training_log.yaml and return list of phase dicts."""
    with open(path) as f:
        return yaml.safe_load(f)

def load_progress_csv(path):
    """Load progress.csv and return dict of lists."""
    data = defaultdict(list)
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                try:
                    data[k].append(float(v))
                except (ValueError, TypeError):
                    data[k].append(v)
    return dict(data)

def find_progress_csv(run_dir, date_str):
    """Find progress.csv files in ray_results matching a date string like '2026-02-08'."""
    pattern = os.path.join(RAY_DIR, f"PPO_curriculum_parking_env_{date_str}*", "progress.csv")
    files = sorted(glob.glob(pattern))
    return files

def get_largest_progress_csv(date_str):
    """Return the progress.csv with the most rows for a given date."""
    files = find_progress_csv(None, date_str)
    if not files:
        return None, None
    best_file, best_rows = None, 0
    for f in files:
        try:
            with open(f) as fp:
                n = sum(1 for _ in fp) - 1  # subtract header
            if n > best_rows:
                best_rows = n
                best_file = f
        except Exception:
            pass
    return best_file, best_rows

def smooth(data, window=20):
    """Apply rolling mean smoothing."""
    if len(data) < window:
        return np.array(data, dtype=float)
    result = np.convolve(np.array(data, dtype=float), np.ones(window)/window, mode='valid')
    # Pad to original length with NaN
    pad = len(data) - len(result)
    return np.concatenate([np.full(pad, np.nan), result])

# ──────────────────────────────────────────────
# Load canonical training data
# ──────────────────────────────────────────────

print("Loading training data...")

# Original RL (reference): best complete 6-phase run
orig_run = os.path.join(CKPT_DIR, "curriculum", "curriculum_20260203_090023", "training_log.yaml")
orig_data = load_training_log(orig_run) if os.path.exists(orig_run) else []

# ChronosCar: 3 runs form a chain (phases 1-5, phase 6, phase 7)
chronos_runs = [
    os.path.join(CKPT_DIR, "chronos", "curriculum_20260208_223402", "training_log.yaml"),
    os.path.join(CKPT_DIR, "chronos", "curriculum_20260209_230526", "training_log.yaml"),
    os.path.join(CKPT_DIR, "chronos", "curriculum_20260210_113810", "training_log.yaml"),
]
chronos_data = []
seen_phases = set()
for run in chronos_runs:
    if os.path.exists(run):
        for phase in load_training_log(run):
            if phase['phase'] not in seen_phases:
                chronos_data.append(phase)
                seen_phases.add(phase['phase'])

# New car: phases 1-4, phase 5, phases 6-7
newcar_runs = [
    os.path.join(CKPT_DIR, "newcar", "curriculum_20260220_211152", "training_log.yaml"),
    os.path.join(CKPT_DIR, "newcar", "curriculum_20260225_134317", "training_log.yaml"),
    os.path.join(CKPT_DIR, "newcar", "curriculum_20260227_144843", "training_log.yaml"),
]
newcar_data = []
seen_phases = set()
for run in newcar_runs:
    if os.path.exists(run):
        for phase in load_training_log(run):
            if phase['phase'] not in seen_phases:
                newcar_data.append(phase)
                seen_phases.add(phase['phase'])

print(f"  Original RL: {len(orig_data)} phases")
print(f"  ChronosCar:  {len(chronos_data)} phases")
print(f"  New Car:     {len(newcar_data)} phases")

# ──────────────────────────────────────────────
# Phase label mapping
# ──────────────────────────────────────────────

PHASE_LABELS = {
    'phase1_foundation':           'Ph1\nFoundation',
    'phase2_random_spawn':         'Ph2\nRandom Spawn',
    'phase3_random_bay_x':         'Ph3\nBay X Rand.',
    'phase4a_random_bay_y_small':  'Ph4a\nBay Y Small',
    'phase4_random_bay_full':      'Ph4\nFull Random',
    'phase4b_random_bay_jitter':   'Ph4b\nJitter Only',
    'phase5_neighbor_jitter':      'Ph5\nNeighbor Jitter',
    'phase5_tight_tol':            'Ph5\nTight Tol.',
    'phase6_random_obstacles':     'Ph6\nPolish',
    'phase7_polish':               'Ph7\nSettling',
    'phase8_yaw_precision':        'Ph8\nYaw Prec.',
}

PHASE_SHORT = {
    'phase1_foundation':           'P1',
    'phase2_random_spawn':         'P2',
    'phase3_random_bay_x':         'P3',
    'phase4a_random_bay_y_small':  'P4a',
    'phase4_random_bay_full':      'P4',
    'phase4b_random_bay_jitter':   'P4b',
    'phase5_neighbor_jitter':      'P5',
    'phase5_tight_tol':            'P5',
    'phase6_random_obstacles':     'P6',
    'phase7_polish':               'P7',
    'phase8_yaw_precision':        'P8',
}

# ──────────────────────────────────────────────
# FIG 1: Success rate progression across curriculum phases
# ──────────────────────────────────────────────

print("\nGenerating Fig 1: Success rate progression...")

fig, ax = plt.subplots(figsize=(10, 5))

datasets = [
    ('original_rl', orig_data),
    ('chronoscar',  chronos_data),
    ('newcar',      newcar_data),
]

for sys_key, data in datasets:
    if not data:
        continue
    phases = [d['phase'] for d in data]
    success = [d['best_success_rate'] * 100 for d in data]
    x = list(range(1, len(phases) + 1))
    label = SYSTEM_LABELS[sys_key]
    color = COLORS[sys_key]
    ax.plot(x, success, 'o-', color=color, label=label, linewidth=2, markersize=7, zorder=3)
    # Annotate each point
    for xi, yi, phase in zip(x, success, phases):
        ax.annotate(f'{yi:.0f}%',
                    xy=(xi, yi), xytext=(0, 8),
                    textcoords='offset points',
                    ha='center', va='bottom', fontsize=7.5, color=color)

# Draw threshold lines
ax.axhline(80, color='gray', linestyle='--', linewidth=0.8, alpha=0.7, label='80% threshold (typical)')
ax.axhline(95, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)

ax.set_xlabel('Curriculum Phase (sequential within each system)')
ax.set_ylabel('Best Success Rate (%)')
ax.set_title('Curriculum Learning Progression — Success Rate per Phase')
ax.set_ylim(60, 108)
ax.set_xlim(0.5, 8.5)
ax.set_xticks(range(1, 9))
ax.set_xticklabels([f'Phase {i}' for i in range(1, 9)])
ax.legend(loc='lower left')
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.0f}%'))

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig1_success_progression.pdf'))
plt.savefig(os.path.join(OUT_DIR, 'fig1_success_progression.png'))
plt.close()
print("  Saved fig1_success_progression.pdf")

# ──────────────────────────────────────────────
# FIG 2: Timesteps per phase (grouped bar chart)
# ──────────────────────────────────────────────

print("Generating Fig 2: Timesteps per phase...")

fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
titles = ['Original RL (Reference)', 'ChronosCar (Deployed)', 'New Car (Encoder + Friction)']

for ax, (sys_key, data), title in zip(axes, datasets, titles):
    if not data:
        ax.set_visible(False)
        continue
    phases = [PHASE_SHORT.get(d['phase'], d['phase']) for d in data]
    ts = [d['timesteps'] / 1e6 for d in data]   # convert to millions
    success = [d['best_success_rate'] * 100 for d in data]

    bars = ax.bar(phases, ts, color=COLORS[sys_key], alpha=0.8, edgecolor='white', linewidth=0.5)

    # Color bars by success rate (darker = lower success)
    for bar, sr in zip(bars, success):
        alpha = 0.5 + 0.5 * (sr / 100)
        bar.set_alpha(alpha)

    # Annotate bars with success rate
    for bar, sr, t in zip(bars, success, ts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{sr:.0f}%', ha='center', va='bottom', fontsize=8,
                fontweight='bold', color=COLORS[sys_key])

    total_M = sum(ts)
    ax.set_title(f'{title}\n(Total: {total_M:.1f}M steps)', fontsize=10)
    ax.set_xlabel('Phase')
    ax.set_ylabel('Training Steps (Millions)')
    ax.tick_params(axis='x', labelrotation=0)

plt.suptitle('Training Cost per Curriculum Phase\n(Annotations show best success rate at phase completion)',
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig2_timesteps_per_phase.pdf'))
plt.savefig(os.path.join(OUT_DIR, 'fig2_timesteps_per_phase.png'))
plt.close()
print("  Saved fig2_timesteps_per_phase.pdf")

# ──────────────────────────────────────────────
# FIG 3: Reward learning curves from progress.csv
# ──────────────────────────────────────────────

print("Generating Fig 3: Reward learning curves...")

# Map each system to its canonical ray_results directories
# Each tuple: (date_string, min_rows_threshold, label)
# We'll find the largest matching CSV for each date
LEARNING_CURVE_SOURCES = {
    'original_rl': [
        # curriculum_20260203_090023: phases by date Feb 3
        # Phase 4 (longest at ~634K steps with reward learning)
        ('2026-02-03_10-17-32tlbee6ch', 'Phase 4: Full Random Bay'),
        # Phase 6 (hardest)
        ('2026-02-03_23-05-40r_wh5zex', 'Phase 6: Random Obstacles (final)'),
    ],
    'chronoscar': [
        # curriculum_20260208_223402 - phase 5 was long (10.8M steps)
        ('2026-02-08_18-07-08mzv07dch', 'Phase 5: Neighbor Jitter'),
        # curriculum_20260209_230526 - phase 6
        ('2026-02-09_21-54-15c80kjb_s', 'Phase 6: Random Obstacles'),
    ],
    'newcar': [
        # curriculum_20260220_211152 - phase 4 (33M steps)
        ('2026-02-20_21-11-522uwqu9_1', 'Phase 4: Full Random Bay'),
        # curriculum_20260225_134317 - phase 5
        ('2026-02-25_15-47-0330j52t22', 'Phase 5: Tighten Tolerance'),
    ],
}

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

for ax, (sys_key, sources) in zip(axes, LEARNING_CURVE_SOURCES.items()):
    ax.set_title(f'{SYSTEM_LABELS[sys_key]}\nReward Learning Curves', fontsize=10)
    ax.set_xlabel('Training Steps (Millions)')
    ax.set_ylabel('Mean Episode Reward')
    ax.axhline(0, color='black', linewidth=0.5, alpha=0.3)

    any_plotted = False
    for i, (dir_suffix, phase_label) in enumerate(sources):
        csv_path = os.path.join(RAY_DIR, f'PPO_curriculum_parking_env_{dir_suffix}', 'progress.csv')
        if not os.path.exists(csv_path):
            # Try to find it by matching the prefix
            pattern = os.path.join(RAY_DIR, f'PPO_curriculum_parking_env_{dir_suffix.split("_")[0]}*', 'progress.csv')
            matches = glob.glob(pattern)
            if matches:
                csv_path = sorted(matches)[0]
            else:
                print(f"  Warning: could not find {dir_suffix}")
                continue

        try:
            data = load_progress_csv(csv_path)
            steps = np.array(data.get('timesteps_total', []))
            reward = np.array(data.get('episode_reward_mean', []))

            if len(steps) < 10:
                continue

            # Normalize steps to relative (start from 0 for this phase)
            steps = (steps - steps[0]) / 1e6

            reward_smooth = smooth(reward, window=max(5, len(reward)//30))
            linestyle = '-' if i == 0 else '--'
            alpha_raw = 0.2
            alpha_smooth = 0.9

            ax.plot(steps, reward, alpha=alpha_raw, color=COLORS[sys_key], linewidth=0.5)
            ax.plot(steps, reward_smooth, linestyle=linestyle, color=COLORS[sys_key],
                    linewidth=2, label=phase_label, alpha=alpha_smooth)
            any_plotted = True
        except Exception as e:
            print(f"  Warning: error reading {csv_path}: {e}")

    if not any_plotted:
        ax.text(0.5, 0.5, 'Data not available\n(run generate_thesis_plots.py\nfrom training machine)',
                ha='center', va='center', transform=ax.transAxes, fontsize=9,
                style='italic', color='gray')

    if any_plotted:
        ax.legend(loc='lower right', fontsize=8)

plt.suptitle('Training Reward Learning Curves\n(Shaded: raw; Solid: smoothed with window=30)',
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig3_reward_curves.pdf'))
plt.savefig(os.path.join(OUT_DIR, 'fig3_reward_curves.png'))
plt.close()
print("  Saved fig3_reward_curves.pdf")

# ──────────────────────────────────────────────
# FIG 4: Final system comparison summary
# ──────────────────────────────────────────────

print("Generating Fig 4: System comparison summary...")

fig, axes = plt.subplots(1, 2, figsize=(13, 6))

# Panel A: Success rate comparison (best final phase)
ax = axes[0]
systems = ['Original RL\n(Reference)', 'ChronosCar\n(Deployed)', 'New Car\n(Phase 7)']
final_success = [
    orig_data[-1]['best_success_rate'] * 100     if orig_data     else 0,
    chronos_data[-1]['best_success_rate'] * 100  if chronos_data  else 0,
    newcar_data[-1]['best_success_rate'] * 100   if newcar_data   else 0,
]
colors = [COLORS['original_rl'], COLORS['chronoscar'], COLORS['newcar']]
bars = ax.bar(systems, final_success, color=colors, alpha=0.85, edgecolor='white', linewidth=0.5,
              width=0.5)
for bar, val in zip(bars, final_success):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val:.0f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
ax.axhline(80, color='gray', linestyle='--', linewidth=1, label='80% threshold')
ax.set_title('Final Success Rate\n(Best phase at training completion)', fontsize=11)
ax.set_ylabel('Success Rate (%)')
ax.set_ylim(0, 105)
ax.legend()

# Panel B: Total training cost
ax = axes[1]
total_steps = [
    sum(d['timesteps'] for d in orig_data)     / 1e6 if orig_data    else 0,
    sum(d['timesteps'] for d in chronos_data)  / 1e6 if chronos_data else 0,
    sum(d['timesteps'] for d in newcar_data)   / 1e6 if newcar_data  else 0,
]
bars = ax.bar(systems, total_steps, color=colors, alpha=0.85, edgecolor='white', linewidth=0.5,
              width=0.5)
for bar, val in zip(bars, total_steps):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val:.1f}M', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_title('Total Training Steps\n(Sum across all curriculum phases)', fontsize=11)
ax.set_ylabel('Environment Steps (Millions)')
ax.set_ylim(0, max(total_steps) * 1.15 + 5)

plt.suptitle('Simulation Training Performance Summary', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig4_system_comparison.pdf'))
plt.savefig(os.path.join(OUT_DIR, 'fig4_system_comparison.png'))
plt.close()
print("  Saved fig4_system_comparison.pdf")

# ──────────────────────────────────────────────
# FIG 5: Phase-by-phase success and reward heatmap / table
# ──────────────────────────────────────────────

print("Generating Fig 5: Phase detail table chart...")

fig, axes = plt.subplots(3, 1, figsize=(12, 9))

for ax, (sys_key, data), title in zip(axes, datasets, titles):
    if not data:
        ax.set_visible(False)
        continue
    phases = [PHASE_SHORT.get(d['phase'], d['phase'][:6]) for d in data]
    success = [d['best_success_rate'] * 100 for d in data]
    rewards = [d['best_reward'] for d in data]
    ts_M = [d['timesteps'] / 1e6 for d in data]

    x = np.arange(len(phases))
    width = 0.35

    ax2 = ax.twinx()

    bars1 = ax.bar(x - width/2, success, width, label='Success Rate (%)',
                   color=COLORS[sys_key], alpha=0.85)
    bars2 = ax2.bar(x + width/2, rewards, width, label='Best Reward',
                    color=COLORS[sys_key], alpha=0.4, hatch='///')

    ax.set_ylabel('Success Rate (%)', color=COLORS[sys_key])
    ax2.set_ylabel('Best Reward', color='gray')
    ax.set_title(f'{title}', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(phases)
    ax.set_ylim(0, 115)
    ax2.set_ylim(0, max(rewards) * 1.15)

    # Annotate timesteps on top
    for xi, t in zip(x, ts_M):
        ax.text(xi, 108, f'{t:.1f}M', ha='center', va='center', fontsize=7.5, color='#555')

    # Legend
    lines1 = mpatches.Patch(color=COLORS[sys_key], alpha=0.85, label='Success Rate (%)')
    lines2 = mpatches.Patch(color=COLORS[sys_key], alpha=0.4, hatch='///', label='Best Reward')
    ax.legend(handles=[lines1, lines2], loc='lower right', fontsize=8)

axes[0].text(0.5, 1.12, '(Steps M shown above each group)', ha='center',
             transform=axes[0].transAxes, fontsize=8, style='italic', color='#666')

plt.suptitle('Per-Phase Training Results: Success Rate and Best Reward\n(Numbers above bars = training steps in Millions)',
             fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig5_phase_detail.pdf'))
plt.savefig(os.path.join(OUT_DIR, 'fig5_phase_detail.png'))
plt.close()
print("  Saved fig5_phase_detail.pdf")

# ──────────────────────────────────────────────
# LaTeX Table
# ──────────────────────────────────────────────

print("\nGenerating LaTeX table...")

table_path = os.path.join(OUT_DIR, 'thesis_training_table.tex')

PHASE_FULL = {
    'phase1_foundation':           'Foundation (fixed spawn/bay)',
    'phase2_random_spawn':         'Random spawn position',
    'phase3_random_bay_x':         'Random bay X position',
    'phase4a_random_bay_y_small':  'Random bay Y (small range)',
    'phase4_random_bay_full':      'Full random bay position',
    'phase4b_random_bay_jitter':   'Neighbor jitter only',
    'phase5_neighbor_jitter':      'Neighbor jitter + tolerance',
    'phase5_tight_tol':            'Tighten tolerance (2.7 cm)',
    'phase6_random_obstacles':     'Settling polish',
    'phase7_polish':               'Tighter settling thresholds',
    'phase8_yaw_precision':        'Yaw precision (4°)',
}

with open(table_path, 'w') as f:
    f.write("% Auto-generated by generate_thesis_plots.py\n")
    f.write("% Training results for all three systems\n\n")

    for sys_key, data, title in zip(
            ['original_rl', 'chronoscar', 'newcar'],
            [orig_data, chronos_data, newcar_data],
            ['Original RL (Reference)', 'ChronosCar (Deployed, 1/28 scale)', 'New Car (Encoder + Friction)']):
        if not data:
            continue

        f.write(f"% {title}\n")
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write(f"\\caption{{Curriculum training results: {title}}}\n")
        f.write(f"\\label{{tab:training_{sys_key}}}\n")
        f.write("\\begin{tabular}{lrrrr}\n")
        f.write("\\hline\n")
        f.write("\\textbf{Phase} & \\textbf{Steps (M)} & \\textbf{Success (\\%)} & \\textbf{Best Reward} & \\textbf{Cumul. Steps (M)} \\\\\n")
        f.write("\\hline\n")

        cumul = 0
        for d in data:
            ts = d['timesteps'] / 1e6
            cumul += ts
            phase_name = PHASE_FULL.get(d['phase'], d['phase'])
            sr = d['best_success_rate'] * 100
            reward = d['best_reward']
            f.write(f"{phase_name} & {ts:.2f} & {sr:.1f}\\% & {reward:.1f} & {cumul:.2f} \\\\\n")

        total_ts = sum(d['timesteps'] for d in data) / 1e6
        f.write("\\hline\n")
        f.write(f"\\textbf{{Total}} & \\textbf{{{total_ts:.2f}}} & & & \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n\n")

print(f"  Saved {table_path}")

# ──────────────────────────────────────────────
# Text summary
# ──────────────────────────────────────────────

summary_path = os.path.join(OUT_DIR, 'training_summary.txt')
with open(summary_path, 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("TRAINING RESULTS SUMMARY\n")
    f.write("Generated by generate_thesis_plots.py\n")
    f.write("=" * 70 + "\n\n")

    for sys_key, data, title in zip(
            ['original_rl', 'chronoscar', 'newcar'],
            [orig_data, chronos_data, newcar_data],
            ['Original RL (Reference)', 'ChronosCar (Deployed)', 'New Car']):
        if not data:
            continue
        f.write(f"{title}\n")
        f.write("-" * 50 + "\n")
        cumul = 0
        for d in data:
            ts = d['timesteps'] / 1e6
            cumul += ts
            sr = d['best_success_rate'] * 100
            f.write(f"  {PHASE_SHORT.get(d['phase'],d['phase']):5s}  {ts:7.2f} M steps  "
                    f"success={sr:5.1f}%  reward={d['best_reward']:7.1f}  cumul={cumul:.2f}M\n")
        total_ts = sum(d['timesteps'] for d in data) / 1e6
        final_sr = data[-1]['best_success_rate'] * 100
        f.write(f"  TOTAL: {total_ts:.2f}M steps | Final success: {final_sr:.1f}%\n\n")

    f.write("=" * 70 + "\n")
    f.write("Hardware result (ChronosCar, 2026-02-27)\n")
    f.write("-" * 50 + "\n")
    f.write("  along   = +0.027 m  (tol 0.055 m)  PASS\n")
    f.write("  lateral = -0.004 m  (tol 0.055 m)  PASS\n")
    f.write("  yaw_err = -2.5°     (tol 8.6°)     PASS\n")
    f.write("  velocity = 0.00 m/s (tol 0.05)     PASS\n")
    f.write("  Steps to park: 40 (~4 seconds)\n")

print(f"  Saved {summary_path}")

# ──────────────────────────────────────────────
# Done
# ──────────────────────────────────────────────

print("\n" + "=" * 60)
print(f"All figures saved to: {OUT_DIR}/")
print("Files generated:")
for fname in sorted(os.listdir(OUT_DIR)):
    fpath = os.path.join(OUT_DIR, fname)
    print(f"  {fname}  ({os.path.getsize(fpath)//1024}KB)")
print("=" * 60)
