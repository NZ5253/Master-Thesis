"""Visualize a trained RLlib PPO checkpoint with matplotlib rendering.

This version:
- Opens a new window for each episode and auto-closes it on completion.
- Does NOT use RLlib private APIs (no Algorithm._checkpoint_info).
- Can run on CPU even if the checkpoint was trained with GPU (default).
- Supports glob patterns for checkpoint paths.

Usage:
  python -m rl.visualize_checkpoint \
    --checkpoint "checkpoints/curriculum/curriculum_*/phase6_random_obstacles/best_checkpoint" \
    --num-episodes 5 --speed 1.0
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle

import ray
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env

# Project imports
try:
    from rl.curriculum_env import create_env_for_rllib, make_curriculum_env
except Exception:  # pragma: no cover
    from curriculum_env import create_env_for_rllib, make_curriculum_env  # type: ignore


# ---------------------------
# Shared helpers (checkpoint + config)
# ---------------------------
def _resolve_checkpoint_path(pattern_or_path: str) -> str:
    matches = sorted(glob.glob(pattern_or_path))
    if not matches:
        return pattern_or_path
    if len(matches) == 1:
        return matches[0]
    matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    chosen = matches[0]
    print(f"[INFO] Multiple checkpoints matched. Using newest: {chosen}")
    return chosen


def _find_upwards(start: Path, names: List[str], max_levels: int = 8) -> Optional[Path]:
    p = start.resolve()
    if p.is_file():
        p = p.parent
    for _ in range(max_levels):
        for n in names:
            cand = p / n
            if cand.exists():
                return cand
        if p.parent == p:
            break
        p = p.parent
    return None


def _as_plain_dict(maybe_cfg: Any) -> Dict[str, Any]:
    if maybe_cfg is None:
        return {}
    if isinstance(maybe_cfg, dict):
        return maybe_cfg
    if hasattr(maybe_cfg, "to_dict") and callable(getattr(maybe_cfg, "to_dict")):
        try:
            return maybe_cfg.to_dict()
        except Exception:
            pass
    try:
        return dict(maybe_cfg)
    except Exception:
        try:
            return vars(maybe_cfg)
        except Exception:
            return {}


def _load_config_from_params_file(params_path: Path) -> Dict[str, Any]:
    if params_path.suffix == ".json":
        data = json.loads(params_path.read_text())
    else:
        with params_path.open("rb") as f:
            data = pickle.load(f)
    if isinstance(data, dict) and "config" in data and isinstance(data["config"], dict):
        return data["config"]
    if isinstance(data, dict):
        return data
    return {}


def _guess_state_file_from_checkpoint_dir(ckpt_dir: Path) -> Optional[Path]:
    for meta_name in ["rllib_checkpoint.json", "checkpoint.json"]:
        meta = ckpt_dir / meta_name
        if meta.exists():
            try:
                meta_data = json.loads(meta.read_text())
                for k in ["state_file", "checkpoint_file", "state_path", "file"]:
                    v = meta_data.get(k)
                    if isinstance(v, str):
                        cand = (ckpt_dir / v).resolve()
                        if cand.exists():
                            return cand
                if isinstance(meta_data.get("data"), dict):
                    v = meta_data["data"].get("state_file")
                    if isinstance(v, str):
                        cand = (ckpt_dir / v).resolve()
                        if cand.exists():
                            return cand
            except Exception:
                pass

    for name in ["algorithm_state.pkl", "algo_state.pkl", "state.pkl"]:
        cand = ckpt_dir / name
        if cand.exists():
            return cand

    candidates = sorted(ckpt_dir.glob("checkpoint-*"))
    for c in candidates:
        if c.is_file():
            return c
    return None


def _load_config_from_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    ckpt = Path(checkpoint_path).expanduser()
    if ckpt.is_file():
        ckpt = ckpt.parent
    ckpt = ckpt.resolve()

    params_file = _find_upwards(ckpt, ["params.pkl", "params.json"])
    if params_file:
        cfg = _load_config_from_params_file(params_file)
        if cfg:
            return cfg

    state_file = _guess_state_file_from_checkpoint_dir(ckpt)
    if state_file and state_file.exists():
        try:
            with state_file.open("rb") as f:
                state = pickle.load(f)
            if isinstance(state, dict):
                for key in ["config", "algorithm_config"]:
                    if key in state:
                        return _as_plain_dict(state[key])
        except Exception as e:
            print(f"[WARN] Failed to read checkpoint state file: {state_file} ({e})")

    return {}


def _patch_config_for_viz(cfg: Dict[str, Any], force_cpu: bool) -> Dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    cfg["num_workers"] = 0
    cfg["evaluation_num_workers"] = 0
    if force_cpu:
        cfg["num_gpus"] = 0
        cfg["num_gpus_per_worker"] = 0
        cfg["num_gpus_per_learner"] = 0
        cfg["num_learners"] = 0
        cfg["num_gpus_per_env_runner"] = 0
    cfg.setdefault("log_level", "ERROR")
    return cfg


def _register_project_env(env_name: str) -> None:
    if env_name == "curriculum_parking_env":
        register_env("curriculum_parking_env", create_env_for_rllib)


def _restore_ppo(checkpoint_path: str, force_cpu: bool) -> PPO:
    cfg = _load_config_from_checkpoint(checkpoint_path)
    if not cfg:
        raise RuntimeError(
            "Could not load RLlib config from checkpoint. "
            "Expected params.json/params.pkl near the checkpoint or an algorithm_state.pkl inside it."
        )

    env_name = cfg.get("env", "curriculum_parking_env")
    if isinstance(env_name, str):
        _register_project_env(env_name)

    cfg = _patch_config_for_viz(cfg, force_cpu=force_cpu)

    algo = PPO(config=cfg)
    algo.restore(checkpoint_path)
    return algo


# ---------------------------
# Rendering helpers
# ---------------------------
def _car_polygon(x: float, y: float, yaw: float, length: float, width: float) -> np.ndarray:
    c = np.array([x, y], dtype=float)
    hl = length / 2.0
    hw = width / 2.0
    corners = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]], dtype=float)
    R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]], dtype=float)
    return (corners @ R.T) + c


def _unwrap_env(env):
    """Try to unwrap Gym/Curriculum envs to the underlying ParkingEnv."""
    # Common in Gym wrappers: `.env` points to the underlying environment.
    if hasattr(env, "env"):
        try:
            return env.env
        except Exception:
            pass
    # Gymnasium style
    if hasattr(env, "unwrapped"):
        try:
            return env.unwrapped
        except Exception:
            pass
    return env


def _infer_world_bounds(env) -> tuple[float, float, float, float]:
    """Return (xmin, xmax, ymin, ymax) with reasonable fallbacks."""
    candidates = [env, getattr(env, "obstacles", None)]
    for c in candidates:
        if c is None:
            continue
        if all(hasattr(c, k) for k in ("x_min", "x_max", "y_min", "y_max")):
            return float(c.x_min), float(c.x_max), float(c.y_min), float(c.y_max)
        # ObstacleManager keeps world_cfg
        if hasattr(c, "world_cfg") and isinstance(getattr(c, "world_cfg"), dict):
            wc = c.world_cfg
            if all(k in wc for k in ("x_min", "x_max", "y_min", "y_max")):
                return float(wc["x_min"]), float(wc["x_max"]), float(wc["y_min"]), float(wc["y_max"])
        if hasattr(c, "world_size"):
            ws = float(getattr(c, "world_size"))
            return -ws / 2.0, ws / 2.0, -ws / 2.0, ws / 2.0
    # Fallback: your typical 4x4m world centered at 0
    return -2.2, 2.2, -2.2, 2.2


def _obb_corners(cx: float, cy: float, w: float, h: float, theta: float):
    """Return 4 corners (Nx2) of an oriented box."""
    import numpy as np

    hw, hh = w / 2.0, h / 2.0
    pts = np.array([[-hw, -hh], [-hw, hh], [hw, hh], [hw, -hh]], dtype=float)
    c, s = float(np.cos(theta)), float(np.sin(theta))
    R = np.array([[c, -s], [s, c]], dtype=float)
    pts = (pts @ R.T) + np.array([cx, cy], dtype=float)
    return pts


def _car_corners_rear_axle(xr: float, yr: float, yaw: float, length: float, width: float):
    """Vehicle rectangle corners assuming (xr,yr) is the rear axle reference."""
    import numpy as np

    hw = width / 2.0
    pts = np.array([[0.0, -hw], [0.0, hw], [length, hw], [length, -hw]], dtype=float)
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    R = np.array([[c, -s], [s, c]], dtype=float)
    pts = (pts @ R.T) + np.array([xr, yr], dtype=float)
    return pts


def _draw_world(ax, env):
    """Draw world, obstacles, ego, and goal for this repo's ParkingEnv."""
    import matplotlib.patches as patches
    import numpy as np

    pe = _unwrap_env(env)

    xmin, xmax, ymin, ymax = _infer_world_bounds(pe)
    ax.clear()  # Ensure clean slate
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)

    # Access obstacle manager (ObstacleManager)
    om = getattr(pe, "obstacles", None) or getattr(env, "obstacle_manager", None)

    # Draw obstacles (including curbs and parked cars) if available
    if om is not None and hasattr(om, "obstacles"):
        for obs in getattr(om, "obstacles", []):
            try:
                cx, cy = float(obs["x"]), float(obs["y"])
                w, h = float(obs["w"]), float(obs["h"])
                theta = float(obs.get("theta", 0.0))
            except Exception:
                continue
            corners = _obb_corners(cx, cy, w, h, theta)
            poly = patches.Polygon(corners, closed=True, fill=False, linewidth=1.0, edgecolor="black")
            ax.add_patch(poly)

    # Vehicle geometry
    vehicle_params = getattr(pe, "vehicle_params", {}) or {}
    length = float(vehicle_params.get("length", 1.0))
    width = float(vehicle_params.get("width", 0.5))

    # Draw ego vehicle
    state = getattr(pe, "state", None)
    if state is not None:
        try:
            xr, yr, yaw = float(state[0]), float(state[1]), float(state[2])
        except Exception:
            xr, yr, yaw = 0.0, 0.0, 0.0

        if om is not None and hasattr(om, "get_car_corners"):
            try:
                corners = om.get_car_corners(np.asarray(state, dtype=float))
            except Exception:
                corners = _car_corners_rear_axle(xr, yr, yaw, length, width)
        else:
            corners = _car_corners_rear_axle(xr, yr, yaw, length, width)

        ego_poly = patches.Polygon(corners, closed=True, fill=False, linewidth=2.0, edgecolor="blue")
        ax.add_patch(ego_poly)
        
        # Draw heading arrow
        ax.arrow(xr, yr, 0.4*np.cos(yaw), 0.4*np.sin(yaw), head_width=0.1, head_length=0.1, fc='blue', ec='blue')

    # Draw goal pose (dashed)
    goal = getattr(pe, "goal", None)
    if goal is not None:
        try:
            gx, gy, gyaw = float(goal[0]), float(goal[1]), float(goal[2])
            goal_state = np.array([gx, gy, gyaw, 0.0], dtype=float)
            if om is not None and hasattr(om, "get_car_corners"):
                gcorners = om.get_car_corners(goal_state)
            else:
                gcorners = _car_corners_rear_axle(gx, gy, gyaw, length, width)
            goal_poly = patches.Polygon(gcorners, closed=True, fill=False, linewidth=1.5, linestyle="--", edgecolor="green")
            ax.add_patch(goal_poly)
        except Exception:
            pass

def visualize_checkpoint(checkpoint: str, num_episodes: int, speed: float, deterministic: bool, force_cpu: bool) -> None:
    checkpoint = _resolve_checkpoint_path(checkpoint)

    # Force CPU before Ray init if requested (prevents NVML/CUDA init errors)
    if force_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    ray.init(ignore_reinit_error=True, include_dashboard=False)
    algo = _restore_ppo(checkpoint, force_cpu=force_cpu)

    cfg = _load_config_from_checkpoint(checkpoint)
    env_cfg = cfg.get("env_config", {})
    env = create_env_for_rllib(env_cfg)

    # Interactive mode for plotting
    plt.ion()

    for ep in range(num_episodes):
        # Create a NEW figure for each episode
        fig, ax = plt.subplots(figsize=(7, 7))
        fig.canvas.manager.set_window_title(f"Episode {ep+1}/{num_episodes}")

        obs_out = env.reset()
        obs = obs_out[0] if isinstance(obs_out, tuple) else obs_out

        done = False
        steps = 0
        total_reward = 0.0

        while not done:
            # 1. Update Visualization
            _draw_world(ax, env)
            ax.set_title(f"Episode {ep+1} | Step {steps} | Reward: {total_reward:.1f}")
            
            # Draw/flush events - critical for animation
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            
            # Pause to control speed
            plt.pause(0.02 / max(0.1, speed))

            # 2. Step Agent
            action = algo.compute_single_action(obs, explore=not deterministic)
            step_out = env.step(action)
            
            if isinstance(step_out, tuple) and len(step_out) == 5:
                obs, reward, terminated, truncated, info = step_out
                done = bool(terminated) or bool(truncated)
            else:
                obs, reward, done, info = step_out

            total_reward += reward
            steps += 1

            if steps > int(cfg.get("horizon", 1000)) + 50:
                done = True

        print(f"[INFO] Episode {ep+1} finished. Steps: {steps}, Reward: {total_reward:.2f}")
        
        # Brief pause at end of episode to see result
        plt.pause(0.5)
        # Close the window automatically
        plt.close(fig)

    plt.ioff()
    algo.stop()
    ray.shutdown()
    try:
        env.close()
    except Exception:
        pass


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--num-episodes", type=int, default=5)
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--deterministic", action="store_true")

    g = p.add_mutually_exclusive_group()
    g.add_argument("--use-gpu", action="store_true", help="Try to run inference on GPU.")
    g.add_argument("--force-cpu", action="store_true", help="Force CPU (default).")

    args = p.parse_args()
    
    # Default to CPU unless explicitly asked for GPU
    force_cpu = True
    if args.use_gpu and not args.force_cpu:
        force_cpu = False

    visualize_checkpoint(
        checkpoint=args.checkpoint,
        num_episodes=args.num_episodes,
        speed=args.speed,
        deterministic=args.deterministic,
        force_cpu=force_cpu,
    )


if __name__ == "__main__":
    main()