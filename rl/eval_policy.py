"""Evaluate a trained RLlib PPO checkpoint on the parking environment.

Key goals of this script:
- Works even when the checkpoint was trained with GPU but the evaluation machine has *no* usable GPU.
- Avoids RLlib private APIs (no Algorithm._checkpoint_info) to stay compatible across Ray versions.
- Supports glob patterns for checkpoint paths (e.g. curriculum_*/phaseX/best_checkpoint).

Typical usage:
  python -m rl.eval_policy \
    --checkpoint "checkpoints/curriculum/curriculum_*/phase6_random_obstacles/best_checkpoint" \
    --num-episodes 100 --deterministic --save-trajectories

Notes
- If you trained with GPU but evaluation crashes with "Found 0 GPUs", run with --force-cpu
  (default is CPU unless you pass --use-gpu).
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import ray
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env


# ---------------------------
# Project imports (module-or-script friendly)
# ---------------------------
try:
    from rl.curriculum_env import create_env_for_rllib
except Exception:  # pragma: no cover
    from curriculum_env import create_env_for_rllib  # type: ignore


# ---------------------------
# Helpers: checkpoint resolution + config loading
# ---------------------------

def _gpu_available() -> bool:
    """Return True if torch reports a usable CUDA GPU."""
    try:
        import torch  # type: ignore
        return bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
    except Exception:
        # If torch is missing/misconfigured, assume no GPU.
        return False

def _resolve_checkpoint_path(pattern_or_path: str) -> str:
    """Resolve a possibly-globbed checkpoint path to a single path.

    If multiple matches exist, pick the most recently modified.
    """
    matches = sorted(glob.glob(pattern_or_path))
    if not matches:
        return pattern_or_path

    if len(matches) == 1:
        return matches[0]

    # Pick newest by mtime (best for curriculum_* patterns)
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
    """Convert AlgorithmConfig-like objects to plain dict when possible."""
    if maybe_cfg is None:
        return {}
    if isinstance(maybe_cfg, dict):
        return maybe_cfg
    if hasattr(maybe_cfg, "to_dict") and callable(getattr(maybe_cfg, "to_dict")):
        try:
            return maybe_cfg.to_dict()
        except Exception:
            pass
    # Last resort: try vars()
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

    # Some Tune runs store {"config": {...}}; others store the config dict directly.
    if isinstance(data, dict) and "config" in data and isinstance(data["config"], dict):
        return data["config"]
    if isinstance(data, dict):
        return data
    return {}


def _guess_state_file_from_checkpoint_dir(ckpt_dir: Path) -> Optional[Path]:
    """Try to locate the pickled algorithm state file inside a checkpoint directory."""
    # Newer RLlib: rllib_checkpoint.json points to algorithm_state.pkl
    for meta_name in ["rllib_checkpoint.json", "checkpoint.json"]:
        meta = ckpt_dir / meta_name
        if meta.exists():
            try:
                meta_data = json.loads(meta.read_text())
                # Common keys: "state_file", "checkpoint_file"
                for k in ["state_file", "checkpoint_file", "state_path", "file"]:
                    v = meta_data.get(k)
                    if isinstance(v, str):
                        cand = (ckpt_dir / v).resolve()
                        if cand.exists():
                            return cand
                # Sometimes nested under "data"
                if isinstance(meta_data.get("data"), dict):
                    v = meta_data["data"].get("state_file")
                    if isinstance(v, str):
                        cand = (ckpt_dir / v).resolve()
                        if cand.exists():
                            return cand
            except Exception:
                pass

    # Common default
    for name in ["algorithm_state.pkl", "algo_state.pkl", "state.pkl"]:
        cand = ckpt_dir / name
        if cand.exists():
            return cand

    # Older Tune format sometimes has a file named like "checkpoint-000123"
    candidates = sorted(ckpt_dir.glob("checkpoint-*"))
    for c in candidates:
        if c.is_file():
            return c

    return None


def _load_config_from_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """Load an RLlib config dict from either Tune params.* or checkpoint state."""
    ckpt = Path(checkpoint_path).expanduser()
    if ckpt.is_file():
        ckpt = ckpt.parent
    ckpt = ckpt.resolve()

    # 1) Prefer params.* in the trial directory
    params_file = _find_upwards(ckpt, ["params.pkl", "params.json"])
    if params_file:
        cfg = _load_config_from_params_file(params_file)
        if cfg:
            return cfg

    # 2) Fallback: read the checkpoint state file (pickled)
    state_file = _guess_state_file_from_checkpoint_dir(ckpt)
    if state_file and state_file.exists():
        try:
            with state_file.open("rb") as f:
                state = pickle.load(f)
            if isinstance(state, dict):
                # Common keys in different RLlib versions
                for key in ["config", "algorithm_config"]:
                    if key in state:
                        return _as_plain_dict(state[key])
        except Exception as e:
            print(f"[WARN] Failed to read checkpoint state file: {state_file} ({e})")

    return {}


def _patch_config_for_eval(cfg: Dict[str, Any], force_cpu: bool) -> Dict[str, Any]:
    """Make the config safe for evaluation on a single process (and optionally CPU-only)."""
    cfg = copy.deepcopy(cfg)

    # Use local evaluation (no rollout workers)
    cfg["num_workers"] = 0
    cfg["evaluation_num_workers"] = 0

    # Avoid any GPU requirement at restore time
    if force_cpu:
        cfg["num_gpus"] = 0
        cfg["num_gpus_per_worker"] = 0
        # Newer RLlib keys (safe to set even if ignored)
        cfg["num_gpus_per_learner"] = 0
        cfg["num_learners"] = 0
        cfg["num_gpus_per_env_runner"] = 0

    # Reduce noise
    cfg.setdefault("log_level", "WARN")
    return cfg


def _register_project_env(env_name: str) -> None:
    """Register custom env(s) that appear in your training config."""
    if env_name == "curriculum_parking_env":
        register_env("curriculum_parking_env", create_env_for_rllib)
    # Add other env names here if you have them.


def _restore_ppo(checkpoint_path: str, force_cpu: bool) -> PPO:
    """Restore PPO weights using a config that won't demand GPUs."""
    cfg = _load_config_from_checkpoint(checkpoint_path)
    if not cfg:
        raise RuntimeError(
            "Could not load RLlib config from checkpoint. "
            "Expected params.json/params.pkl near the checkpoint or an algorithm_state.pkl inside it."
        )

    env_name = cfg.get("env", "curriculum_parking_env")
    if isinstance(env_name, str):
        _register_project_env(env_name)

    cfg = _patch_config_for_eval(cfg, force_cpu=force_cpu)

    # Create algo with patched config, then restore weights.
    algo = PPO(config=cfg)
    algo.restore(checkpoint_path)
    return algo


# ---------------------------
# Env stepping helpers (gym vs gymnasium compatibility)
# ---------------------------
def _reset_env(env) -> Tuple[np.ndarray, Dict[str, Any]]:
    out = env.reset()
    if isinstance(out, tuple) and len(out) == 2:
        obs, info = out
    else:
        obs, info = out, {}
    return np.asarray(obs, dtype=np.float32), info


def _step_env(env, action) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
    out = env.step(action)
    if isinstance(out, tuple) and len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = bool(terminated) or bool(truncated)
    else:
        obs, reward, done, info = out
    return np.asarray(obs, dtype=np.float32), float(reward), bool(done), info


@dataclass
class EpisodeResult:
    ep: int
    reward: float
    length: int
    success: bool
    collision: bool
    timeout: bool


def evaluate(
    checkpoint: str,
    num_episodes: int,
    deterministic: bool,
    save_trajectories: bool,
    force_cpu: bool,
    out_dir: Optional[str],
    seed: int,
) -> Dict[str, Any]:
    checkpoint = _resolve_checkpoint_path(checkpoint)

    ray.init(ignore_reinit_error=True, include_dashboard=False)

    algo = _restore_ppo(checkpoint, force_cpu=force_cpu)

    # Build a standalone env for stepping (keeps evaluation logic simple).
    # We use the same env_config that was saved in the training config.
    cfg = _load_config_from_checkpoint(checkpoint)
    env_cfg = cfg.get("env_config", {})
    env = create_env_for_rllib(env_cfg)

    policy = algo.get_policy()
    if policy is None:
        raise RuntimeError("No default policy found in restored algorithm.")

    if save_trajectories:
        out_path = Path(out_dir or "eval_trajectories") / f"eval_{int(time.time())}"
        out_path.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Saving trajectories to: {out_path}")
    else:
        out_path = None

    results: List[EpisodeResult] = []

    rng = np.random.default_rng(seed)

    for ep in range(num_episodes):
        # Make episode deterministic across runs when seed is set.
        try:
            env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        except TypeError:
            pass

        obs, info = _reset_env(env)
        done = False
        ep_reward = 0.0
        ep_len = 0

        # Trajectory buffers
        traj = {
            "obs": [],
            "actions": [],
            "rewards": [],
            "infos": [],
        }

        while not done:
            # RLlib expects actions in the env's action space format
            action = algo.compute_single_action(obs, explore=not deterministic)

            next_obs, reward, done, step_info = _step_env(env, action)

            ep_reward += reward
            ep_len += 1

            if save_trajectories:
                traj["obs"].append(obs.copy())
                traj["actions"].append(np.array(action))
                traj["rewards"].append(reward)
                traj["infos"].append(step_info)

            obs = next_obs

            # Optional safety break if env doesn't terminate properly
            if ep_len > int(cfg.get("horizon", 1000)) + 50:
                step_info = dict(step_info or {})
                step_info["forced_timeout"] = True
                done = True
                break

        # Success flags: adapt to your env's info keys.
        info_final = step_info if isinstance(step_info, dict) else {}
        success = bool(info_final.get("success", False))
        collision = bool(info_final.get("collision", False))
        timeout = bool(info_final.get("timeout", False) or info_final.get("forced_timeout", False))

        results.append(
            EpisodeResult(ep=ep, reward=ep_reward, length=ep_len, success=success, collision=collision, timeout=timeout)
        )

        if save_trajectories and out_path is not None:
            np.savez_compressed(
                out_path / f"episode_{ep:04d}.npz",
                obs=np.asarray(traj["obs"], dtype=np.float32),
                actions=np.asarray(traj["actions"]),
                rewards=np.asarray(traj["rewards"], dtype=np.float32),
                # infos might contain objects; store as JSON strings per step
                infos=np.asarray([json.dumps(x, default=str) for x in traj["infos"]], dtype=object),
            )

        if (ep + 1) % max(1, num_episodes // 10) == 0:
            succ_rate = 100.0 * sum(r.success for r in results) / len(results)
            print(f"[INFO] Episode {ep+1}/{num_episodes} | success={succ_rate:.1f}%")

    # Summary
    n = len(results)
    success_rate = sum(r.success for r in results) / n if n else 0.0
    collision_rate = sum(r.collision for r in results) / n if n else 0.0
    timeout_rate = sum(r.timeout for r in results) / n if n else 0.0
    avg_len = sum(r.length for r in results) / n if n else 0.0
    avg_reward = sum(r.reward for r in results) / n if n else 0.0

    summary = {
        "checkpoint": checkpoint,
        "num_episodes": n,
        "success_rate": success_rate,
        "collision_rate": collision_rate,
        "timeout_rate": timeout_rate,
        "avg_length": avg_len,
        "avg_reward": avg_reward,
        "trajectories_dir": str(out_path) if out_path else None,
    }

    print("\n====================")
    print("EVALUATION SUMMARY")
    print("====================")
    print(f"Checkpoint:   {summary['checkpoint']}")
    print(f"Episodes:     {n}")
    print(f"Success:      {success_rate*100:.1f}%")
    print(f"Collisions:   {collision_rate*100:.1f}%")
    print(f"Timeouts:     {timeout_rate*100:.1f}%")
    print(f"Avg length:   {avg_len:.1f}")
    print(f"Avg reward:   {avg_reward:.2f}")
    if out_path:
        print(f"Trajectories: {out_path}")
    print("====================\n")

    algo.stop()
    ray.shutdown()
    try:
        env.close()
    except Exception:
        pass

    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path or glob pattern.")
    p.add_argument("--num-episodes", type=int, default=20)
    p.add_argument("--deterministic", action="store_true", help="Disable exploration.")
    p.add_argument("--save-trajectories", action="store_true", help="Save per-step trajectories to .npz files.")
    p.add_argument("--out-dir", type=str, default=None, help="Directory to save trajectories.")
    p.add_argument("--seed", type=int, default=0)

    # GPU flags
    g = p.add_mutually_exclusive_group()
    g.add_argument("--use-gpu", action="store_true", help="Try to evaluate on GPU.")
    g.add_argument("--force-cpu", action="store_true", help="Force CPU even if a GPU exists.")

    args = p.parse_args()

    # Default to CPU unless explicitly asked for GPU (keeps it robust on GPU-less machines)
    force_cpu = True
    if args.use_gpu and not args.force_cpu:
        force_cpu = False

    evaluate(
        checkpoint=args.checkpoint,
        num_episodes=args.num_episodes,
        deterministic=args.deterministic,
        save_trajectories=args.save_trajectories,
        force_cpu=force_cpu,
        out_dir=args.out_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
