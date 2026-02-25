"""
Gymnasium wrapper for ParkingEnv to enable RL training with RLlib.

This wrapper makes the existing ParkingEnv compatible with Gymnasium API
while preserving all the MPC environment logic and stability.
"""

import gymnasium as gym
import numpy as np
import yaml
from typing import Dict, Any, Tuple, Optional

from env.parking_env import ParkingEnv
from env.reward_parallel_parking import compute_reward_wrapper_parallel


class GymParkingEnv(gym.Env):
    """
    Gymnasium-compatible wrapper for parallel parking environment.

    Observation Space: Box(7,) containing:
        [along, lateral, yaw_err, v, dist_front, dist_left, dist_right]

    Action Space: Box(2,) containing:
        [steer, accel] both in range [-1, 1]
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, config: Optional[Dict[str, Any]] = None, scenario: str = "parallel"):
        """
        Initialize Gymnasium parking environment.

        Args:
            config: Environment configuration dict (from config_env.yaml)
            scenario: Scenario name ('parallel' or 'perpendicular')
        """
        super().__init__()

        # RLlib may accidentally pass an EnvContext/dict instead of a scenario string.
        if not isinstance(scenario, str):
            if hasattr(scenario, 'get'):
                scenario = scenario.get('scenario', 'parallel')
            else:
                scenario = 'parallel'

        self.scenario = scenario

        # Load config if not provided
        if config is None:
            with open("config_env.yaml", "r") as f:
                config = yaml.safe_load(f)

        # Always flatten scenario config (whether loaded or passed in)
        cfg_full = dict(config)
        if "scenarios" in cfg_full and self.scenario in cfg_full["scenarios"]:
            scenario_cfg = cfg_full["scenarios"][self.scenario]
            for key, val in scenario_cfg.items():
                cfg_full[key] = val

        self.config = cfg_full

        # Create underlying parking environment
        self.env = ParkingEnv(cfg_full)

        # Define observation space: 7D continuous (bay-frame)
        # [along, lateral, yaw_err, v, dist_front, dist_left, dist_right]
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(7,),
            dtype=np.float32
        )

        # Define action space: 2D continuous [steer, accel]
        # Normalized to [-1, 1], will be scaled to actual limits
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )

        # Get vehicle limits for action scaling
        vehicle_params = config.get("vehicle", {})
        base_max_steer = float(vehicle_params.get("max_steer", 0.523))  # ~30 degrees
        base_max_acc = float(vehicle_params.get("max_acc", 1.0))

        # Optional per-phase action scaling (lets early curriculum learn without slamming full-lock + full accel)
        # Can be provided under the scenario config, e.g.
        #   action_scale: {steer: 0.7, accel: 0.6}
        act_scale = cfg_full.get("action_scale", {}) or {}
        self.steer_scale = float(act_scale.get("steer", 1.0))
        self.accel_scale = float(act_scale.get("accel", 1.0))

        self.max_steer = base_max_steer * self.steer_scale
        self.max_acc = base_max_acc * self.accel_scale

        # Anti-jitter: penalize rapid action changes (helps kill tiny oscillations).
        reward_cfg = cfg_full.get("reward", {})
        # Smoothness penalty on action deltas (in scaled units). Default is small; override per curriculum phase.
        self.action_smooth_w = float(reward_cfg.get("action_smooth_w", 0.02))
        # Avoid double-counting smoothness if reward includes smoothness terms (smooth_w_far/smooth_w_near).
        if "action_smooth_w" not in reward_cfg and ("smooth_w_far" in reward_cfg or "smooth_w_near" in reward_cfg):
            self.action_smooth_w = 0.0
        self._prev_action_scaled = None

        # Early termination when "settled" - car stops moving when parked well
        self.early_termination_enabled = bool(reward_cfg.get("early_termination_enabled", True))
        self.settled_steps_required = int(reward_cfg.get("settled_steps_required", 5))  # N steps to confirm settled
        self.settling_v_threshold = float(reward_cfg.get("settling_v_threshold", 0.015))
        self.settling_steer_threshold = float(reward_cfg.get("settling_steer_threshold", 0.08))
        self._settled_counter = 0

        # Episode tracking
        self.episode_step = 0
        self.episode_reward = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Default randomization behavior comes from phase/scenario config (curriculum needs this).
        bay_cfg = self.config.get("parking", {}).get("bay", {})
        spawn_cfg = self.config.get("spawn_lane", {})

        randomize_bay = bool(bay_cfg.get("randomize", True))
        randomize_spawn = bool(spawn_cfg.get("randomize", True))

        # Allow explicit overrides via options
        if options:
            if "randomize" in options:
                randomize_bay = bool(options["randomize"])
                randomize_spawn = bool(options["randomize"])
            if "randomize_bay" in options:
                randomize_bay = bool(options["randomize_bay"])
            if "randomize_spawn" in options:
                randomize_spawn = bool(options["randomize_spawn"])

        obs, info = self.env.reset(
            seed=seed,
            randomize_bay=randomize_bay,
            randomize_spawn=randomize_spawn,
        )

        self.prev_action = np.zeros(2, dtype=np.float32)
        self.prev_state = self.env.state.copy()
        self.episode_step = 0
        self._last_info = info
        self.episode_reward = 0.0
        self._prev_action_scaled = None
        self._settled_counter = 0

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Action [steer, accel] in range [-1, 1]

        Returns:
            observation: Next observation
            reward: Step reward
            terminated: Whether episode ended (success or failure)
            truncated: Whether episode was truncated (max steps)
            info: Additional info dict
        """
        self.episode_step += 1

        # Scale actions from [-1, 1] to actual limits
        steer = float(np.clip(action[0], -1.0, 1.0)) * self.max_steer
        accel = float(np.clip(action[1], -1.0, 1.0)) * self.max_acc
        scaled_action = np.array([steer, accel], dtype=float)

        # Smoothness penalty on action deltas (scaled units)
        if self._prev_action_scaled is None or self.action_smooth_w <= 0.0:
            d_act = np.zeros(2, dtype=float)
            smooth_penalty = 0.0
            prev_action_scaled = None
        else:
            d_act = scaled_action - self._prev_action_scaled
            smooth_penalty = -self.action_smooth_w * float(d_act[0]**2 + 0.5 * d_act[1]**2)
            prev_action_scaled = self._prev_action_scaled

        # Store previous state for reward computation
        prev_state = self.env.state.copy()

        # Execute step in underlying environment
        obs, env_reward, done, env_info = self.env.step(scaled_action)

        # Extract termination string from env (single source of truth)
        termination = env_info.get("termination", "unknown")

        # Compute RL-specific reward using sophisticated parallel parking reward
        # Obstacle proximity signal (cheap shaping). Uses the same distance_features already in the observation.
        dist_front, dist_left, dist_right = self.env.obstacles.distance_features(self.env.state)
        min_obstacle_dist = float(min(dist_front, dist_left, dist_right))

        reward, reward_info = compute_reward_wrapper_parallel(
            prev_state,
            self.env.state,
            self.env.goal,
            termination,
            self.env.bay_center,
            self.config,
            action_scaled=scaled_action,
            prev_action_scaled=prev_action_scaled,
            min_obstacle_dist=min_obstacle_dist,
        )

        # Apply anti-jitter penalty
        reward = float(reward + smooth_penalty)
        reward_info["smooth_penalty"] = float(smooth_penalty)
        reward_info["d_steer"] = float(d_act[0])
        reward_info["d_accel"] = float(d_act[1])

        # Update episode reward
        self.episode_reward += reward

        # Gymnasium uses separate terminated and truncated flags
        terminated = termination in ["success", "collision"]
        truncated = termination == "max_steps"

        # EARLY TERMINATION: If car is "settled" for N steps, end episode as success
        # This prevents unnecessary oscillations after the car is already parked
        if self.early_termination_enabled and not terminated and not truncated:
            is_settled = reward_info.get("is_settled", False)
            if is_settled:
                self._settled_counter += 1
                if self._settled_counter >= self.settled_steps_required:
                    # Trigger early success!
                    terminated = True
                    termination = "success"
                    env_info["termination"] = "success"
                    env_info["success"] = True
                    env_info["early_termination"] = True
                    # Add success reward (already computed but let's ensure it's there)
                    success_reward = float(self.config.get("reward", {}).get("success_reward", 200.0))
                    reward += success_reward
                    reward_info["early_success_bonus"] = success_reward
            else:
                self._settled_counter = 0  # Reset if not settled

        self._prev_action_scaled = scaled_action
        self.prev_state = self.env.state.copy()

        # Combine info dicts
        info = {
            **env_info,
            **reward_info,
            "episode_step": self.episode_step,
            "episode_reward": self.episode_reward,
            "action_raw": action.copy(),
            "action_scaled": scaled_action.copy(),
        }

        return obs.astype(np.float32), float(reward), terminated, truncated, info

    def render(self):
        """Render environment (not implemented yet)."""
        pass

    def close(self):
        """Clean up resources."""
        pass


def make_parking_env(scenario: str = "parallel", config: Optional[Dict] = None):
    """Factory function to create the parking environment.

    This is used by RLlib during training/evaluation and by standalone scripts.
    The `scenario` argument should be a string (e.g., 'parallel'). If an RLlib
    EnvContext/dict gets passed by mistake, we fall back to reading its 'scenario' key.
    """
    if not isinstance(scenario, str):
        if hasattr(scenario, 'get'):
            scenario = scenario.get('scenario', 'parallel')
        else:
            scenario = 'parallel'
    return GymParkingEnv(config=config, scenario=scenario)