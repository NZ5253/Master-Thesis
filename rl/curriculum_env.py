"""
Curriculum-aware environment factory for progressive RL training.

Manages phase-based environment configuration that gradually increases difficulty.
"""

import copy
import yaml
from typing import Dict, Any, Optional
from pathlib import Path

from rl.gym_parking_env import GymParkingEnv


class CurriculumManager:
    """
    Manages curriculum phases and environment configuration.

    Loads curriculum config and provides phase-specific environment configs.
    """

    def __init__(self, curriculum_config_path: str = "rl/curriculum_config.yaml"):
        """
        Initialize curriculum manager.

        Args:
            curriculum_config_path: Path to curriculum YAML config
        """
        self.config_path = Path(curriculum_config_path)

        # Load curriculum config
        with open(self.config_path, "r") as f:
            self.curriculum = yaml.safe_load(f)

        # Extract phases
        self.phases = self.curriculum["curriculum"]
        self.training_configs = self.curriculum["training_config"]
        self.progression = self.curriculum["progression"]

        # Get base config path (defaults to config_env.yaml)
        # Can be overridden in curriculum config with "base_config" key
        self.base_config_path = self.curriculum.get("base_config", "config_env.yaml")

        # Phase order
        # Prefer the order defined in the YAML (insertion order). This prevents
        # referencing phases that are not present in rl/curriculum_config.yaml.
        self.phase_order = list(self.phases.keys())

        self.current_phase_idx = 0

    def get_phase(self, phase_name: str) -> Dict[str, Any]:
        """Get phase configuration by name."""
        if phase_name not in self.phases:
            raise ValueError(f"Unknown phase: {phase_name}")
        return self.phases[phase_name]

    def get_current_phase_name(self) -> str:
        """Get current phase name."""
        return self.phase_order[self.current_phase_idx]

    def get_current_phase(self) -> Dict[str, Any]:
        """Get current phase configuration."""
        return self.get_phase(self.get_current_phase_name())

    def get_training_config(self, phase_name: str) -> Dict[str, Any]:
        """Get training hyperparameters for a phase."""
        if phase_name not in self.training_configs:
            raise ValueError(f"No training config for phase: {phase_name}")
        return self.training_configs[phase_name]

    def advance_phase(self) -> bool:
        """
        Advance to next phase.

        Returns:
            True if advanced, False if already at final phase
        """
        if self.current_phase_idx < len(self.phase_order) - 1:
            self.current_phase_idx += 1
            return True
        return False

    def is_final_phase(self) -> bool:
        """Check if current phase is the final one."""
        return self.current_phase_idx == len(self.phase_order) - 1

    def get_phase_env_config(self, phase_name: str, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge phase-specific config with base environment config.

        Args:
            phase_name: Name of curriculum phase
            base_config: Base environment config from config_env.yaml

        Returns:
            Merged config dict
        """
        phase = self.get_phase(phase_name)
        phase_env_config = phase.get("env_config", {})

        # Deep copy base config
        merged = copy.deepcopy(base_config)

        # Merge phase-specific settings
        # For parallel scenario, update the parallel section
        if "parallel" in phase_env_config:
            if "scenarios" in merged and "parallel" in merged["scenarios"]:
                # Update existing scenario config
                self._deep_update(merged["scenarios"]["parallel"], phase_env_config["parallel"])
            else:
                # Create scenario if it doesn't exist
                if "scenarios" not in merged:
                    merged["scenarios"] = {}
                merged["scenarios"]["parallel"] = phase_env_config["parallel"]

        return merged

    def _deep_update(self, target: Dict[str, Any], source: Dict[str, Any]):
        """Deep update a dictionary."""
        # Some subtrees must be replaced (not merged) to avoid leftover keys
        # like center_x_min/center_x_max accidentally keeping randomization on in Phase 1.
        REPLACE_SUBTREES = {
            "bay",
            "spawn_lane",
            "success",
            "obstacles",
            "neighbor",
            "random",
        }

        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                if key in REPLACE_SUBTREES:
                    # Replace subtree (avoids leftover keys keeping randomization on)
                    if key == "obstacles" and isinstance(target.get(key, None), dict):
                        # Preserve world bounds unless explicitly overridden in the phase.
                        old_world = copy.deepcopy(target[key].get("world")) if isinstance(target[key].get("world", None), dict) else None
                        target[key] = copy.deepcopy(value)
                        if old_world is not None and "world" not in target[key]:
                            target[key]["world"] = old_world
                    else:
                        target[key] = copy.deepcopy(value)
                else:
                    self._deep_update(target[key], value)
            else:
                target[key] = copy.deepcopy(value)

    def get_phase_summary(self, phase_name: str) -> str:
        """Get human-readable summary of a phase."""
        phase = self.get_phase(phase_name)
        train_cfg = self.get_training_config(phase_name)

        summary = f"""
Phase: {phase['name']}
Description: {phase['description']}
Target Timesteps: {phase['timesteps']:,}
Success Threshold: {phase['success_threshold']:.1%}

Training Config:
  - Learning Rate: {train_cfg['lr']}
  - Entropy Coeff: {train_cfg['entropy_coeff']}
  - Batch Size: {train_cfg['train_batch_size']}
  - SGD Iterations: {train_cfg['num_sgd_iter']}
  - Eval Interval: {train_cfg['eval_interval']}
"""
        return summary.strip()


class CurriculumEnv(GymParkingEnv):
    """
    Environment wrapper that supports curriculum learning.

    Can be reconfigured dynamically to change difficulty level.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        scenario: str = "parallel",
        curriculum_manager: Optional[CurriculumManager] = None,
        phase_name: Optional[str] = None,
    ):
        """
        Initialize curriculum-aware environment.

        Args:
            config: Base environment config
            scenario: Scenario name
            curriculum_manager: Optional curriculum manager instance
            phase_name: Optional specific phase to use
        """
        self.curriculum_manager = curriculum_manager
        self.phase_name = phase_name

        # If curriculum manager provided, merge phase config
        if curriculum_manager is not None and phase_name is not None:
            # Load base config if not provided
            if config is None:
                base_config_path = curriculum_manager.base_config_path
                with open(base_config_path, "r") as f:
                    config = yaml.safe_load(f)

            # Merge with phase-specific config
            config = curriculum_manager.get_phase_env_config(phase_name, config)

        # Initialize parent
        super().__init__(config=config, scenario=scenario)

    def set_phase(self, phase_name: str):
        """
        Reconfigure environment for a new curriculum phase.

        Args:
            phase_name: Name of new phase
        """
        if self.curriculum_manager is None:
            raise RuntimeError("Cannot set phase without curriculum_manager")

        self.phase_name = phase_name

        # Reload config from the curriculum's base config path
        base_config_path = self.curriculum_manager.base_config_path
        with open(base_config_path, "r") as f:
            base_config = yaml.safe_load(f)

        merged_config = self.curriculum_manager.get_phase_env_config(phase_name, base_config)

        # Recreate environment with new config
        self.__init__(
            config=merged_config,
            scenario=self.scenario,
            curriculum_manager=self.curriculum_manager,
            phase_name=phase_name,
        )


def make_curriculum_env(
    scenario: str = "parallel",
    phase_name: Optional[str] = None,
    curriculum_config_path: str = "rl/curriculum_config.yaml",
):
    """
    Factory function to create curriculum-aware environment.

    Args:
        scenario: Scenario name
        phase_name: Optional phase name (if None, uses phase1_foundation)
        curriculum_config_path: Path to curriculum config

    Returns:
        CurriculumEnv instance
    """
    curriculum_manager = CurriculumManager(curriculum_config_path)

    if phase_name is None:
        phase_name = curriculum_manager.phase_order[0]

    # Load base config from the curriculum's specified config file
    base_config_path = curriculum_manager.base_config_path
    with open(base_config_path, "r") as f:
        base_config = yaml.safe_load(f)

    return CurriculumEnv(
        config=base_config,
        scenario=scenario,
        curriculum_manager=curriculum_manager,
        phase_name=phase_name,
    )


def create_env_for_rllib(env_config: Dict[str, Any]):
    """
    Environment factory for RLlib.

    Args:
        env_config: Dict with 'scenario', 'phase_name', 'curriculum_config_path'

    Returns:
        CurriculumEnv instance
    """
    return make_curriculum_env(
        scenario=env_config.get("scenario", "parallel"),
        phase_name=env_config.get("phase_name", None),
        curriculum_config_path=env_config.get("curriculum_config_path", "rl/curriculum_config.yaml"),
    )


if __name__ == "__main__":
    # Test curriculum manager
    print("=" * 80)
    print("Testing Curriculum Manager")
    print("=" * 80)

    manager = CurriculumManager()

    print(f"\nTotal phases: {len(manager.phase_order)}")
    print(f"Phase order: {manager.phase_order}\n")

    # Test each phase
    for phase_name in manager.phase_order:
        print("-" * 80)
        print(manager.get_phase_summary(phase_name))
        print()

    # Test environment creation
    print("=" * 80)
    print("Testing Curriculum Environment")
    print("=" * 80)

    env = make_curriculum_env(scenario="parallel", phase_name="phase1_foundation")
    print(f"\nPhase 1 environment created")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    obs, info = env.reset(seed=42)
    print(f"\nReset successful")
    print(f"Goal: {info.get('goal_pose')}")
    print(f"Bay center: {info['bay_center']}")

    # Test phase advancement
    print("\n" + "=" * 80)
    print("Testing Phase Advancement")
    print("=" * 80)

    for i, phase_name in enumerate(manager.phase_order[:3]):  # Test first 3 phases
        print(f"\n[Phase {i+1}] {phase_name}")
        env = make_curriculum_env(scenario="parallel", phase_name=phase_name)
        obs, info = env.reset(seed=42)
        print(f"  Goal: {info.get('goal_pose')}")
        print(f"  Bay center: {info['bay_center']}")

    print("\n✓ All curriculum tests passed!")