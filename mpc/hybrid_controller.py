"""
Hybrid TEB+MPC Controller

This module implements the hybrid controller that combines TEB planning and MPC tracking
to eliminate oscillations in parking maneuvers.

Architecture:
1. TEB plans ONCE at the start -> Creates committed reference trajectory
2. MPC tracks the reference -> No re-planning oscillations
3. Smooth transition to final convergence

Usage:
    controller = HybridController(config_path="mpc/config_mpc.yaml",
                                  env_cfg=env_cfg, dt=0.1)

    # First call plans trajectory
    control = controller.get_control(state, goal, obstacles, profile="parallel")

    # Subsequent calls track the reference
    control = controller.get_control(state, goal, obstacles, profile="parallel")
"""

import numpy as np
from typing import Optional, List
from dataclasses import dataclass

from mpc.teb_mpc import TEBMPC, VehicleState, ParkingGoal, Obstacle, MPCSolution
from mpc.reference_trajectory import ReferenceTrajectory


@dataclass
class HybridControllerState:
    """Internal state of the hybrid controller."""
    mode: str  # "planning", "tracking", "final_convergence", "goal_reached"
    reference: Optional[ReferenceTrajectory]
    current_step: int
    planning_count: int
    tracking_count: int

    # NEW: time left (seconds) to stay on current reference step (dt_array-driven)
    ref_step_time_left: float = 0.0



class HybridController:
    """
    Hybrid TEB+MPC controller for oscillation-free parking.

    This controller combines TEB planning (once) with MPC tracking to eliminate
    the oscillations caused by receding horizon re-planning.

    Key Features:
    - Plans reference trajectory once at start using TEB
    - Tracks reference smoothly using MPC (no re-planning)
    - Graceful transition to final convergence
    - Handles goal-reached detection

    Attributes:
        mpc: Underlying TEBMPC solver
        state: Current controller state
        final_convergence_threshold: Distance to switch from tracking to final convergence
        goal_reached_tolerance: Tolerance for goal-reached detection
    """

    def __init__(self, config_path: str, env_cfg: dict, dt: float):
        """
        Initialize hybrid controller.

        Args:
            config_path: Path to MPC config YAML
            env_cfg: Environment configuration dict
            dt: Control timestep (seconds)
        """
        self.env_cfg = env_cfg
        self.mpc = TEBMPC(config_path=config_path, env_cfg=env_cfg, dt=dt)

        # Internal state
        self.state = HybridControllerState(
            mode="planning",
            reference=None,
            current_step=0,
            planning_count=0,
            tracking_count=0
        )

        # Configuration
        self.final_convergence_threshold = 0.15  # Switch to final convergence at 15cm
        # FIXED: Make tolerances STRICTER than environment to prevent premature stopping
        # Environment: pos<5cm, yaw<6°, v<6cm/s
        self.goal_reached_pos_tol = 0.04         # 4cm CAR CENTER (stricter than env's 5cm)
        self.goal_reached_yaw_tol = 0.10         # 5.7° yaw tolerance (matches environment)
        self.goal_reached_vel_tol = 0.02         # 2cm/s velocity (stricter than env's 6cm/s)
        self.dt = float(dt)

        # Diagnostics
        self.last_info = {}

    def reset(self):
        """Reset controller state (call before new episode)."""
        self.mpc.reset_warm_start()
        self.state = HybridControllerState(
            mode="planning",
            reference=None,
            current_step=0,
            planning_count=0,
            tracking_count=0,
            ref_step_time_left = 0.0
        )
        self.last_info = {}

    def get_control(self, state: VehicleState, goal: ParkingGoal,
                   obstacles: List[Obstacle], profile: str = "parallel") -> np.ndarray:
        """
        Get control command for current state.

        This is the main interface method. On first call, it plans a reference
        trajectory using TEB. On subsequent calls, it tracks the reference using MPC.

        Args:
            state: Current vehicle state
            goal: Parking goal
            obstacles: List of obstacles
            profile: Parking profile ("parallel" or "perpendicular")

        Returns:
            Control vector [steering, acceleration]
        """
        # Mode 1: Planning (first call or after reset)
        if self.state.mode == "planning":
            return self._plan_and_execute(state, goal, obstacles, profile)

        # Mode 2: Tracking reference
        elif self.state.mode == "tracking":
            return self._track_reference(state, goal, obstacles, profile)

        # Mode 3: Final convergence (after reference ends)
        elif self.state.mode == "final_convergence":
            return self._final_convergence(state, goal, obstacles, profile)

        # Mode 4: Goal reached - BRAKE to fully stop
        elif self.state.mode == "goal_reached":
            # Apply braking to bring car to complete stop
            max_acc = float(self.env_cfg.get("vehicle", {}).get("max_acc", 1.0))
            desired_accel = -float(state.v) / max(self.dt, 1e-6)
            desired_accel = float(np.clip(desired_accel, -max_acc, max_acc))
            return np.array([0.0, desired_accel], dtype=float)

        else:
            raise ValueError(f"Unknown controller mode: {self.state.mode}")

    def _plan_and_execute(self, state: VehicleState, goal: ParkingGoal,
                         obstacles: List[Obstacle], profile: str) -> np.ndarray:
        """Plan reference trajectory and execute first control."""
        print(f"\n[HYBRID CONTROLLER] Planning reference trajectory...")

        # Plan using TEB
        reference = self.mpc.plan_trajectory(
            state=state,
            goal=goal,
            obstacles=obstacles,
            profile=profile
        )

        if not reference.success:
            print(f"[HYBRID CONTROLLER] ❌ Planning failed, braking to stop")
            self.last_info = {"status": "planning_failed"}
            max_acc = float(self.env_cfg.get("vehicle", {}).get("max_acc", 1.0))
            desired_accel = -float(state.v) / max(self.dt, 1e-6)
            desired_accel = float(np.clip(desired_accel, -max_acc, max_acc))
            return np.array([0.0, desired_accel], dtype=float)

        # Store reference and transition to tracking
        self.state.reference = reference
        self.state.current_step = 0
        self.state.planning_count += 1
        self.state.mode = "tracking"

        # NEW: initialize time-left for ref step 0 using reference.dt_array
        if reference.dt_array is not None and len(reference.dt_array) > 0:
            self.state.ref_step_time_left = float(reference.dt_array[0])
        else:
            self.state.ref_step_time_left = self.dt

        print(f"[HYBRID CONTROLLER] ✓ Planned {reference.n_steps} steps ({reference.total_time:.2f}s)")

        # Execute first control from reference
        control = reference.get_control_at_step(0)
        self.mpc._last_executed_control = control.copy()

        self.last_info = {
            "status": "planned",
            "reference_steps": reference.n_steps,
            "reference_time": reference.total_time,
            "mode": "tracking"
        }

        return control

    def _track_reference(self, state: VehicleState, goal: ParkingGoal,
                        obstacles: List[Obstacle], profile: str) -> np.ndarray:
        """Track reference trajectory using MPC."""

        # Check if we've exhausted the reference
        if self.state.current_step >= self.state.reference.n_steps:
            # Check if close enough to goal
            pos_err = np.sqrt((state.x - goal.x)**2 + (state.y - goal.y)**2)
            yaw_err = abs((state.yaw - goal.yaw + np.pi) % (2 * np.pi) - np.pi)

            if pos_err < self.goal_reached_pos_tol and \
               yaw_err < self.goal_reached_yaw_tol and \
               abs(state.v) < self.goal_reached_vel_tol:
                # Goal reached!
                self.state.mode = "goal_reached"
                self.last_info = {"status": "goal_reached", "final_pos_err": pos_err}
                return np.zeros(2)

            elif pos_err < self.final_convergence_threshold:
                # Close to goal, switch to final convergence
                self.state.mode = "final_convergence"
                print(f"[HYBRID CONTROLLER] Switching to final convergence (pos_err={pos_err:.3f}m)")
                return self._final_convergence(state, goal, obstacles, profile)

            else:
                # Far from goal, continue tracking last waypoint
                print(f"[HYBRID CONTROLLER] ⚠️ Reference ended but far from goal (pos_err={pos_err:.3f}m)")
                self.state.mode = "planning"
                self.state.reference = None
                self.state.current_step = 0
                self.state.ref_step_time_left = 0.0
                return self._plan_and_execute(state, goal, obstacles, profile)

        # Track reference using MPC
        sol = self.mpc.track_trajectory(
            state=state,
            reference=self.state.reference,
            obstacles=obstacles,
            step=self.state.current_step,
            profile=profile
        )

        if not sol.success:
            print(f"[HYBRID CONTROLLER] ❌ Tracking failed at step {self.state.current_step} (braking to stop)")
            self.last_info = {"status": "tracking_failed", "step": self.state.current_step}
            max_acc = float(self.env_cfg.get("vehicle", {}).get("max_acc", 1.0))
            desired_accel = -float(state.v) / max(self.dt, 1e-6)
            desired_accel = float(np.clip(desired_accel, -max_acc, max_acc))
            return np.array([0.0, desired_accel], dtype=float)

        # Increment tracking count
        self.state.tracking_count += 1

        # NEW: dt-consistent playback of the reference (fixed to allow completion)
        ref = self.state.reference
        if ref is not None and ref.dt_array is not None and len(ref.dt_array) > 0:
            self.state.ref_step_time_left -= self.dt

            # Advance index while we've consumed the time for the current ref step
            while self.state.ref_step_time_left <= 0.0 and self.state.current_step < ref.n_steps:
                self.state.current_step += 1
                if self.state.current_step < ref.n_steps:
                    # dt_array is indexed by step; guard length just in case
                    i = min(self.state.current_step, len(ref.dt_array) - 1)
                    self.state.ref_step_time_left += float(ref.dt_array[i])
        else:
            # Fallback: if dt_array missing, step each tick (old behavior)
            self.state.current_step += 1

        # If reference is now exhausted, switch modes immediately
        if self.state.current_step >= ref.n_steps:
            self.state.mode = "final_convergence"
            return self._final_convergence(state, goal, obstacles, profile)

        # Diagnostics
        pos_err = np.sqrt((state.x - goal.x)**2 + (state.y - goal.y)**2)
        self.last_info = {
            "status": "tracking",
            "step": self.state.current_step,
            "pos_err": pos_err,
            "mode": "tracking"
        }

        return sol.controls[0]

    def _final_convergence(self, state: VehicleState, goal: ParkingGoal,
                          obstacles: List[Obstacle], profile: str) -> np.ndarray:
        """
        Final convergence mode after reference ends.

        Uses standard MPC with careful goal-reaching to avoid aggressive maneuvering.
        """

        # Check if goal reached - CRITICAL: Use CAR CENTER (same as environment)
        # Environment checks car center position, not rear-axle
        L = float(self.env_cfg.get("vehicle", {}).get("length", 0.36))
        dist_to_center = L / 2.0 - 0.05  # 0.13m for L=0.36 (matches env calculation)

        # Convert rear-axle to car center
        cx = state.x + dist_to_center * np.cos(state.yaw)
        cy = state.y + dist_to_center * np.sin(state.yaw)

        # Goal is stored as rear-axle, convert to center
        goal_cx = goal.x + dist_to_center * np.cos(goal.yaw)
        goal_cy = goal.y + dist_to_center * np.sin(goal.yaw)

        # Check car center position (matches environment's _is_success)
        pos_err = np.sqrt((cx - goal_cx)**2 + (cy - goal_cy)**2)
        yaw_err = abs((state.yaw - goal.yaw + np.pi) % (2 * np.pi) - np.pi)

        if pos_err < self.goal_reached_pos_tol and \
           yaw_err < self.goal_reached_yaw_tol and \
           abs(state.v) < self.goal_reached_vel_tol:
            # Goal reached!
            self.state.mode = "goal_reached"
            self.last_info = {"status": "goal_reached", "final_pos_err": pos_err}
            print(f"[HYBRID CONTROLLER] ✓ Goal reached (pos_err={pos_err:.4f}m)")
            return np.zeros(2)

        # NEW: If very close and well-aligned, enter HOLD MODE
        # This prevents drift and guarantees termination
        # CRITICAL: Threshold (4cm) must be SMALLER than success threshold (5cm)
        if pos_err < 0.04 and yaw_err < 0.05:
            # Within 4cm and 3° - just BRAKE TO STOP
            if abs(state.v) > 0.01:
                # Apply braking to stop
                max_acc = float(self.env_cfg.get("vehicle", {}).get("max_acc", 1.0))
                desired_accel = -float(state.v) / max(self.dt, 1e-6)
                desired_accel = float(np.clip(desired_accel, -max_acc, max_acc))
                self.last_info = {
                    "status": "final_hold_braking",
                    "pos_err": pos_err,
                    "yaw_err": yaw_err,
                    "mode": "final_convergence"
                }
                return np.array([0.0, desired_accel], dtype=float)
            else:
                # Already stopped - hold position
                self.last_info = {
                    "status": "final_hold_stopped",
                    "pos_err": pos_err,
                    "yaw_err": yaw_err,
                    "mode": "final_convergence"
                }
                return np.array([0.0, 0.0], dtype=float)

        # Use standard MPC for final approach with distance-adaptive weights
        # FIXED: Keep weights strong throughout - don't weaken when close
        original_w_xy = self.mpc.w_goal_xy
        original_w_theta = self.mpc.w_goal_theta
        original_w_v = self.mpc.w_goal_v

        # Distance-adaptive weights
        # Strategy: STAY STRONG when close (prevent drift)
        if pos_err > 0.05:
            # Far from goal (>5cm): strong pull, normal velocity penalty
            self.mpc.w_goal_xy = 600.0
            self.mpc.w_goal_theta = 180.0
            self.mpc.w_goal_v = 3.0  # Normal velocity penalty
        else:
            # Close to goal (≤5cm): STAY STRONG + FORCE STOPPING
            # This ensures aggressive braking in the final centimeters
            self.mpc.w_goal_xy = 600.0  # Keep strong (was 250-400)
            self.mpc.w_goal_theta = 180.0  # Keep strong (was 75-120)
            self.mpc.w_goal_v = 10.0  # 10x stronger - FORCE STOP (was 3.0)
            # Debug logging (only print once when entering this region)
            if not hasattr(self, '_logged_velocity_boost'):
                print(f"[HYBRID] Velocity boost activated: w_goal_v=10.0 (pos_err={pos_err:.4f}m)", flush=True)
                self._logged_velocity_boost = True

        sol = self.mpc.solve(
            state=state,
            goal=goal,
            obstacles=obstacles,
            profile=profile
        )

        # Restore original weights
        self.mpc.w_goal_xy = original_w_xy
        self.mpc.w_goal_theta = original_w_theta
        self.mpc.w_goal_v = original_w_v

        if not sol.success:
            print(f"[HYBRID CONTROLLER] ❌ Final convergence failed")
            self.last_info = {"status": "final_convergence_failed"}
            return np.zeros(2)

        self.last_info = {
            "status": "final_convergence",
            "pos_err": pos_err,
            "yaw_err": yaw_err,
            "mode": "final_convergence"
        }

        return sol.controls[0]

    def get_info(self) -> dict:
        """
        Get diagnostic information about controller state.

        Returns:
            Dictionary with current mode, step, errors, etc.
        """
        info = {
            "mode": self.state.mode,
            "current_step": self.state.current_step,
            "planning_count": self.state.planning_count,
            "tracking_count": self.state.tracking_count,
        }

        if self.state.reference is not None:
            info["reference_steps"] = self.state.reference.n_steps
            info["reference_time"] = self.state.reference.total_time

        info.update(self.last_info)
        return info

    def is_goal_reached(self) -> bool:
        """Check if goal has been reached."""
        return self.state.mode == "goal_reached"

    def get_reference_trajectory(self) -> Optional[ReferenceTrajectory]:
        """Get the current reference trajectory (for visualization/analysis)."""
        return self.state.reference


def test_hybrid_controller():
    """Test script for HybridController."""
    import yaml
    import copy
    from env.parking_env import ParkingEnv

    print("="*70)
    print("Testing HybridController")
    print("="*70)

    # Load environment config
    with open("config_env.yaml", "r") as f:
        cfg_full = yaml.safe_load(f)

    # Merge parallel scenario
    scenario = "parallel"
    env_cfg = copy.deepcopy(cfg_full)
    scenario_cfg = copy.deepcopy(cfg_full["scenarios"][scenario])
    for key, val in scenario_cfg.items():
        env_cfg[key] = val

    # Create environment
    env = ParkingEnv(config=env_cfg)
    env.reset()

    # Create hybrid controller
    controller = HybridController(
        config_path="mpc/config_mpc.yaml",
        env_cfg=env_cfg,
        dt=env_cfg["dt"]
    )

    # Get initial state and goal
    state = VehicleState(
        x=env.state[0],
        y=env.state[1],
        yaw=env.state[2],
        v=env.state[3]
    )

    goal = ParkingGoal(
        x=env.goal[0],
        y=env.goal[1],
        yaw=env.goal[2]
    )

    # Convert obstacles
    obstacles = []
    for o in env.obstacles.obstacles:
        cx, cy, w, h = o["x"], o["y"], o["w"], o["h"]
        obstacles.append(Obstacle(cx=cx, cy=cy, hx=w/2, hy=h/2))

    print(f"\n[TEST] Initial state: x={state.x:.2f}, y={state.y:.2f}, yaw={state.yaw:.2f}")
    print(f"[TEST] Goal: x={goal.x:.2f}, y={goal.y:.2f}, yaw={goal.yaw:.2f}")

    # Run control loop
    max_steps = 100
    for step in range(max_steps):
        # Get control from hybrid controller
        control = controller.get_control(state, goal, obstacles, profile="parallel")

        # Execute in environment
        obs, reward, done, info = env.step(control)

        # Update state
        state = VehicleState(
            x=env.state[0],
            y=env.state[1],
            yaw=env.state[2],
            v=env.state[3]
        )

        # Get controller info
        ctrl_info = controller.get_info()

        # Progress
        pos_err = np.sqrt((state.x - goal.x)**2 + (state.y - goal.y)**2)

        if step % 10 == 0 or step < 5:
            print(f"  [Step {step:2d}] mode={ctrl_info['mode']:<18} pos_err={pos_err:.3f}m  "
                  f"steer={control[0]:+.3f} accel={control[1]:+.3f}")

        # Check termination
        if controller.is_goal_reached():
            print(f"\n✓ Goal reached at step {step}")
            print(f"  Final error: {pos_err:.4f}m")
            break

        if done and not info.get("success", False):
            print(f"\n❌ Episode ended at step {step} (collision or failure)")
            break

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    final_info = controller.get_info()
    print(f"Mode: {final_info['mode']}")
    print(f"Total steps: {step}")
    print(f"Planning count: {final_info['planning_count']}")
    print(f"Tracking count: {final_info['tracking_count']}")

    if controller.state.reference is not None:
        ref = controller.state.reference
        print(f"Reference: {ref.n_steps} steps, {ref.total_time:.2f}s")

    print("="*70)


if __name__ == "__main__":
    test_hybrid_controller()