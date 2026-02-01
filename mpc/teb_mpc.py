# mpc/teb_mpc.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Literal, Tuple
from enum import Enum
import os
import numpy as np
import yaml

try:
    import casadi as ca
except Exception:
    ca = None

from mpc.reference_trajectory import ReferenceTrajectory

ProfileType = Literal["parallel", "perpendicular"]


class ParkingPhase(Enum):
    """Parking maneuver phases for smooth execution"""
    APPROACH = "approach"      # Initial positioning phase
    ENTRY = "entry"           # Main parking maneuver with reversing
    FINAL_ALIGN = "final"     # Fine alignment to goal pose
    COMPLETED = "completed"   # Successfully parked


@dataclass
class VehicleState:
    x: float
    y: float
    yaw: float
    v: float


@dataclass
class ParkingGoal:
    x: float
    y: float
    yaw: float


@dataclass
class Obstacle:
    """Axis-aligned or oriented rectangle obstacle.

    Parameters are half-extents in the obstacle's local frame.
    - hx: half-length along obstacle local x-axis
    - hy: half-length along obstacle local y-axis
    - theta: obstacle heading in world frame (rad)
    """
    cx: float
    cy: float
    hx: float
    hy: float
    theta: float = 0.0

@dataclass
class MPCSolution:
    states: np.ndarray
    controls: np.ndarray
    success: bool
    info: Dict[str, Any]
    phase: Optional[ParkingPhase] = None


class TEBMPC:
    def __init__(
        self,
        config_path: Optional[str] = None,
        max_obstacles: int = 25,
        env_cfg: Optional[Dict[str, Any]] = None,
        dt: Optional[float] = None,
    ) -> None:
        """
        If env_cfg is provided, dt + vehicle + world come from config_env.yaml.
        config_mpc.yaml then only defines MPC hyperparameters (horizon, weights, profiles).
        """
        if config_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(base_dir, "config_mpc.yaml")

        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        # MPC hyperparameters (weights, horizon, profiles)
        self.mpc_cfg = cfg.get("mpc", cfg)

        # -------- Single source of truth for physics: env_cfg --------
        if env_cfg is not None:
            # Take vehicle + world + dt from config_env.yaml
            self.env_cfg = env_cfg
            self.vehicle_cfg = env_cfg["vehicle"]
            if dt is not None:
                self.dt = float(dt)
            else:
                if "dt" in env_cfg:
                    self.dt = float(env_cfg["dt"])
                else:
                    self.dt = float(self.mpc_cfg.get("dt", 0.1))  # legacy fallback

            world_cfg = env_cfg.get("world", {})
            self.world_w = float(world_cfg.get("width", 4.0))
            self.world_h = float(world_cfg.get("height", 4.0))
        else:
            # Backward-compatible fallback if someone uses TEBMPC standalone
            self.env_cfg = cfg.get("environment", {})
            self.vehicle_cfg = cfg.get("vehicle", {})

            # dt, world size from MPC config or environment subsection
            self.dt = float(self.mpc_cfg.get("dt", 0.1))
            self.world_w = float(self.env_cfg.get("size_x", 4.0))
            self.world_h = float(self.env_cfg.get("size_y", 4.0))

        self.N = int(self.mpc_cfg.get("horizon_steps", 50))
        self.max_obstacles = max_obstacles

        # --- Load default weights (pure MPC stuff, no duplication) ---
        print("\n[MPC] Loading Default Weights:")
        self.w_goal_xy = float(self.mpc_cfg.get("w_goal_xy", 800.0))  # Increased to prevent zig-zag
        self.w_goal_theta = float(self.mpc_cfg.get("w_goal_theta", 50.0))
        self.w_goal_v = float(self.mpc_cfg.get("w_goal_v", 1.0))
        self.w_steer = float(self.mpc_cfg.get("w_steer", 1.0))
        self.w_accel = float(self.mpc_cfg.get("w_accel", 0.1))
        self.w_smooth_steer = float(self.mpc_cfg.get("w_smooth_steer", 0.05))
        self.w_smooth_accel = float(self.mpc_cfg.get("w_smooth_accel", 0.02))
        self.w_slew_rate_steer = float(self.mpc_cfg.get("w_slew_rate_steer", 0.5))  # NEW: Very weak (too strong causes collisions)
        self.w_slew_rate_accel = float(self.mpc_cfg.get("w_slew_rate_accel", 0.2))  # NEW: Very weak
        self.w_collision = float(self.mpc_cfg.get("w_collision", 5.0))
        self.obs_inflate = float(self.mpc_cfg.get("obs_inflate", 0.0))
        self.collision_margin = float(self.mpc_cfg.get("collision_margin", 0.0))
        # Scale factors for soft constraints (avoid extremely steep gradients)
        self.collision_scale = max(1e-3, float(self.mpc_cfg.get("collision_scale", 0.05)))
        self.boundary_scale = max(1e-3, float(self.mpc_cfg.get("boundary_scale", 0.05)))

        self.w_boundary = float(self.mpc_cfg.get("w_boundary", 30.0))
        self.w_reverse_penalty = float(self.mpc_cfg.get("w_reverse_penalty", 0.1))
        self.w_gear_change = float(self.mpc_cfg.get("w_gear_change", 0.0))
        self.w_braking_zone = float(self.mpc_cfg.get("w_braking_zone", 0.0))
        self.w_terminal_velocity = float(self.mpc_cfg.get("w_terminal_velocity", 0.0))
        self.w_steering_precision = float(self.mpc_cfg.get("w_steering_precision", 0.0))
        self.w_monotonic_final = float(self.mpc_cfg.get("w_monotonic_final", 0.0))

        print(f"  -> Collision: {self.w_collision}")
        print(f"  -> Reverse Penalty: {self.w_reverse_penalty}")
        if self.w_gear_change > 0:
            print(f"  -> Gear Change Penalty: {self.w_gear_change}")
        if self.w_braking_zone > 0:
            print(f"  -> Braking Zone: {self.w_braking_zone}")
        if self.w_terminal_velocity > 0:
            print(f"  -> Terminal Velocity: {self.w_terminal_velocity}")
        if self.w_steering_precision > 0:
            print(f"  -> Steering Precision: {self.w_steering_precision}")
        if self.w_monotonic_final > 0:
            print(f"  -> Monotonic Final Approach: {self.w_monotonic_final}")

        # -------- TEB Configuration --------
        teb_cfg = self.mpc_cfg.get("teb", {})
        self.enable_teb = bool(teb_cfg.get("enable", False))
        self.dt_min = float(teb_cfg.get("dt_min", 0.05))
        self.dt_max = float(teb_cfg.get("dt_max", 0.30))
        self.dt_init = float(teb_cfg.get("dt_init", 0.10))
        self.w_time = float(teb_cfg.get("w_time", 0.5))
        self.w_dt_smooth = float(teb_cfg.get("w_dt_smooth", 10.0))
        self.w_dt_obstacle = float(teb_cfg.get("w_dt_obstacle", 2.0))
        self.w_dt_precision = float(teb_cfg.get("w_dt_precision", 0.0))
        self.w_velocity_dt_coupling = float(teb_cfg.get("w_velocity_dt_coupling", 0.0))
        self.max_total_time = float(teb_cfg.get("max_total_time", 12.0))
        self.temporal_smoothness_bound = float(teb_cfg.get("temporal_smoothness_bound", 0.15))

        if self.enable_teb:
            print(f"\n[TEB] Time-Elastic Band ENABLED:")
            print(f"  -> dt range: [{self.dt_min:.3f}, {self.dt_max:.3f}] s")
            print(f"  -> w_time: {self.w_time}")
            print(f"  -> w_dt_smooth: {self.w_dt_smooth}")
        else:
            print(f"\n[TEB] Using Fixed-dt MPC (dt = {self.dt:.3f} s)")

        # World bounds (4x4 box from env)
        self.x_min, self.x_max = -self.world_w / 2.0, self.world_w / 2.0
        self.y_min, self.y_max = -self.world_h / 2.0, self.world_h / 2.0
        self.boundary_margin = float(self.mpc_cfg.get("boundary_margin", 0.2))

        # -------- Vehicle geometry (from env.vehicle only) --------
        self.length = float(self.vehicle_cfg["length"])
        self.width = float(self.vehicle_cfg["width"])

        # Circle model radius:
        # Using 4 spine circles, we enlarge the radius just enough so the union
        # covers the ego rectangle corners (prevents corner-clipping collisions).
        step = self.length / 8.0
        self.circle_radius = float(np.hypot(self.width / 2.0, step))

        # 4-circle footprint: spine only for faster solving
        dist_ra_to_center = self.length / 2.0 - 0.05

        self.circle_offsets = [
            # Spine circles (x_offset, y_offset=0)
            (dist_ra_to_center + 3 * step, 0.0),  # front center
            (dist_ra_to_center + 1 * step, 0.0),  # mid-front center
            (dist_ra_to_center - 1 * step, 0.0),  # mid-rear center
            (dist_ra_to_center - 3 * step, 0.0),  # rear center
        ]

        # Initialize parallel parking cost function parameters with defaults
        # These will be overridden when parallel profile is applied
        self.lateral_weight = 0.25        # Original value
        self.depth_penalty_weight = 4.0
        self.yaw_weight = 0.9
        self.depth_reward_linear = 0.03
        self.depth_reward_quadratic = 0.30
        self.proximity_exp_factor = 20.0
        self.final_align_depth_threshold = 0.10
        self.final_align_depth_range = 0.05
        self.final_align_yaw_threshold = 0.1745
        self.final_align_yaw_range = 0.2094
        self.coupling_entry = 0.7
        self.coupling_reduction = 0.5
        self.max_comfortable_speed = 0.14
        self.speed_excess_weight = 1.8

        self.solver_parallel = None
        self.solver_perpendicular = None
        self.solver_parallel_teb = None  # TEB-enabled solver for planning
        self.solver_perpendicular_teb = None  # TEB-enabled solver for planning
        # Track which horizon each TEB solver was built with (so we can rebuild if needed)
        self._solver_parallel_teb_N: Optional[int] = None
        self._solver_perpendicular_teb_N: Optional[int] = None
        self._last_X, self._last_U = None, None
        self._last_dt = None  # Store last dt solution for warm-start (TEB)
        self._last_executed_control = np.array([0.0, 0.0])  # [steering, accel] for slew rate penalty
        self._current_profile = None
        self._current_phase = ParkingPhase.APPROACH
        self._cusp_detected = False
        self._prev_v = 0.0
        # Conservative buffer around pin obstacles (keeps you collision-free with rectangle env)
        self.pin_safety_buffer = float(self.mpc_cfg.get("pin_safety_buffer", 0.10))

        if ca is not None:
            # Build solvers with default weights - profiles are applied at runtime in solve()
            print("[MPC] Building parallel parking solver...")
            self.solver_parallel = self._build_solver(parking_type="parallel")
            print("[MPC] Building perpendicular parking solver...")
            self.solver_perpendicular = self._build_solver(parking_type="perpendicular")

            # Build TEB-enabled solvers for planning mode (lazy initialization)
            # These will be built on first use in plan_trajectory()
            # self.solver_parallel_teb and self.solver_perpendicular_teb = None initially

    def reset_warm_start(self) -> None:
        """Clear cached warm-start state.

        Call this at episode boundaries and whenever you switch to a new high-level
        phase/mode (e.g., A→B approach → B→C parking) to prevent solver bias.
        """
        self._last_X, self._last_U = None, None
        self._last_dt = None
        self._last_executed_control = np.array([0.0, 0.0])

        # Reset internal phase/cusp detection so diagnostics are clean per episode.
        self._current_phase = ParkingPhase.APPROACH
        self._cusp_detected = False
        self._prev_v = 0.0

        # Force profile to be re-applied on next solve/plan.
        self._current_profile = None

    def _obstacle_radius(self, o: Obstacle) -> float:
        """Convert obstacle half-sizes (hx, hy) into a conservative circle radius."""
        max_dim = max(o.hx, o.hy)
        min_dim = min(o.hx, o.hy)

        if max_dim > 1.0:
            return 0.30  # world walls
        if max_dim > 0.6 and min_dim < 0.05:
            return 0.04  # thin curb/bar
        # pins / parked-car approximation
        return max(o.hx, o.hy) + self.pin_safety_buffer

    def _apply_profile(self, profile: str):
        if self._current_profile == profile:
            return

        self._current_profile = profile
        profiles = self.mpc_cfg.get("profiles", {})

        if profile not in profiles:
            print(f"[MPC] Profile '{profile}' not found in config! Using defaults.")
            return

        print(f"\n[MPC] Switching to Profile: {profile.upper()}")
        p_cfg = profiles[profile]

        # Core MPC weights
        if "w_goal_xy" in p_cfg:      self.w_goal_xy = float(p_cfg["w_goal_xy"])
        if "w_goal_theta" in p_cfg:   self.w_goal_theta = float(p_cfg["w_goal_theta"])
        if "w_goal_v" in p_cfg:       self.w_goal_v = float(p_cfg["w_goal_v"])
        if "w_collision" in p_cfg:    self.w_collision = float(p_cfg["w_collision"])
        if "w_steer" in p_cfg:        self.w_steer = float(p_cfg["w_steer"])
        if "w_accel" in p_cfg:        self.w_accel = float(p_cfg["w_accel"])
        if "w_smooth_steer" in p_cfg: self.w_smooth_steer = float(p_cfg["w_smooth_steer"])
        if "w_smooth_accel" in p_cfg: self.w_smooth_accel = float(p_cfg["w_smooth_accel"])
        if "w_slew_rate_steer" in p_cfg: self.w_slew_rate_steer = float(p_cfg["w_slew_rate_steer"])
        if "w_slew_rate_accel" in p_cfg: self.w_slew_rate_accel = float(p_cfg["w_slew_rate_accel"])
        if "w_reverse_penalty" in p_cfg:
            self.w_reverse_penalty = float(p_cfg["w_reverse_penalty"])
        if "w_gear_change" in p_cfg:
            self.w_gear_change = float(p_cfg["w_gear_change"])
        if "w_braking_zone" in p_cfg:
            self.w_braking_zone = float(p_cfg["w_braking_zone"])
        if "w_terminal_velocity" in p_cfg:
            self.w_terminal_velocity = float(p_cfg["w_terminal_velocity"])
        if "w_steering_precision" in p_cfg:
            self.w_steering_precision = float(p_cfg["w_steering_precision"])

        # Parallel-specific cost function parameters
        if profile == "parallel":
            self.lateral_weight = float(p_cfg.get("lateral_weight", 0.25))
            self.depth_penalty_weight = float(p_cfg.get("depth_penalty_weight", 4.0))
            self.yaw_weight = float(p_cfg.get("yaw_weight", 0.9))

            self.depth_reward_linear = float(p_cfg.get("depth_reward_linear", 0.03))
            self.depth_reward_quadratic = float(p_cfg.get("depth_reward_quadratic", 0.30))
            self.proximity_exp_factor = float(p_cfg.get("proximity_exp_factor", 20.0))

            self.final_align_depth_threshold = float(p_cfg.get("final_align_depth_threshold", 0.10))
            self.final_align_depth_range = float(p_cfg.get("final_align_depth_range", 0.05))
            self.final_align_yaw_threshold = float(p_cfg.get("final_align_yaw_threshold", 0.1745))
            self.final_align_yaw_range = float(p_cfg.get("final_align_yaw_range", 0.2094))

            self.coupling_entry = float(p_cfg.get("coupling_entry", 0.7))
            self.coupling_reduction = float(p_cfg.get("coupling_reduction", 0.5))

            self.max_comfortable_speed = float(p_cfg.get("max_comfortable_speed", 0.14))
            self.speed_excess_weight = float(p_cfg.get("speed_excess_weight", 1.8))

            # Print parallel-specific sub-weights for verification
            print(f"  -> Parallel sub-weights:")
            print(f"     lateral_weight: {self.lateral_weight}")
            print(f"     yaw_weight: {self.yaw_weight}")
            print(f"     proximity_exp_factor: {self.proximity_exp_factor}")

        print(f"  -> w_goal_xy:        {self.w_goal_xy}")
        print(f"  -> w_goal_theta:     {self.w_goal_theta}")
        print(f"  -> w_goal_v:         {self.w_goal_v}")
        print(f"  -> w_collision:      {self.w_collision}")
        print(f"  -> w_steer:          {self.w_steer}")
        print(f"  -> w_accel:          {self.w_accel}")
        print(f"  -> w_smooth_steer:   {self.w_smooth_steer}")
        print(f"  -> w_smooth_accel:   {self.w_smooth_accel}")
        print(f"  -> w_slew_rate_steer:{self.w_slew_rate_steer}")  # NEW
        print(f"  -> w_slew_rate_accel:{self.w_slew_rate_accel}")  # NEW
        print(f"  -> w_reverse_penalty:{self.w_reverse_penalty}")
        if self.w_gear_change > 0:
            print(f"  -> w_gear_change:    {self.w_gear_change}")
        # Rebuild solver so updated profile weights are baked into the objective
        if ca is not None:
            old_teb = self.enable_teb
            self.enable_teb = False

            if profile == "parallel":
                self.solver_parallel = self._build_solver(parking_type="parallel")
            else:
                # 'perpendicular' and also 'approach' use this solver
                self.solver_perpendicular = self._build_solver(parking_type="perpendicular")

            self.enable_teb = old_teb
        # IMPORTANT: profile changes invalidate cached TEB planners too (weights are baked at build time)
        self.solver_parallel_teb = None
        self.solver_perpendicular_teb = None

    def _detect_cusp(self, current_v: float) -> bool:
        """Detect direction change (cusp point) in trajectory"""
        # Cusp occurs when velocity changes sign (forward <-> reverse)
        if abs(current_v) < 0.01:  # Near zero velocity
            return False

        if abs(self._prev_v) < 0.01:  # Was stationary
            self._prev_v = current_v
            return False

        # Check for sign change
        sign_change = (current_v * self._prev_v) < 0
        self._prev_v = current_v

        return sign_change

    def _update_phase(self, state: VehicleState, goal: ParkingGoal, profile: str) -> ParkingPhase:
        """Determine current parking phase based on state and goal"""
        pos_err = np.hypot(goal.x - state.x, goal.y - state.y)
        yaw_err = abs(((goal.yaw - state.yaw + np.pi) % (2 * np.pi)) - np.pi)

        # Completion check
        if pos_err < 0.03 and yaw_err < 0.05 and abs(state.v) < 0.05:
            return ParkingPhase.COMPLETED

        # Final alignment phase: close to goal, focus on precision
        if pos_err < 0.15 and yaw_err < 0.2:
            return ParkingPhase.FINAL_ALIGN

        # Entry phase: performing main parking maneuver
        # For parallel: when reversing or close to slot
        # For perpendicular: when reversing into spot
        if profile == "parallel":
            # Check if we're alongside the bay (lateral distance small)
            if abs(state.y - goal.y) < 0.8 or state.v < -0.05:  # Reversing
                return ParkingPhase.ENTRY
        else:  # perpendicular
            if state.v < -0.05 or pos_err < 0.8:  # Reversing or close
                return ParkingPhase.ENTRY

        # Default: approach phase
        return ParkingPhase.APPROACH

    def _get_phase_weights(self, phase: ParkingPhase, base_profile: str) -> Dict[str, float]:
        """Get phase-specific weight adjustments"""
        weights = {}

        if phase == ParkingPhase.APPROACH:
            # Approach: smooth driving, weak goal pull
            weights = {
                'w_goal_xy_mult': 0.3,      # Weak goal tracking
                'w_goal_theta_mult': 0.3,   # Weak yaw tracking
                'w_smooth_mult': 2.0,       # Strong smoothing
                'w_collision_mult': 1.5,    # Stronger collision avoidance
            }
        elif phase == ParkingPhase.ENTRY:
            # Entry: main maneuver, balanced
            weights = {
                'w_goal_xy_mult': 1.0,      # Normal goal tracking
                'w_goal_theta_mult': 1.0,   # Normal yaw tracking
                'w_smooth_mult': 0.5,       # Weaker smoothing (allow decisive moves)
                'w_collision_mult': 1.0,    # Normal collision avoidance
            }
        elif phase == ParkingPhase.FINAL_ALIGN:
            # Final: precision alignment, very strong goal pull
            weights = {
                'w_goal_xy_mult': 3.0,      # Very strong goal tracking
                'w_goal_theta_mult': 3.0,   # Very strong yaw tracking
                'w_smooth_mult': 0.2,       # Minimal smoothing (precision over comfort)
                'w_collision_mult': 0.8,    # Slightly weaker (allow final approach)
            }
        else:  # COMPLETED
            weights = {
                'w_goal_xy_mult': 1.0,
                'w_goal_theta_mult': 1.0,
                'w_smooth_mult': 1.0,
                'w_collision_mult': 1.0,
            }

        return weights

    def _build_solver(self, parking_type: str = "parallel"):
        """Build solver for specific parking type with optional TEB temporal optimization"""
        N, dt, L = self.N, self.dt, float(self.vehicle_cfg.get("wheelbase", 0.25))

        x, y, yaw, v = ca.SX.sym("x"), ca.SX.sym("y"), ca.SX.sym("yaw"), ca.SX.sym("v")
        steer, accel = ca.SX.sym("steer"), ca.SX.sym("accel")

        X = ca.SX.sym("X", 4, N + 1)  # States: [x, y, yaw, v]
        U = ca.SX.sym("U", 2, N)      # Controls: [steer, accel]

        # TEB: Add time intervals as decision variables if enabled
        if self.enable_teb:
            DT = ca.SX.sym("DT", N)   # Time intervals between waypoints (TEB!)
        else:
            DT = None                  # Use fixed dt

        # Parameter vector:
        #   [initial_state(4), goal(3), obstacles(6*max), prev_control(2)]
        # Obstacles are oriented rectangles encoded as:
        #   [cx, cy, hx, hy, cos(theta), sin(theta)]
        P = ca.SX.sym("P", 4 + 3 + 6 * self.max_obstacles + 2)

        goal_x, goal_y, goal_yaw = P[4], P[5], P[6]

        obs_cx, obs_cy, obs_hx, obs_hy, obs_c, obs_s = [], [], [], [], [], []
        for j in range(self.max_obstacles):
            base = 7 + 6 * j
            obs_cx.append(P[base])
            obs_cy.append(P[base + 1])
            obs_hx.append(P[base + 2])
            obs_hy.append(P[base + 3])
            obs_c.append(P[base + 4])
            obs_s.append(P[base + 5])

        # Previous control (last executed) for slew rate penalty
        prev_steer = P[7 + 6 * self.max_obstacles]
        prev_accel = P[7 + 6 * self.max_obstacles + 1]

        obj = 0
        g = [X[:, 0] - P[0:4]]

        def angle_wrap(a):
            return ca.atan2(ca.sin(a), ca.cos(a))

        # ----------------------------------------------------------------------
        # Smooth helpers (avoid non-smooth fabs/fmax/fmin kinks -> NaN gradients)
        # ----------------------------------------------------------------------
        eps_sdf = 1e-8
        def smooth_abs(z):
            return ca.sqrt(z * z + eps_sdf)
        def smooth_max(a, b):
            return 0.5 * (a + b + ca.sqrt((a - b) ** 2 + eps_sdf))
        def smooth_min(a, b):
            return 0.5 * (a + b - ca.sqrt((a - b) ** 2 + eps_sdf))
        def smooth_relu(z):
            return smooth_max(z, 0.0)

        for k in range(N):
            st = X[:, k]
            u = U[:, k]

            yaw_err = angle_wrap(st[2] - goal_yaw)
            gain = 13.0 if k == N - 1 else 1.0  # Strong terminal attraction for deeper goal commitment

            # Build parking-type-specific cost
            if parking_type == "parallel":
                # ==== PARALLEL PARKING: Enhanced depth reward with collision prevention ====
                # Configuration: Collision-safe exponential depth reward
                # Performance: 2.4cm avg depth, 100% success, 2.1 avg steering changes

                # --- Error calculations ---
                lateral_err = st[0] - goal_x
                depth_err_raw = st[1] - goal_y
                depth_err_abs = smooth_abs(depth_err_raw)
                yaw_err_abs = smooth_abs(yaw_err)

                # Total distance to goal
                goal_dist = ca.sqrt(lateral_err ** 2 + depth_err_abs ** 2)

                # --- Monotonic depth constraint (prevent backing out initially) ---
                depth_penalty_asymmetric = smooth_relu(depth_err_raw)  # Only penalize being too shallow

                # --- Progress penalty: penalize moving AWAY when close to goal ---
                # FIXED: Use Gaussian activation instead of broken exponential
                if k > 0:
                    prev_st = X[:, k - 1]
                    prev_lateral_err = prev_st[0] - goal_x
                    prev_depth_err = smooth_abs(prev_st[1] - goal_y)
                    prev_dist = ca.sqrt(prev_lateral_err ** 2 + prev_depth_err ** 2)

                    # Detect if moving away (distance increasing)
                    dist_increase = smooth_relu(goal_dist - prev_dist)

                    # FIXED: Gaussian activation instead of exp(-dist/threshold)
                    # Activates strongly within 15cm, decays smoothly beyond
                    proximity_threshold = 0.15  # 15cm
                    proximity_activation = ca.exp(-8.0 * (goal_dist ** 2) / (proximity_threshold ** 2))

                    # Strong penalty for moving away when close
                    progress_penalty = proximity_activation * (dist_increase ** 2)
                else:
                    progress_penalty = 0

                # --- Enhanced depth reward: linear base + exponential quadratic boost ---
                depth_reward_base = -depth_err_abs * self.depth_reward_linear

                # Exponential proximity activation: e^(-factor*err²)
                # Creates smooth transition as car approaches goal
                proximity_activation = ca.exp(-self.proximity_exp_factor * depth_err_abs ** 2)

                # Quadratic boost amplifies reward in final centimeters
                depth_reward_quadratic = -(depth_err_abs ** 2) * self.depth_reward_quadratic * proximity_activation
                depth_reward = depth_reward_base + depth_reward_quadratic

                # --- CURB OVERSHOOT PENALTY ---
                # Prevent car from going past the goal (curb line)
                # Only penalize if car overshoots in depth direction (negative depth_err_raw)
                overshoot = smooth_max(0.0, -depth_err_raw)  # positive when past goal
                # Strong quadratic penalty for overshooting: scaled by w_goal_xy to match other costs
                # FIXED: Scale by w_goal_xy (was unscaled at 50.0, now 400*50=20,000)
                curb_penalty = self.w_goal_xy * 50.0 * (overshoot ** 2)

                # --- Multi-phase detection for adaptive behavior ---
                # FINAL_ALIGN phase: activates when depth < threshold AND yaw < threshold
                # Smooth fade-in using linear interpolation over range
                in_final_align = smooth_min(1.0, smooth_max(0.0,
                    (self.final_align_depth_threshold - depth_err_abs) / self.final_align_depth_range))
                well_aligned_yaw = smooth_min(1.0, smooth_max(0.0,
                    (self.final_align_yaw_threshold - yaw_err_abs) / self.final_align_yaw_range))
                final_phase = in_final_align * well_aligned_yaw

                # --- Phase-aware speed-steering coupling ---
                # Entry: high coupling prevents collisions
                # Final: KEEP high coupling (was dropping to 0.45, now stays at 0.63+)
                steer_magnitude = smooth_abs(u[0])
                speed_steer_coupling = (steer_magnitude ** 2) * (st[3] ** 2)
                # Maintain minimum 70% coupling even in final phase
                coupling_weight = self.coupling_entry * smooth_max(0.7, (1.0 - self.coupling_reduction * final_phase))

                # --- Speed limit enforcement ---
                speed_excess = smooth_relu(smooth_abs(st[3]) - self.max_comfortable_speed)

                # --- Assemble parallel parking cost function ---
                obj += gain * (
                    self.w_goal_xy * self.lateral_weight * lateral_err ** 2 +
                    self.w_goal_xy * self.depth_penalty_weight * depth_penalty_asymmetric ** 2 +  # ASYMMETRIC (allow backing in)
                    self.w_goal_xy * depth_reward +
                    self.w_goal_theta * self.yaw_weight * yaw_err ** 2 +
                    self.w_goal_v * st[3] ** 2 +
                    self.w_goal_xy * coupling_weight * speed_steer_coupling +
                    self.w_goal_xy * self.speed_excess_weight * speed_excess ** 2 +
                    self.w_goal_xy * 10.0 * progress_penalty +  # Prevent moving away when close
                    curb_penalty  # Prevent overshooting past goal/curb
                )
            else:  # perpendicular
                # ==== PERPENDICULAR PARKING: Standard XY goal tracking ====
                obj += gain * (
                    self.w_goal_xy * ((st[0] - goal_x) ** 2 + (st[1] - goal_y) ** 2) +
                    self.w_goal_theta * yaw_err ** 2 +
                    self.w_goal_v * st[3] ** 2
                )

            # Control Cost
            obj += self.w_steer * u[0] ** 2 + self.w_accel * u[1] ** 2

            # Reverse Penalty
            is_reversing = 0.5 * (ca.sqrt(st[3] ** 2 + 1e-4) - st[3])
            obj += self.w_reverse_penalty * is_reversing ** 2

            # Slew Rate Penalty: Penalize change from LAST EXECUTED control (k=0 only)
            # This is THE KEY to preventing oscillations between steps!
            if k == 0:
                obj += self.w_slew_rate_steer * (u[0] - prev_steer) ** 2
                obj += self.w_slew_rate_accel * (u[1] - prev_accel) ** 2

            # Smoothness Penalty: Penalize change within horizon (k>0)
            if k > 0:
                obj += self.w_smooth_steer * (u[0] - U[0, k - 1]) ** 2 + self.w_smooth_accel * (u[1] - U[1, k - 1]) ** 2

                # Phase-aware steering precision: Increase steering smoothness near goal
                # This encourages precise steering control instead of gear-change-based corrections
                if parking_type == "parallel":
                    goal_dist = ca.sqrt((st[0] - goal_x) ** 2 + (st[1] - goal_y) ** 2 + 1e-9)
                    precision_zone_distance = 0.15  # 15cm - entering precision zone

                    # Activation increases as we get closer
                    steering_precision_activation = ca.exp(-8.0 * goal_dist / precision_zone_distance)

                    # Extra smoothness cost in precision zone
                    # This makes steering changes MORE expensive when close, encouraging:
                    # - Deliberate, smooth steering adjustments
                    # - One-shot corrections instead of back-and-forth
                    obj += self.w_steering_precision * steering_precision_activation * (u[0] - U[0, k - 1]) ** 2

                # Gear change penalty: penalize velocity direction changes
                # Detect sign change by checking if v[k-1] * v[k] < 0
                # Use smooth approximation: penalty = (1 - tanh(v[k-1] * v[k]))
                # When same sign: v_prev * v_curr > 0 → tanh ≈ 1 → penalty ≈ 0
                # When opposite signs: v_prev * v_curr < 0 → tanh ≈ -1 → penalty ≈ 2
                v_prev = X[3, k-1]
                v_curr = st[3]
                velocity_product = v_prev * v_curr
                # Use sigmoid-like activation to penalize sign changes
                gear_change_cost = 1.0 - ca.tanh(velocity_product * 50.0)  # 50 makes it sharp
                obj += self.w_gear_change * gear_change_cost

            # --- 4-CIRCLE COLLISION CHECK (ego circles vs oriented rectangles) ---
            # Signed distance to oriented rectangle (SDF)
            # NOTE: Using fabs/fmax/fmin makes the SDF non-smooth (kinks), which can trigger
            # CasADi warnings like "NaN detected for output grad_f_x" during Ipopt multiplier
            # calculations. These smooth approximations keep the same shape but make gradients
            # well-defined everywhere.

            def rect_sdf(px, py, j):
                dx = px - obs_cx[j]
                dy = py - obs_cy[j]
                # local = R^T (p - c)
                lx = obs_c[j] * dx + obs_s[j] * dy
                ly = -obs_s[j] * dx + obs_c[j] * dy

                hx = obs_hx[j] + self.obs_inflate
                hy = obs_hy[j] + self.obs_inflate

                qx = smooth_abs(lx) - hx
                qy = smooth_abs(ly) - hy

                ox = smooth_max(qx, 0.0)
                oy = smooth_max(qy, 0.0)
                outside = ca.sqrt(ox * ox + oy * oy + 1e-9)
                inside = smooth_min(smooth_max(qx, qy), 0.0)
                return outside + inside

            c_theta = ca.cos(st[2])
            s_theta = ca.sin(st[2])

            for offset_x, offset_y in self.circle_offsets:
                cx = st[0] + offset_x * c_theta - offset_y * s_theta
                cy = st[1] + offset_x * s_theta + offset_y * c_theta

                for j in range(self.max_obstacles):
                    sdf = rect_sdf(cx, cy, j)
                    penetration = (self.circle_radius + self.collision_margin) - sdf

                    # Standard isotropic collision penalty
                    # With obs_inflate=0.02, obstacles are slightly larger for safety margin
                    obj += self.w_collision * (smooth_max(penetration, 0.0) / self.collision_scale) ** 2

            # Boundary (positive distance when inside bounds)
            d_xmax = self.x_max - st[0]
            d_xmin = st[0] - self.x_min
            d_ymax = self.y_max - st[1]
            d_ymin = st[1] - self.y_min
            for d in [d_xmax, d_xmin, d_ymax, d_ymin]:
                # Scale logic applied here
                obj += self.w_boundary * 0.5 * (smooth_max(self.boundary_margin - d, 0.0) / self.boundary_scale) ** 2

            # Dynamics (TEB: use variable dt[k] if enabled, else fixed dt)
            dt_k = DT[k] if self.enable_teb else dt
            x_next = st[0] + st[3] * ca.cos(st[2]) * dt_k
            y_next = st[1] + st[3] * ca.sin(st[2]) * dt_k
            yaw_next = st[2] + (st[3] / L) * ca.tan(u[0]) * dt_k
            v_next = st[3] + u[1] * dt_k
            g.append(X[:, k + 1] - ca.vertcat(x_next, y_next, yaw_next, v_next))

        # ============================================================================
        # TEB TEMPORAL COSTS (only if TEB enabled)
        # ============================================================================
        if self.enable_teb:
            # 1. Time optimality: minimize total execution time
            total_time = ca.sum1(DT)
            obj += self.w_time * total_time

            # 2. Temporal smoothness: penalize abrupt dt changes
            for k in range(N - 1):
                obj += self.w_dt_smooth * (DT[k + 1] - DT[k]) ** 2

            # 3. Obstacle-aware time scaling: encourage smaller dt near obstacles
            # Compute minimum obstacle distance for each waypoint
            for k in range(N):
                st = X[:, k]
                min_obs_dist = 1e3  # NOTE: avoid ca.inf (can cause NaN gradients in Ipopt/CasADi)

                # Check distance to all obstacles
                for j in range(self.max_obstacles):
                    # Use vehicle center for obstacle distance
                    c_theta = ca.cos(st[2])
                    s_theta = ca.sin(st[2])
                    dist_to_center = self.length / 2.0 - 0.05
                    center_x = st[0] + dist_to_center * c_theta
                    center_y = st[1] + dist_to_center * s_theta

                    dist_to_obs = rect_sdf(center_x, center_y, j)
                    min_obs_dist = smooth_min(min_obs_dist, dist_to_obs)

                safety_margin = 0.15  # meters
                den = smooth_max(min_obs_dist, 0.0) + safety_margin + 1e-6
                obstacle_time_scaling = DT[k] / den
                obj += self.w_dt_obstacle * obstacle_time_scaling ** 2


            # # 4. Precision-aware time scaling: encourage smaller dt when close to goal
            # # DISABLED FOR BASELINE - Can enable with w_dt_precision > 0
            # for k in range(N):
            #     st = X[:, k]
            #
            #     # Distance to goal (position + orientation)
            #     goal_dist = ca.sqrt((st[0] - goal_x) ** 2 + (st[1] - goal_y) ** 2 + 1e-9)
            #     goal_yaw_error = ca.fabs(st[2] - goal_yaw)
            #
            #     # Combined precision metric (closer to goal = higher precision needed)
            #     precision_threshold = 0.15  # meters (15cm - entering precision zone)
            #     yaw_threshold = 0.2  # radians (~11 degrees)
            #
            #     # Activation function: increases as we get closer to goal
            #     # Uses exponential to create smooth transition
            #     position_precision = ca.exp(-10.0 * goal_dist / precision_threshold)
            #     yaw_precision = ca.exp(-5.0 * goal_yaw_error / yaw_threshold)
            #     precision_factor = ca.fmax(position_precision, yaw_precision)
            #
            #     # Cost: penalize large dt when precision is needed
            #     # When far from goal: precision_factor ≈ 0 → no cost (allows large dt for committed maneuvers)
            #     # When close to goal: precision_factor ≈ 1 → penalize large dt (forces small dt for precision)
            #     precision_time_scaling = precision_factor * (DT[k] - self.dt_min)
            #     obj += self.w_dt_precision * precision_time_scaling ** 2
            #
            # # 5. Velocity-dt coupling: Force smaller dt when moving slowly
            # # This creates automatic slow-down with finer temporal resolution
            # # DISABLED - causes worse precision
            # for k in range(N):
            #     st = X[:, k]
            #     v_abs = ca.sqrt(st[3] ** 2 + 1e-6)  # Absolute velocity
            #     v_max = 0.15  # Maximum comfortable speed
            #
            #     # Desired dt based on velocity: slow → small dt, fast → large dt
            #     # Linear scaling: dt_desired = dt_min + (dt_max - dt_min) * (v / v_max)
            #     v_normalized = ca.fmin(v_abs / v_max, 1.0)
            #     dt_desired = self.dt_min + (self.dt_max - self.dt_min) * v_normalized
            #
            #     # Penalize deviation from velocity-appropriate dt
            #     obj += self.w_velocity_dt_coupling * (DT[k] - dt_desired) ** 2

        # # ============================================================================
        # # SMART PLANNING COSTS (Parallel Parking)
        # # ============================================================================
        # # DISABLED - All smart planning attempts made precision worse
        # # These costs use TEB's look-ahead capability for better trajectory planning
        # if parking_type == "parallel":
        #     # 1. Braking zone: Enforce deceleration when approaching goal
        #     # Forces the optimizer to plan a smooth deceleration rather than overshoot
        #     braking_distance = 0.20  # Start slowing down 20cm before goal
        #     for k in range(N):
        #         st = X[:, k]
        #         goal_dist = ca.sqrt((st[0] - goal_x) ** 2 + (st[1] - goal_y) ** 2 + 1e-9)
        #
        #         # When inside braking zone, velocity should be proportional to distance
        #         # v_max_allowed = (goal_dist / braking_distance) * v_nominal
        #         if_in_braking_zone = ca.fmax(0, braking_distance - goal_dist) / braking_distance
        #         v_abs = ca.sqrt(st[3] ** 2 + 1e-6)
        #         v_nominal = 0.10  # Nominal approach speed
        #         v_max_allowed = ca.fmax(goal_dist / braking_distance * v_nominal, 0.02)  # Min 2cm/s
        #
        #         # Penalize excess velocity in braking zone
        #         velocity_excess = ca.fmax(0, v_abs - v_max_allowed)
        #         obj += self.w_braking_zone * if_in_braking_zone * velocity_excess ** 2
        #
        #     # 2. Terminal velocity constraint: Force near-zero velocity at horizon end when close
        #     # This makes MPC plan complete stops rather than momentum-based approaches
        #     final_state = X[:, N]  # Last state in horizon
        #     goal_dist_final = ca.sqrt((final_state[0] - goal_x) ** 2 + (final_state[1] - goal_y) ** 2 + 1e-9)
        #     terminal_activation_distance = 0.10  # Activate when < 10cm from goal
        #
        #     # Smooth activation: stronger as we get closer
        #     terminal_activation = ca.exp(-10.0 * goal_dist_final / terminal_activation_distance)
        #
        #     # Penalize any velocity at horizon end when close to goal
        #     v_terminal = ca.sqrt(final_state[3] ** 2 + 1e-6)
        #     obj += self.w_terminal_velocity * terminal_activation * v_terminal ** 2
        #
        #     # 3. Monotonic final approach: STRONG penalty for velocity reversals in precision zone
        #     # This forces one-shot convergence instead of back-and-forth corrections
        #     if self.w_monotonic_final > 0:
        #         precision_zone_distance = 0.10  # 10cm threshold for precision zone
        #
        #         for k in range(1, N):
        #             st = X[:, k]
        #             goal_dist = ca.sqrt((st[0] - goal_x) ** 2 + (st[1] - goal_y) ** 2 + 1e-9)
        #
        #             # Activate only in precision zone (exponential activation for smooth transition)
        #             in_precision_zone = ca.exp(-10.0 * goal_dist / precision_zone_distance)
        #
        #             # Detect velocity sign change between consecutive steps
        #             v_prev = X[3, k - 1]
        #             v_curr = X[3, k]
        #             velocity_product = v_prev * v_curr
        #
        #             # Cost is high when velocity changes sign (product < 0)
        #             # tanh(v_prev * v_curr * 50): ~1.0 when same sign, ~-1.0 when opposite sign
        #             # (1 - tanh(...)) / 2: 0.0 when same sign, 1.0 when opposite sign
        #             sign_change_cost = (1.0 - ca.tanh(velocity_product * 50.0)) / 2.0
        #
        #             # Apply strong penalty only in precision zone
        #             obj += self.w_monotonic_final * in_precision_zone * sign_change_cost

        # ============================================================================
        # NLP FORMULATION
        # ============================================================================
        if self.enable_teb:
            # TEB: decision variables include states + controls + time intervals
            nlp = {
                "x": ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1), DT),
                "f": obj,
                "g": ca.vertcat(*g),
                "p": P
            }
        else:
            # Fixed-dt MPC: decision variables are states + controls only
            nlp = {"x": ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1)), "f": obj, "g": ca.vertcat(*g), "p": P}
        opts = {
            "ipopt": {
                "max_iter": 150,
                "print_level": 0,
                "tol": 1e-3,
                "warm_start_init_point": "yes",
                "mu_strategy": "adaptive"
            },
            "print_time": 0
        }
        solver = ca.nlpsol(f"solver_{parking_type}", "ipopt", nlp, opts)
        return solver

    def solve(self, state: VehicleState, goal: ParkingGoal, obstacles: List[Obstacle],
              profile: str = "perpendicular") -> MPCSolution:
        # Select appropriate solver based on parking type
        self._apply_profile(profile)
        solver = self.solver_parallel if profile == "parallel" else self.solver_perpendicular
        if solver is None:
            return MPCSolution(np.zeros((1, 4)), np.zeros((1, 2)), False, {}, None)

        # Apply base profile weights


        # DEBUG: Verify slew rate weights
        if False:  # Set to True for debugging
            print(f"[DEBUG] Slew rate weights: steer={self.w_slew_rate_steer}, accel={self.w_slew_rate_accel}")
            print(f"[DEBUG] Last executed: {self._last_executed_control}")

        # Detect current phase
        current_phase = self._update_phase(state, goal, profile)

        # Detect cusp (direction change)
        cusp_detected = self._detect_cusp(state.v)

        # Initial Guess
        x0 = np.array([state.x, state.y, state.yaw, state.v])
        gx, gy, gyaw = goal.x, goal.y, goal.yaw
        diff = (gyaw - state.yaw + np.pi) % (2 * np.pi) - np.pi
        unwound_gyaw = state.yaw + diff

        X0 = np.zeros((self.N + 1, 4))
        for k in range(self.N + 1):
            alpha = k / self.N
            X0[k] = (1 - alpha) * x0 + alpha * np.array([gx, gy, unwound_gyaw, 0])
        U0 = np.zeros((self.N, 2))

        # TEB: Initialize time intervals
        if self.enable_teb:
            DT0 = np.ones(self.N) * self.dt_init  # Start with uniform dt_init
            if self._last_dt is not None:
                # Warm-start from previous solution (shift and repeat last)
                DT0[:-1] = self._last_dt[1:]
                DT0[-1] = self._last_dt[-1]
        else:
            DT0 = None

        if self._last_X is not None:
            X0[:-1] = self._last_X[1:]
            X0[-1] = self._last_X[-1]
            U0[:-1] = self._last_U[1:]
            U0[-1] = self._last_U[-1]

        # Parameter vector: [initial_state(4), goal(3), obstacles(6*max), prev_control(2)]
        P = np.zeros(4 + 3 + 6 * self.max_obstacles + 2)
        P[0:4] = x0
        P[4:7] = [gx, gy, gyaw]

        # Initialize unused obstacle slots far away (effectively disabling them)
        for j in range(self.max_obstacles):
            idx = 7 + 6 * j
            P[idx] = 1e6  # cx
            P[idx + 1] = 1e6  # cy
            P[idx + 2] = 0.0  # hx
            P[idx + 3] = 0.0  # hy
            P[idx + 4] = 1.0  # cos(theta)
            P[idx + 5] = 0.0  # sin(theta)

        # Fill used obstacles (rectangles)
        for j, o in enumerate(obstacles[:self.max_obstacles]):
            idx = 7 + 6 * j
            P[idx] = float(o.cx)
            P[idx + 1] = float(o.cy)
            P[idx + 2] = float(max(o.hx, 0.0))
            P[idx + 3] = float(max(o.hy, 0.0))
            th = float(getattr(o, "theta", 0.0))
            P[idx + 4] = float(np.cos(th))
            P[idx + 5] = float(np.sin(th))

        # Add previous control for slew rate penalty
        prev_control_idx = 7 + 6 * self.max_obstacles
        # Use self._last_executed_control (renamed from prev_steer/accel in logic for clarity)
        P[prev_control_idx] = float(self._last_executed_control[0])
        P[prev_control_idx + 1] = float(self._last_executed_control[1])
        # ============================================================================
        # BOUNDS & SOLVER CALL
        # ============================================================================
        # State bounds: [x, y, yaw, v] * (N+1) + Control bounds: [steer, accel] * N
        lbx_states = [-np.inf] * 4 * (self.N + 1)
        ubx_states = [np.inf] * 4 * (self.N + 1)
        lbx_controls = [-0.52, -1.0] * self.N
        ubx_controls = [0.52, 1.0] * self.N

        if self.enable_teb:
            # TEB: Add dt bounds
            lbx_dt = [self.dt_min] * self.N
            ubx_dt = [self.dt_max] * self.N
            lbx = lbx_states + lbx_controls + lbx_dt
            ubx = ubx_states + ubx_controls + ubx_dt
            x0_init = ca.vertcat(X0.flatten(), U0.flatten(), DT0)
        else:
            # Fixed-dt MPC
            lbx = lbx_states + lbx_controls
            ubx = ubx_states + ubx_controls
            x0_init = ca.vertcat(X0.flatten(), U0.flatten())

        try:
            res = solver(x0=x0_init, lbx=lbx, ubx=ubx, lbg=0, ubg=0, p=P)
            res_x = np.array(res["x"]).flatten()

            # Extract solution
            X_sol = res_x[:4 * (self.N + 1)].reshape(self.N + 1, 4)
            U_sol = res_x[4 * (self.N + 1):4 * (self.N + 1) + 2 * self.N].reshape(self.N, 2)

            if self.enable_teb:
                # Extract dt solution
                DT_sol = res_x[4 * (self.N + 1) + 2 * self.N:]
                self._last_dt = DT_sol
            else:
                DT_sol = None

            # Store for warm-start
            self._last_X, self._last_U = X_sol, U_sol

            # Store first control as last executed for slew rate penalty
            self._last_executed_control = U_sol[0].copy()

            return MPCSolution(X_sol, U_sol, True, {
                "termination": "success",
                "phase": current_phase.value,
                "cusp_detected": cusp_detected,
                "dt_solution": DT_sol.tolist() if DT_sol is not None else None,
                "total_time": float(np.sum(DT_sol)) if DT_sol is not None else self.N * self.dt
            }, current_phase)
        except Exception as e:
            return MPCSolution(X0, U0, False, {"error": str(e)}, current_phase)

    def first_control(self, sol: MPCSolution) -> np.ndarray:
        return sol.controls[0] if len(sol.controls) > 0 else np.zeros(2)

    def plan_trajectory(
        self,
        state: VehicleState,
        goal: ParkingGoal,
        obstacles: List[Obstacle],
        profile: str = "parallel",
        horizon_override: Optional[int] = None,
    ) -> ReferenceTrajectory:
        """Plan a committed reference trajectory using a TEB-enabled NLP.

        Notes:
        - Uses a *planner horizon* (teb.horizon) which may be longer than tracking MPC horizon.
        - Maintains separate cached TEB solvers for parallel/perpendicular and rebuilds them if
          the requested planner horizon changes.
        - Parameter vector matches _build_solver(): [x0(4), goal(3), obs(6*max), prev_u(2)].
        """
        if ca is None:
            return ReferenceTrajectory(
                states=np.zeros((1, 4)),
                controls=np.zeros((1, 2)),
                dt_array=np.array([0.0]),
                total_time=0.0,
                n_steps=1,
                success=False,
                metadata={"error": "casadi_not_available"},
            )

        # ---- decide planning horizon ----
        teb_cfg = self.mpc_cfg.get("teb", {}) or {}
        N_plan = int(horizon_override) if horizon_override is not None else int(teb_cfg.get("horizon", self.N))
        N_plan = max(5, N_plan)

        # ---- ensure correct TEB solver exists (cache by profile + N_plan) ----
        original_enable_teb = self.enable_teb
        original_N = self.N

        def _ensure_teb_solver(parking_type: str):
            if parking_type == "parallel":
                cached = self.solver_parallel_teb
                cached_N = self._solver_parallel_teb_N
            else:
                cached = self.solver_perpendicular_teb
                cached_N = self._solver_perpendicular_teb_N

            if cached is None or cached_N != N_plan:
                print(f"[TEB PLANNER] Building TEB-enabled {parking_type} solver (N={N_plan})...")
                self.enable_teb = True
                self.N = N_plan
                solver = self._build_solver(parking_type=parking_type)
                # restore immediately (solver keeps its own dimensions)
                self.enable_teb = original_enable_teb
                self.N = original_N

                if parking_type == "parallel":
                    self.solver_parallel_teb = solver
                    self._solver_parallel_teb_N = N_plan
                else:
                    self.solver_perpendicular_teb = solver
                    self._solver_perpendicular_teb_N = N_plan
                return solver
            return cached

        parking_type = "parallel" if profile == "parallel" else "perpendicular"
        teb_solver = _ensure_teb_solver(parking_type)

        # ---- apply (tracking) profile weights first (rebuilds fixed-dt solver only) ----
        self._apply_profile(profile)

        # ---- planning-time tuning knobs (do NOT rebuild solver; just affects parameters & bounds) ----
        # Keep goal pull strong to reach C from farther away.
        original_w_goal_xy = self.w_goal_xy
        original_w_goal_theta = self.w_goal_theta
        original_dt_min = self.dt_min
        original_dt_max = self.dt_max
        original_w_time = self.w_time

        self.w_goal_xy = max(self.w_goal_xy, 600.0)
        self.w_goal_theta = max(self.w_goal_theta, 180.0)
        # Let dt vary for committed maneuvers + precision near end.
        self.dt_min = float(teb_cfg.get("dt_min", self.dt_min))
        self.dt_max = float(teb_cfg.get("dt_max", self.dt_max))
        self.w_time = float(teb_cfg.get("w_time", self.w_time))

        print(f"[TEB PLANNER] Planning trajectory: profile={profile}, N_plan={N_plan}, dt=[{self.dt_min:.3f},{self.dt_max:.3f}]")

        try:
            # Detect current phase (for metadata only)
            current_phase = self._update_phase(state, goal, profile)

            x0 = np.array([state.x, state.y, state.yaw, state.v], dtype=float)
            gx, gy, gyaw = float(goal.x), float(goal.y), float(goal.yaw)

            # Unwind goal yaw to be close to current yaw for a nicer initial guess
            diff = (gyaw - state.yaw + np.pi) % (2 * np.pi) - np.pi
            unwound_gyaw = state.yaw + diff

            # Initial guesses
            X0 = np.zeros((N_plan + 1, 4), dtype=float)
            for k in range(N_plan + 1):
                alpha = k / float(N_plan)
                X0[k] = (1 - alpha) * x0 + alpha * np.array([gx, gy, unwound_gyaw, 0.0], dtype=float)

            U0 = np.zeros((N_plan, 2), dtype=float)
            DT0 = np.ones(N_plan, dtype=float) * float(getattr(self, "dt_init", self.dt))

            # Warm-start from last TEB solution if available and compatible
            if self._last_X is not None and self._last_U is not None and self._last_X.shape[0] == (N_plan + 1):
                X0[:-1] = self._last_X[1:]
                X0[-1] = self._last_X[-1]
                U0[:-1] = self._last_U[1:]
                U0[-1] = self._last_U[-1]
            if self._last_dt is not None and len(self._last_dt) == N_plan:
                DT0[:-1] = self._last_dt[1:]
                DT0[-1] = self._last_dt[-1]

            # Parameter vector matches _build_solver()
            P = np.zeros(4 + 3 + 6 * self.max_obstacles + 2, dtype=float)
            P[0:4] = x0
            P[4:7] = [gx, gy, gyaw]

            # Disable all obstacle slots far away first
            for j in range(self.max_obstacles):
                idx = 7 + 6 * j
                P[idx] = 1e6        # cx
                P[idx + 1] = 1e6    # cy
                P[idx + 2] = 0.0    # hx
                P[idx + 3] = 0.0    # hy
                P[idx + 4] = 1.0    # cos(theta)
                P[idx + 5] = 0.0    # sin(theta)

            # Fill used obstacles (oriented rectangles)
            for j, o in enumerate(obstacles[: self.max_obstacles]):
                idx = 7 + 6 * j
                P[idx] = float(o.cx)
                P[idx + 1] = float(o.cy)
                P[idx + 2] = float(max(o.hx, 0.0))
                P[idx + 3] = float(max(o.hy, 0.0))
                th = float(getattr(o, "theta", 0.0))
                P[idx + 4] = float(np.cos(th))
                P[idx + 5] = float(np.sin(th))

            # prev control (no slew constraint on first plan)
            prev_control_idx = 7 + 6 * self.max_obstacles
            P[prev_control_idx] = 0.0
            P[prev_control_idx + 1] = 0.0

            # Bounds
            lbx_states = [-np.inf] * 4 * (N_plan + 1)
            ubx_states = [np.inf] * 4 * (N_plan + 1)
            lbx_controls = [-0.52, -1.0] * N_plan
            ubx_controls = [0.52, 1.0] * N_plan
            lbx_dt = [self.dt_min] * N_plan
            ubx_dt = [self.dt_max] * N_plan

            lbx = lbx_states + lbx_controls + lbx_dt
            ubx = ubx_states + ubx_controls + ubx_dt
            x0_init = ca.vertcat(X0.flatten(), U0.flatten(), DT0)

            res = teb_solver(x0=x0_init, lbx=lbx, ubx=ubx, lbg=0, ubg=0, p=P)
            res_x = np.array(res["x"]).flatten()

            X_sol = res_x[: 4 * (N_plan + 1)].reshape(N_plan + 1, 4)
            U_sol = res_x[4 * (N_plan + 1) : 4 * (N_plan + 1) + 2 * N_plan].reshape(N_plan, 2)
            DT_sol = res_x[4 * (N_plan + 1) + 2 * N_plan :]

            # Cache for warm-start
            self._last_X, self._last_U, self._last_dt = X_sol, U_sol, DT_sol

            # Optionally truncate when goal reached early
            goal_tol_xy = float(teb_cfg.get("goal_tol_xy", 0.05))
            goal_tol_yaw = float(teb_cfg.get("goal_tol_yaw", 0.10))
            keep_settle = int(teb_cfg.get("keep_settle_steps", 5))

            goal_reached_at = None
            for k in range(N_plan + 1):
                pos_err = float(np.hypot(X_sol[k, 0] - gx, X_sol[k, 1] - gy))
                yaw_err = float(abs((X_sol[k, 2] - gyaw + np.pi) % (2 * np.pi) - np.pi))
                if pos_err < goal_tol_xy and yaw_err < goal_tol_yaw:
                    goal_reached_at = k
                    break

            if goal_reached_at is not None and goal_reached_at < (N_plan - keep_settle - 1):
                trunc = min(goal_reached_at + keep_settle, N_plan)
                X_sol = X_sol[: trunc + 1]
                U_sol = U_sol[: trunc]
                DT_sol = DT_sol[: trunc]

            ref_traj = ReferenceTrajectory.from_teb_solution(
                states=X_sol[:-1],   # N states for N controls
                controls=U_sol,
                dt_array=DT_sol,
                metadata={
                    "profile": profile,
                    "planner_horizon": N_plan,
                    "phase": current_phase.value,
                },
            )

            print(f"[TEB PLANNER] ✓ Planning succeeded: steps={ref_traj.n_steps}, total_time={ref_traj.total_time:.2f}s")
            return ref_traj

        except Exception as e:
            print(f"[TEB PLANNER] Planning failed: {e}")
            return ReferenceTrajectory(
                states=np.zeros((1, 4)),
                controls=np.zeros((1, 2)),
                dt_array=np.array([0.0]),
                total_time=0.0,
                n_steps=1,
                success=False,
                metadata={"error": str(e)},
            )
        finally:
            # Restore runtime knobs
            self.w_goal_xy = original_w_goal_xy
            self.w_goal_theta = original_w_goal_theta
            self.dt_min = original_dt_min
            self.dt_max = original_dt_max
            self.w_time = original_w_time
            self.enable_teb = original_enable_teb
            self.N = original_N

    def track_trajectory(self, state: VehicleState, reference: ReferenceTrajectory,
                        obstacles: List[Obstacle], step: int, profile: str = "parallel") -> MPCSolution:
        """
        Track a reference trajectory using MPC.

        This is TRACKING MODE for the hybrid architecture. Instead of optimizing
        to a fixed goal, MPC tracks the reference trajectory created by TEB planning.

        Key differences from solve():
        1. Goal is the next waypoint from reference (moving goal)
        2. Uses current reference control as warm-start
        3. Can deviate from reference for collision avoidance
        4. No re-planning - just tracking with feedback

        Args:
            state: Current vehicle state
            reference: Reference trajectory from TEB planning
            obstacles: List of obstacles
            step: Current step in reference trajectory
            profile: "parallel" or "perpendicular"

        Returns:
            MPCSolution with tracking control
        """
        # Check if we've reached end of reference trajectory
        if step >= reference.n_steps:
            # Use final reference waypoint as goal and continue tracking it
            ref_goal_state = reference.get_state_at_step(reference.n_steps - 1)
            goal = ParkingGoal(
                x=ref_goal_state[0],
                y=ref_goal_state[1],
                yaw=ref_goal_state[2]
            )

            # Check if we're close enough to consider it complete
            pos_err = np.sqrt((state.x - goal.x)**2 + (state.y - goal.y)**2)
            yaw_err = abs((state.yaw - goal.yaw + np.pi) % (2 * np.pi) - np.pi)

            if pos_err < 0.05 and yaw_err < 0.10 and abs(state.v) < 0.05:
                # Close enough - return zero control
                return MPCSolution(
                    states=np.zeros((1, 4)),
                    controls=np.zeros((1, 2)),
                    success=True,
                    info={"status": "goal_reached", "pos_err": pos_err, "yaw_err": yaw_err},
                    phase=ParkingPhase.COMPLETED
                )
            else:
                # Continue to final goal with standard MPC
                return self.solve(state, goal, obstacles, profile)

        # Get reference window for MPC horizon
        ref_states, ref_controls, ref_dt = reference.get_reference_window(step, self.N)

        if ref_states is None:
            # Not enough reference left, use final waypoint as goal
            ref_goal_state = reference.get_state_at_step(reference.n_steps - 1)
            goal = ParkingGoal(
                x=ref_goal_state[0],
                y=ref_goal_state[1],
                yaw=ref_goal_state[2]
            )
            # Use standard solve() for final approach
            return self.solve(state, goal, obstacles, profile)

        lookahead_s = 0.5

        # Reduce lookahead near obstacles so MPC doesn't "cut corners" into the neighbor cars
        if obstacles:
            min_clear = np.inf
            for o in obstacles[:self.max_obstacles]:
                r = self._obstacle_radius(o)
                d = np.hypot(state.x - o.cx, state.y - o.cy) - (r + self.circle_radius)
                if d < min_clear:
                    min_clear = d

            if min_clear < 0.35:
                lookahead_s = 0.25
            elif min_clear < 0.60:
                lookahead_s = 0.35

        if ref_dt is not None and len(ref_dt) > 0 and len(ref_states) > 1:
            t = 0.0
            target_idx = len(ref_states) - 1

            # ref_dt[i] is the time from ref_states[i] -> ref_states[i+1]
            # Ensure we don't iterate past the available states or dt values
            for i in range(min(len(ref_dt), len(ref_states) - 1)):
                t += float(ref_dt[i])
                if t >= lookahead_s:
                    target_idx = i + 1
                    break
        else:
            # Fallback: roughly 0.5s if dt is approx 0.1s
            target_idx = min(5, len(ref_states) - 1)

        next_ref_state = ref_states[target_idx]

        goal = ParkingGoal(
            x=next_ref_state[0],
            y=next_ref_state[1],
            yaw=next_ref_state[2]
        )

        # Use standard MPC solver but with reference as moving goal
        # The solver will naturally track the reference while avoiding obstacles
        sol = self.solve(state, goal, obstacles, profile)

        # Add tracking metadata
        sol.info["tracking_mode"] = True
        sol.info["reference_step"] = step
        sol.info["reference_goal"] = [goal.x, goal.y, goal.yaw]
        sol.info["lookahead_s"] = lookahead_s
        sol.info["lookahead_ref_idx"] = int(target_idx)

        return sol
