# mpc/staged_controller.py
from __future__ import annotations

import numpy as np
from typing import List, Optional, Dict, Any

from mpc.teb_mpc import TEBMPC, VehicleState, ParkingGoal, Obstacle
from mpc.hybrid_controller import HybridController


def _wrap_to_pi(a: float) -> float:
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


class StagedAtoBtoCController:
    """
    Stages:
      APPROACH: A->B using receding-horizon MPC (re-solve every step)
      WAIT:     brake/settle for a short time
      PARK:     B->C using HybridController (plan once + track) OR baseline MPC

    This keeps HybridController unchanged and just wraps it.
    """

    def __init__(
            self,
            env_cfg: dict,
            dt: float,
            config_path: str = "mpc/config_mpc.yaml",
            max_obstacles: int = 25,
            use_hybrid_for_parking: bool = True,
            wait_time_s: float = 0.5,
            pos_tol: float = 0.10,
            yaw_tol: float = 0.15,
            vel_tol: float = 0.06,
            B_pos_tol: float = 0.25,
            B_yaw_tol: float = 0.50,
            B_hold_steps: int = 7,
            B_vel_tol: float = 0.05,
    ):
        self.env_cfg = env_cfg
        self.dt = float(dt)

        # A->B controller: simplest stable approach (receding horizon)
        self.approach_mpc = TEBMPC(max_obstacles=max_obstacles, env_cfg=env_cfg)

        # B->C controller
        self.use_hybrid_for_parking = bool(use_hybrid_for_parking)
        self.parking_controller: Optional[HybridController] = None
        if self.use_hybrid_for_parking:
            self.parking_controller = HybridController(
                config_path=config_path,
                env_cfg=env_cfg,
                dt=self.dt,
            )

        self.wait_steps = max(0, int(round(float(wait_time_s) / self.dt)))
        self.pos_tol = float(pos_tol)
        self.yaw_tol = float(yaw_tol)
        self.vel_tol = float(vel_tol)
        self.B_pos_tol = float(B_pos_tol)
        self.B_yaw_tol = float(B_yaw_tol)
        self.B_hold_steps = int(B_hold_steps)
        self.B_vel_tol = float(B_vel_tol)

        self.reset()

    def reset(self):
        self.stage = "APPROACH"  # APPROACH -> WAIT -> PARK
        self.goal_C: Optional[ParkingGoal] = None
        self.goal_B: Optional[ParkingGoal] = None
        self._wait_remaining = 0
        self._profile: str = "parallel"

        # B-stage transition logic state
        self._tick = 0
        self._B_hold_counter = 0
        self._B_first_enter_step = None
        self._B_wait_trigger_step = None
        self._B_trigger_snapshot = None  # (pos_err, yaw_err, v, approach_profile)

        self._info: Dict[str, Any] = {}

        # Phase 2.2: Profile switching state
        self._current_approach_profile = "approach_aggressive"  # Start with aggressive
        self._prev_metric = None  # Renamed from _prev_pos_err (tracks pos + yaw metric)
        self._stuck_counter = 0  # Count steps with no improvement

        self.approach_mpc.reset_warm_start()
        if self.parking_controller is not None:
            self.parking_controller.reset()

    def reset_episode(self, goal_C: ParkingGoal, profile: str, bay_center: Optional[np.ndarray] = None):
        """
        Reset for new episode.

        Args:
            goal_C: Final parking goal
            profile: Scenario profile (parallel/perpendicular)
            bay_center: Optional [x, y, yaw] of bay center for Phase 4 random bay positions
        """
        self._profile = profile
        self.goal_C = goal_C
        self.goal_B = self._compute_goal_B(goal_C, profile, bay_center)

        self.stage = "APPROACH"
        self._wait_remaining = 0

        # Reset B-stage logic
        self._tick = 0
        self._B_hold_counter = 0
        self._B_first_enter_step = None
        self._B_wait_trigger_step = None
        self._B_trigger_snapshot = None

        # Reset Phase 2.2 approach logic state (IMPORTANT: per-episode)
        self._current_approach_profile = "approach_aggressive"
        self._prev_metric = None  # Renamed
        self._stuck_counter = 0

        self.approach_mpc.reset_warm_start()
        if self.parking_controller is not None:
            self.parking_controller.reset()

        self._info = {
            "stage": self.stage,
            "goal_B": (self.goal_B.x, self.goal_B.y, self.goal_B.yaw),
            "goal_C": (self.goal_C.x, self.goal_C.y, self.goal_C.yaw),
        }

    def get_info(self) -> Dict[str, Any]:
        out = dict(self._info)
        out["stage"] = self.stage
        out["mode"] = out.get("mode", "")
        if self.parking_controller is not None:
            out["parking_info"] = self.parking_controller.get_info()

        # Persist B-trigger stats across all stages
        out["B_first_enter_step"] = self._B_first_enter_step
        out["B_wait_trigger_step"] = self._B_wait_trigger_step

        if self._B_trigger_snapshot is not None:
            pe, ye, vv, prof = self._B_trigger_snapshot
            out.update({
                "B_pos_err_at_trigger": pe,
                "B_yaw_err_at_trigger": ye,
                "B_v_at_trigger": vv,
                "B_profile_at_trigger": prof,
            })

        return out

    def get_control(
            self,
            state: VehicleState,
            goal_C: ParkingGoal,
            obstacles: List[Obstacle],
            profile: str = "parallel",
    ) -> np.ndarray:
        # Safety: if episode wasn't initialized, initialize now
        if self.goal_C is None or self.goal_B is None or profile != self._profile:
            self.reset_episode(goal_C, profile)

        assert self.goal_C is not None and self.goal_B is not None

        # ----------------- APPROACH (A->B) -----------------
        if self.stage == "APPROACH":
            self._tick += 1

            # Phase 2.2: Calculate distance and yaw error to B
            dx = float(state.x - self.goal_B.x)
            dy = float(state.y - self.goal_B.y)
            pos_err = float(np.hypot(dx, dy))
            yaw_err = abs(_wrap_to_pi(state.yaw - self.goal_B.yaw))

            # Phase 2.2: Profile switching logic
            # Switch from aggressive -> settle when close to B
            if self._current_approach_profile == "approach_aggressive":
                # Threshold: within 50cm AND yaw < 15° -> switch to settle
                if pos_err < 0.50 and yaw_err < 0.26:  # 0.26 rad ≈ 15°
                    self._current_approach_profile = "approach_settle"
                    print(f"[STAGED] Switching to settle profile (pos_err={pos_err:.3f}m, yaw_err={yaw_err:.3f}rad)",
                          flush=True)

            # Phase 2.2: Anti-stuck guard (yaw-aware)
            # Near B, pos_err may not drop much while yaw is being corrected — don't reset there.
            metric = pos_err + 0.35 * yaw_err  # 1 rad ~ 0.35 m weight (works well here)
            if self._prev_metric is not None:
                improvement = self._prev_metric - metric

                # Only apply stuck logic when still "far-ish" from B (0.35m)
                if pos_err > 0.35:
                    if improvement < 0.01:
                        self._stuck_counter += 1
                        if self._stuck_counter >= 25:
                            print(f"[STAGED] Anti-stuck: resetting warm start (metric={metric:.3f})", flush=True)
                            self.approach_mpc.reset_warm_start()
                            self._stuck_counter = 0
                    else:
                        self._stuck_counter = 0
                else:
                    self._stuck_counter = 0  # disable near B
            self._prev_metric = metric

            # Solve MPC with current profile
            sol = self.approach_mpc.solve(state, self.goal_B, obstacles, profile=self._current_approach_profile)
            action = np.array([float(sol.controls[0, 0]), float(sol.controls[0, 1])], dtype=float)

            # --- B-Stage Hold Logic ---
            # Enter-B can be permissive, but Trigger-to-PARK must be tighter.
            in_B = (pos_err < self.B_pos_tol) and (yaw_err < self.B_yaw_tol)

            # Trigger gate (prevents handoff when yaw is still "open")
            # Your fail pkls often trigger at ~0.19-0.28 rad -> this blocks those.
            yaw_hold_tol = min(self.B_yaw_tol, 0.18)  # ~10.3 deg
            yaw_settle_tol = min(yaw_hold_tol, 0.12)  # ~6.9 deg (only then we "stop/settle")

            # Record first entry
            if in_B and self._B_first_enter_step is None:
                self._B_first_enter_step = self._tick

            # Hold gate: require low speed + consecutive steps + tighter yaw
            good_B = (pos_err < self.B_pos_tol) and (yaw_err < yaw_hold_tol) and (
                    abs(float(state.v)) < self.B_vel_tol
            )

            if good_B:
                self._B_hold_counter += 1
            else:
                self._B_hold_counter = 0

            # In-B behavior:
            # - If yaw is still not good enough, allow small motion (so yaw CAN improve),
            #   but cap speed/accel to prevent "forward escape" arcs.
            # - If yaw is good, then settle to near-zero speed.
            if in_B:
                steer_cmd = float(action[0])
                accel_cmd = float(action[1])
                v = float(state.v)

                max_acc = float((self.env_cfg.get("vehicle", {}) or {}).get("max_acc", 1.0))

                if yaw_err >= yaw_settle_tol:
                    # Yaw not ready: keep steering authority, but keep motion slow
                    v_cap = 0.08  # m/s (small but nonzero -> yaw can converge)

                    # If already too fast, brake toward v_cap
                    if abs(v) > v_cap:
                        accel_cmd = float(np.clip(-np.sign(v) * max_acc, -max_acc, max_acc))
                    else:
                        # Limit accel so we don't build speed while near obstacles
                        accel_cmd = float(np.clip(accel_cmd, -0.35 * max_acc, 0.35 * max_acc))
                else:
                    # Yaw good: settle speed to ~0 without killing steering
                    accel_stop = -np.sign(v) * 0.4
                    if v > self.B_vel_tol:
                        accel_cmd = min(accel_cmd, accel_stop)
                    elif v < -self.B_vel_tol:
                        accel_cmd = max(accel_cmd, accel_stop)
                    else:
                        accel_cmd = 0.0

                    accel_cmd = float(np.clip(accel_cmd, -max_acc, max_acc))

                action = np.array([steer_cmd, accel_cmd], dtype=float)

            # Trigger transition only after K consecutive good steps
            if self._B_hold_counter >= self.B_hold_steps:
                self.stage = "WAIT"
                self._wait_remaining = self.wait_steps
                self._B_wait_trigger_step = self._tick
                self._B_trigger_snapshot = (pos_err, yaw_err, float(state.v), self._current_approach_profile)

                self.approach_mpc.reset_warm_start()
                if self.parking_controller is not None:
                    self.parking_controller.reset()

            self._info = {
                "mode": "A_to_B",
                "stage": self.stage,
                "approach_profile": self._current_approach_profile,
                "pos_err_to_B": pos_err,
                "yaw_err_to_B": yaw_err,
                "B_first_enter_step": self._B_first_enter_step,
                "B_wait_trigger_step": self._B_wait_trigger_step,
                "B_hold_counter": self._B_hold_counter,
                "B_hold_steps": self.B_hold_steps,
                "B_vel_tol": self.B_vel_tol,
                "B_yaw_hold_tol": yaw_hold_tol,
                "B_yaw_settle_tol": yaw_settle_tol,
                "v": float(state.v),
            }
            if self._B_trigger_snapshot is not None:
                pe, ye, vv, prof = self._B_trigger_snapshot
                self._info.update({
                    "B_pos_err_at_trigger": pe,
                    "B_yaw_err_at_trigger": ye,
                    "B_v_at_trigger": vv,
                    "B_profile_at_trigger": prof,
                })

            return action

        # ----------------- WAIT -----------------
        if self.stage == "WAIT":
            v = float(state.v)
            if abs(v) > 0.03:
                # Accelerate opposite to velocity to stop (clipped to vehicle limits)
                max_acc = float((self.env_cfg.get("vehicle", {}) or {}).get("max_acc", 1.0))
                desired_accel = -v / max(self.dt, 1e-6)  # stop within ~1 step
                desired_accel = float(np.clip(desired_accel, -max_acc, max_acc))
                action = np.array([0.0, desired_accel], dtype=float)
            else:
                action = np.zeros(2, dtype=float)

            self._wait_remaining -= 1
            if (self._wait_remaining <= 0) and (abs(float(state.v)) < self.vel_tol):
                self.stage = "PARK"
                self.approach_mpc.reset_warm_start()
                if self.parking_controller is not None:
                    self.parking_controller.reset()

            self._info = {"mode": "wait", "stage": self.stage}
            return action

        # ----------------- PARK (B->C) -----------------
        if self.parking_controller is not None:
            action = self.parking_controller.get_control(state, self.goal_C, obstacles, profile=profile)
            self._info = {"mode": "B_to_C_hybrid", "stage": self.stage}
            return action

        # Fallback: baseline MPC replan to goal_C
        sol = self.approach_mpc.solve(state, self.goal_C, obstacles, profile=profile)
        action = np.array([float(sol.controls[0, 0]), float(sol.controls[0, 1])], dtype=float)
        self._info = {"mode": "B_to_C_baseline", "stage": self.stage}
        return action

    def _in_pose_region(self, state: VehicleState, goal: ParkingGoal,
                        pos_tol: Optional[float] = None,
                        yaw_tol: Optional[float] = None) -> bool:
        pos_tol = self.pos_tol if pos_tol is None else float(pos_tol)
        yaw_tol = self.yaw_tol if yaw_tol is None else float(yaw_tol)
        dx = float(state.x - goal.x)
        dy = float(state.y - goal.y)
        pos_err = float(np.hypot(dx, dy))
        yaw_err = abs(_wrap_to_pi(float(state.yaw - goal.yaw)))
        return (pos_err < pos_tol) and (yaw_err < yaw_tol)

    # ----------------- helpers -----------------
    def _reached_pose(self, state: VehicleState, goal: ParkingGoal) -> bool:
        dx = float(state.x - goal.x)
        dy = float(state.y - goal.y)
        pos_err = float(np.hypot(dx, dy))
        yaw_err = abs(_wrap_to_pi(float(state.yaw - goal.yaw)))
        return (pos_err < self.pos_tol) and (yaw_err < self.yaw_tol) and (abs(float(state.v)) < self.vel_tol)

    def _compute_goal_B(self, goal_C: ParkingGoal, profile: str,
                        bay_center: Optional[np.ndarray] = None) -> ParkingGoal:
        scenarios = (self.env_cfg.get("scenarios", {}) or {})
        if isinstance(scenarios, dict) and profile in scenarios:
            scen = scenarios.get(profile, {}) or {}
        else:
            # env_cfg might already be scenario-only (parallel/perpendicular dict)
            scen = self.env_cfg or {}

        # Preferred: explicit approach_pose in config
        ap = scen.get("approach_pose", None)
        if isinstance(ap, dict):
            x_off = float(ap.get("x_offset", 0.0))
            yaw_B = float(ap.get("yaw", 0.0))

            # Phase 4: Check if we have bay_center and y_offset (relative positioning)
            if bay_center is not None and "y_offset_from_bay" in ap:
                bay_y = float(bay_center[1])
                y_offset = float(ap["y_offset_from_bay"])
                y_B = bay_y + y_offset
            # Legacy: absolute Y position
            elif "y" in ap:
                y_B = float(ap["y"])
            else:
                # Fallback
                y_B = float(goal_C.y)

            return ParkingGoal(
                x=float(goal_C.x + x_off),
                y=y_B,
                yaw=yaw_B,
            )

        # Fallback: derive from spawn_lane midpoints (works, but B moves if spawn ranges change)
        spawn = scen.get("spawn_lane", {}) or {}
        y_min = float(spawn.get("y_min", goal_C.y))
        y_max = float(spawn.get("y_max", y_min))
        yaw = float(spawn.get("yaw", goal_C.yaw))

        x_min_off = float(spawn.get("x_min_offset", 0.0))
        x_max_off = float(spawn.get("x_max_offset", x_min_off))
        x_off = 0.5 * (x_min_off + x_max_off)

        return ParkingGoal(x=float(goal_C.x + x_off), y=float(0.5 * (y_min + y_max)), yaw=yaw)