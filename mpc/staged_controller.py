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

        self.reset()

    def reset(self):
        self.stage = "APPROACH"  # APPROACH -> WAIT -> PARK
        self.goal_C: Optional[ParkingGoal] = None
        self.goal_B: Optional[ParkingGoal] = None
        self._wait_remaining = 0
        self._profile: str = "parallel"
        self._info: Dict[str, Any] = {}

        self.approach_mpc.reset_warm_start()
        if self.parking_controller is not None:
            self.parking_controller.reset()

    def reset_episode(self, goal_C: ParkingGoal, profile: str):
        self._profile = profile
        self.goal_C = goal_C
        self.goal_B = self._compute_goal_B(goal_C, profile)

        self.stage = "APPROACH"
        self._wait_remaining = 0

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
            sol = self.approach_mpc.solve(state, self.goal_B, obstacles, profile="approach")

            action = np.array([float(sol.controls[0, 0]), float(sol.controls[0, 1])], dtype=float)

            if self._in_pose_region(state, self.goal_B):
                self.stage = "WAIT"
                self._wait_remaining = self.wait_steps
                self.approach_mpc.reset_warm_start()
                if self.parking_controller is not None:
                    self.parking_controller.reset()

            self._info = {"mode": "A_to_B", "stage": self.stage}
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

    def _in_pose_region(self, state: VehicleState, goal: ParkingGoal) -> bool:
        dx = float(state.x - goal.x)
        dy = float(state.y - goal.y)
        pos_err = float(np.hypot(dx, dy))
        yaw_err = abs(_wrap_to_pi(float(state.yaw - goal.yaw)))
        return (pos_err < self.pos_tol) and (yaw_err < self.yaw_tol)

    # ----------------- helpers -----------------
    def _reached_pose(self, state: VehicleState, goal: ParkingGoal) -> bool:
        dx = float(state.x - goal.x)
        dy = float(state.y - goal.y)
        pos_err = float(np.hypot(dx, dy))
        yaw_err = abs(_wrap_to_pi(float(state.yaw - goal.yaw)))
        return (pos_err < self.pos_tol) and (yaw_err < self.yaw_tol) and (abs(float(state.v)) < self.vel_tol)

    def _compute_goal_B(self, goal_C: ParkingGoal, profile: str) -> ParkingGoal:
        scenarios = (self.env_cfg.get("scenarios", {}) or {})
        if isinstance(scenarios, dict) and profile in scenarios:
            scen = scenarios.get(profile, {}) or {}
        else:
            # env_cfg might already be scenario-only (parallel/perpendicular dict)
            scen = self.env_cfg or {}

        # Preferred: explicit fixed approach_pose in config
        ap = scen.get("approach_pose", None)
        if isinstance(ap, dict) and ("y" in ap) and ("yaw" in ap):
            x_off = float(ap.get("x_offset", 0.0))
            return ParkingGoal(
                x=float(goal_C.x + x_off),
                y=float(ap["y"]),
                yaw=float(ap["yaw"]),
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