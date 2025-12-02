# mpc/teb_mpc.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Literal
import os
import numpy as np
import yaml

try:
    import casadi as ca
except Exception:
    ca = None

ProfileType = Literal["parallel", "perpendicular"]


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
    cx: float
    cy: float
    hx: float
    hy: float


@dataclass
class MPCSolution:
    states: np.ndarray
    controls: np.ndarray
    success: bool
    info: Dict[str, Any]


class TEBMPC:
    def __init__(self, config_path: Optional[str] = None, max_obstacles: int = 25) -> None:
        if config_path is None:
            base_dir = os.path.dirname(os.path.dirname(__file__))
            config_path = os.path.join(base_dir, "mpc", "config_mpc.yaml")

        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        self.vehicle_cfg = cfg.get("vehicle", {})
        self.mpc_cfg = cfg.get("mpc", cfg)
        self.env_cfg = cfg.get("environment", {})

        self.dt = float(self.mpc_cfg.get("dt", 0.1))
        self.N = int(self.mpc_cfg.get("horizon_steps", 50))
        self.max_obstacles = max_obstacles

        # --- LOAD DEFAULT WEIGHTS ---
        print("\n[MPC] Loading Default Weights:")
        self.w_goal_xy = float(self.mpc_cfg.get("w_goal_xy", 80.0))
        self.w_goal_theta = float(self.mpc_cfg.get("w_goal_theta", 50.0))
        self.w_goal_v = float(self.mpc_cfg.get("w_goal_v", 1.0))
        self.w_steer = float(self.mpc_cfg.get("w_steer", 1.0))
        self.w_accel = float(self.mpc_cfg.get("w_accel", 0.1))
        self.w_smooth_steer = float(self.mpc_cfg.get("w_smooth_steer", 0.05))
        self.w_smooth_accel = float(self.mpc_cfg.get("w_smooth_accel", 0.02))
        self.w_collision = float(self.mpc_cfg.get("w_collision", 5.0))
        self.w_boundary = float(self.mpc_cfg.get("w_boundary", 30.0))
        self.w_reverse_penalty = float(self.mpc_cfg.get("w_reverse_penalty", 0.1))

        print(f"  -> Collision: {self.w_collision}")
        print(f"  -> Reverse Penalty: {self.w_reverse_penalty}")

        # World bounds
        self.world_w = float(self.env_cfg.get("size_x", 4.0))
        self.world_h = float(self.env_cfg.get("size_y", 4.0))
        self.x_min, self.x_max = -self.world_w / 2, self.world_w / 2
        self.y_min, self.y_max = -self.world_h / 2, self.world_h / 2
        self.boundary_margin = 0.2

        # Geometry
        self.length = float(self.vehicle_cfg.get("length", 0.36))
        self.width = float(self.vehicle_cfg.get("width", 0.26))
        self.circle_radius = self.width / 4.0

        # --- UPDATED: 4-CIRCLE MODEL ---
        # Calculate 4 offsets relative to the rear axle
        # dist_ra_to_center aligns the geometric center relative to the rear axle
        dist_ra_to_center = self.length / 2.0 - 0.05
        step = self.length / 8.0

        # 4 circles distributed evenly along the spine
        self.offsets = [
            dist_ra_to_center + 3 * step,  # Front
            dist_ra_to_center + 1 * step,
            dist_ra_to_center - 1 * step,
            dist_ra_to_center - 3 * step  # Rear
        ]

        self.solver = None
        self._last_X, self._last_U = None, None
        self._current_profile = None

        if ca is not None: self._build_solver()

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

        if "w_goal_xy" in p_cfg: self.w_goal_xy = float(p_cfg["w_goal_xy"])
        if "w_goal_theta" in p_cfg: self.w_goal_theta = float(p_cfg["w_goal_theta"])
        if "w_collision" in p_cfg: self.w_collision = float(p_cfg["w_collision"])
        if "w_steer" in p_cfg: self.w_steer = float(p_cfg["w_steer"])
        if "w_smooth_steer" in p_cfg: self.w_smooth_steer = float(p_cfg["w_smooth_steer"])
        if "w_reverse_penalty" in p_cfg: self.w_reverse_penalty = float(p_cfg["w_reverse_penalty"])

        print(f"  -> w_goal_xy: {self.w_goal_xy}")
        print(f"  -> w_goal_theta: {self.w_goal_theta}")
        print(f"  -> w_collision: {self.w_collision}")
        print(f"  -> w_steer: {self.w_steer}")
        print(f"  -> w_smooth_steer: {self.w_smooth_steer}")
        print(f"  -> w_reverse_penalty: {self.w_reverse_penalty}")

    def _build_solver(self) -> None:
        N, dt, L = self.N, self.dt, float(self.vehicle_cfg.get("wheelbase", 0.25))

        x, y, yaw, v = ca.SX.sym("x"), ca.SX.sym("y"), ca.SX.sym("yaw"), ca.SX.sym("v")
        steer, accel = ca.SX.sym("steer"), ca.SX.sym("accel")

        X = ca.SX.sym("X", 4, N + 1)
        U = ca.SX.sym("U", 2, N)
        P = ca.SX.sym("P", 4 + 3 + 3 * self.max_obstacles)

        goal_x, goal_y, goal_yaw = P[4], P[5], P[6]
        obs_cx, obs_cy, obs_r = [], [], []
        for j in range(self.max_obstacles):
            base = 7 + 3 * j
            obs_cx.append(P[base])
            obs_cy.append(P[base + 1])
            obs_r.append(P[base + 2])

        obj = 0
        g = [X[:, 0] - P[0:4]]

        def angle_wrap(a):
            return ca.atan2(ca.sin(a), ca.cos(a))

        for k in range(N):
            st = X[:, k]
            u = U[:, k]

            yaw_err = angle_wrap(st[2] - goal_yaw)
            gain = 8.0 if k == N - 1 else 1.0

            # Goal Cost
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

            if k > 0:
                obj += self.w_smooth_steer * (u[0] - U[0, k - 1]) ** 2 + self.w_smooth_accel * (u[1] - U[1, k - 1]) ** 2

            # --- UPDATED: 4-CIRCLE COLLISION CHECK ---
            c_theta = ca.cos(st[2])
            s_theta = ca.sin(st[2])

            # Loop over all 4 ego circles
            for offset in self.offsets:
                # Calculate circle center
                cx = st[0] + offset * c_theta
                cy = st[1] + offset * s_theta

                # Check against all obstacles
                for j in range(self.max_obstacles):
                    dist_sq = (cx - obs_cx[j]) ** 2 + (cy - obs_cy[j]) ** 2 + 1e-6
                    dist = ca.sqrt(dist_sq)
                    pen = (obs_r[j] + self.circle_radius) - dist
                    # Soft constraint
                    obj += self.w_collision * 0.5 * (ca.fmax(0, pen)) ** 2

            # Boundary
            for val, limit, margin in [(st[0], self.x_min, 1), (st[0], self.x_max, -1),
                                       (st[1], self.y_min, 1), (st[1], self.y_max, -1)]:
                dist_bound = (limit - val) * margin
                obj += self.w_boundary * 0.5 * (ca.fmax(0, self.boundary_margin - dist_bound)) ** 2

            # Dynamics
            x_next = st[0] + st[3] * ca.cos(st[2]) * dt
            y_next = st[1] + st[3] * ca.sin(st[2]) * dt
            yaw_next = st[2] + (st[3] / L) * ca.tan(u[0]) * dt
            v_next = st[3] + u[1] * dt
            g.append(X[:, k + 1] - ca.vertcat(x_next, y_next, yaw_next, v_next))

        nlp = {"x": ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1)), "f": obj, "g": ca.vertcat(*g), "p": P}
        opts = {"ipopt": {"max_iter": 200, "print_level": 0, "tol": 1e-3, "warm_start_init_point": "yes"},
                "print_time": 0}
        self.solver = ca.nlpsol("solver", "ipopt", nlp, opts)

    def solve(self, state: VehicleState, goal: ParkingGoal, obstacles: List[Obstacle],
              profile: str = "perpendicular") -> MPCSolution:
        if self.solver is None: return MPCSolution(np.zeros((1, 4)), np.zeros((1, 2)), False, {})

        self._apply_profile(profile)

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

        if self._last_X is not None:
            X0[:-1] = self._last_X[1:]
            X0[-1] = self._last_X[-1]
            U0[:-1] = self._last_U[1:]
            U0[-1] = self._last_U[-1]

        P = np.zeros(4 + 3 + 3 * self.max_obstacles)
        P[0:4] = x0
        P[4:7] = [gx, gy, gyaw]

        for j, o in enumerate(obstacles[:self.max_obstacles]):
            idx = 7 + 3 * j
            P[idx] = o.cx
            P[idx + 1] = o.cy

            # OLD
            # r = np.hypot(o.hx, o.hy)

            # NEW: smaller radius for cars, slightly larger for walls
            r = np.hypot(o.hx, o.hy)
            if max(o.hx, o.hy) > 1.0:  # world wall
                r = 0.30
            else:  # parked cars etc.
                r = 0.10  # <- key: lets you get deeper into the gap
            P[idx + 2] = r

        lbx = [-np.inf] * 4 * (self.N + 1) + [-0.52, -1.0] * self.N
        ubx = [np.inf] * 4 * (self.N + 1) + [0.52, 1.0] * self.N

        try:
            res = self.solver(x0=ca.vertcat(X0.flatten(), U0.flatten()), lbx=lbx, ubx=ubx, lbg=0, ubg=0, p=P)
            res_x = np.array(res["x"]).flatten()
            X_sol = res_x[:4 * (self.N + 1)].reshape(self.N + 1, 4)
            U_sol = res_x[4 * (self.N + 1):].reshape(self.N, 2)
            self._last_X, self._last_U = X_sol, U_sol

            return MPCSolution(X_sol, U_sol, True, {"termination": "success"})
        except Exception:
            return MPCSolution(X0, U0, False, {})

    def first_control(self, sol: MPCSolution) -> np.ndarray:
        return sol.controls[0] if len(sol.controls) > 0 else np.zeros(2)