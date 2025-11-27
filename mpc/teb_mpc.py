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
    x: float;
    y: float;
    yaw: float;
    v: float


@dataclass
class ParkingGoal:
    x: float;
    y: float;
    yaw: float


@dataclass
class Obstacle:
    cx: float;
    cy: float;
    hx: float;
    hy: float


@dataclass
class MPCSolution:
    states: np.ndarray
    controls: np.ndarray
    success: bool
    info: Dict[str, Any]


class TEBMPC:
    def __init__(self, config_path: Optional[str] = None, max_obstacles: int = 8) -> None:
        if config_path is None:
            base_dir = os.path.dirname(os.path.dirname(__file__))
            config_path = os.path.join(base_dir, "mpc", "config_mpc.yaml")

        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        self.vehicle_cfg = cfg.get("vehicle", {})
        self.mpc_cfg = cfg.get("mpc", cfg)
        self.env_cfg = cfg.get("environment", {})

        self.dt = float(self.mpc_cfg.get("dt", 0.1))
        self.N = int(self.mpc_cfg.get("horizon_steps", 25))
        self.max_obstacles = max_obstacles

        # Weights
        self.w_goal_xy = float(self.mpc_cfg.get("w_goal_xy", 80.0))
        self.w_goal_theta = float(self.mpc_cfg.get("w_goal_theta", 50.0))
        self.w_goal_v = float(self.mpc_cfg.get("w_goal_v", 1.0))
        self.w_steer = float(self.mpc_cfg.get("w_steer", 1.0))
        self.w_accel = float(self.mpc_cfg.get("w_accel", 0.1))
        self.w_smooth_steer = float(self.mpc_cfg.get("w_smooth_steer", 0.05))
        self.w_smooth_accel = float(self.mpc_cfg.get("w_smooth_accel", 0.02))
        self.w_collision = float(self.mpc_cfg.get("w_collision", 5.0))
        self.w_boundary = float(self.mpc_cfg.get("w_boundary", 30.0))

        # World bounds
        self.world_w = float(self.env_cfg.get("size_x", 4.0))
        self.world_h = float(self.env_cfg.get("size_y", 4.0))
        self.x_min, self.x_max = -self.world_w / 2, self.world_w / 2
        self.y_min, self.y_max = -self.world_h / 2, self.world_h / 2
        self.boundary_margin = 0.2

        # --- MULTI-CIRCLE GEOMETRY ---
        # No more hacks. We use real dimensions.
        self.length = float(self.vehicle_cfg.get("length", 0.36))
        self.width = float(self.vehicle_cfg.get("width", 0.26))

        # We place 2 circles along the spine.
        # Radius covers the width perfectly.
        self.circle_radius = self.width / 2.0  # 0.13m

        # Offset from center (rear axle approx) to the two circles.
        # Ideally, we want to cover the length 0.36.
        # Circle 1 at +offset, Circle 2 at -offset relative to Geometry Center.
        # But our state (x,y) is the Rear Axle.
        # Rear Axle is ~0.05m from back. Center is ~0.13m ahead of Rear Axle.
        dist_ra_to_center = self.length / 2.0 - 0.05

        # We place circles +/- 0.09m from the geometric center
        spacing = self.length / 4.0
        self.offset_front = dist_ra_to_center + spacing
        self.offset_rear = dist_ra_to_center - spacing

        self.solver = None
        self._last_X, self._last_U = None, None

        if ca is not None: self._build_solver()

    def _build_solver(self) -> None:
        N, dt, L = self.N, self.dt, float(self.vehicle_cfg.get("wheelbase", 0.25))

        # Symbols
        x, y, yaw, v = ca.SX.sym("x"), ca.SX.sym("y"), ca.SX.sym("yaw"), ca.SX.sym("v")
        states = ca.vertcat(x, y, yaw, v)
        steer, accel = ca.SX.sym("steer"), ca.SX.sym("accel")
        controls = ca.vertcat(steer, accel)

        X = ca.SX.sym("X", 4, N + 1)
        U = ca.SX.sym("U", 2, N)
        # P: [x0(4), goal(3), obstacles(3*max_obs)]
        P = ca.SX.sym("P", 4 + 3 + 3 * self.max_obstacles)

        # Unpack Parameters
        goal_x, goal_y, goal_yaw = P[4], P[5], P[6]
        obs_cx, obs_cy, obs_r = [], [], []
        for j in range(self.max_obstacles):
            base = 7 + 3 * j
            obs_cx.append(P[base]);
            obs_cy.append(P[base + 1]);
            obs_r.append(P[base + 2])

        obj = 0
        g = []
        g.append(X[:, 0] - P[0:4])  # Initial state constraint

        def angle_wrap(a):
            return ca.atan2(ca.sin(a), ca.cos(a))

        for k in range(N):
            st = X[:, k];
            u = U[:, k]

            # 1. Goal Cost
            yaw_err = angle_wrap(st[2] - goal_yaw)
            # Higher weight at terminal step
            gain = 5.0 if k == N - 1 else 1.0
            obj += gain * (
                    self.w_goal_xy * ((st[0] - goal_x) ** 2 + (st[1] - goal_y) ** 2) +
                    self.w_goal_theta * yaw_err ** 2 +
                    self.w_goal_v * st[3] ** 2
            )

            # 2. Control Cost
            obj += self.w_steer * u[0] ** 2 + self.w_accel * u[1] ** 2
            if k > 0:
                obj += self.w_smooth_steer * (u[0] - U[0, k - 1]) ** 2 + self.w_smooth_accel * (u[1] - U[1, k - 1]) ** 2

            # 3. ADVANCED COLLISION AVOIDANCE (Multi-Circle)
            # Calculate positions of the two circles
            # Circle 1 (Front-ish)
            c1_x = st[0] + self.offset_front * ca.cos(st[2])
            c1_y = st[1] + self.offset_front * ca.sin(st[2])

            # Circle 2 (Rear-ish)
            c2_x = st[0] + self.offset_rear * ca.cos(st[2])
            c2_y = st[1] + self.offset_rear * ca.sin(st[2])

            for j in range(self.max_obstacles):
                # Distance to Front Circle
                dist1 = ca.sqrt((c1_x - obs_cx[j]) ** 2 + (c1_y - obs_cy[j]) ** 2 + 1e-6)
                pen1 = (obs_r[j] + self.circle_radius) - dist1
                # Squared Hinge Loss
                obj += self.w_collision * 0.5 * (ca.fmax(0, pen1)) ** 2

                # Distance to Rear Circle
                dist2 = ca.sqrt((c2_x - obs_cx[j]) ** 2 + (c2_y - obs_cy[j]) ** 2 + 1e-6)
                pen2 = (obs_r[j] + self.circle_radius) - dist2
                obj += self.w_collision * 0.5 * (ca.fmax(0, pen2)) ** 2

            # 4. Boundary Cost
            for val, limit, margin in [(st[0], self.x_min, 1), (st[0], self.x_max, -1),
                                       (st[1], self.y_min, 1), (st[1], self.y_max, -1)]:
                # Penalize if val < x_min + margin  OR  val > x_max - margin
                # Generalized: (limit - val)*sign < margin
                dist_bound = (limit - val) * margin
                # if inside margin, penalize
                pen = self.boundary_margin - dist_bound
                # Only penalize if close to wall (pen > 0) AND actually inside box (dist_bound > 0 approx)
                # Actually simpler: cost = exp(-dist) or max(0, margin - dist)^2
                # Here we check distance from edge
                dist_from_edge = ca.fabs(val - limit)
                pen_bound = self.boundary_margin - dist_from_edge
                obj += self.w_boundary * 0.5 * (ca.fmax(0, pen_bound)) ** 2

            # 5. Dynamics
            beta = ca.atan(ca.tan(u[0]) / 2.0)  # CoG slip angle approximation
            x_next = st[0] + st[3] * ca.cos(st[2] + beta) * dt
            y_next = st[1] + st[3] * ca.sin(st[2] + beta) * dt
            yaw_next = st[2] + (st[3] / L) * ca.sin(beta) * dt  # CoG dynamics
            # Note: For low speed parking, simple rear-axle kinematics (beta=0) is also fine
            # but this is slightly more accurate if we defined CoG.
            # Let's stick to the rear-axle model you used in sim to match exactly:
            # x_next_simple = st[0] + st[3] * ca.cos(st[2]) * dt
            # y_next_simple = st[1] + st[3] * ca.sin(st[2]) * dt
            # yaw_next_simple = st[2] + (st[3] / L) * ca.tan(u[0]) * dt

            # Using your Rear-Axle model to match simulation:
            x_next_ra = st[0] + st[3] * ca.cos(st[2]) * dt
            y_next_ra = st[1] + st[3] * ca.sin(st[2]) * dt
            yaw_next_ra = st[2] + (st[3] / L) * ca.tan(u[0]) * dt
            v_next_ra = st[3] + u[1] * dt

            g.append(X[:, k + 1] - ca.vertcat(x_next_ra, y_next_ra, yaw_next_ra, v_next_ra))

        nlp = {"x": ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1)), "f": obj, "g": ca.vertcat(*g), "p": P}
        opts = {"ipopt": {"max_iter": 30, "print_level": 0, "tol": 1e-3}, "print_time": 0}
        self.solver = ca.nlpsol("solver", "ipopt", nlp, opts)

    def solve(self, state: VehicleState, goal: ParkingGoal, obstacles: List[Obstacle],
              profile: str = "perpendicular") -> MPCSolution:
        if self.solver is None: return MPCSolution(np.zeros((1, 4)), np.zeros((1, 2)), False, {})

        # Initial Guess (Smart Yaw Unwrap)
        x0 = np.array([state.x, state.y, state.yaw, state.v])
        gx, gy, gyaw = goal.x, goal.y, goal.yaw

        # Unwrap yaw
        diff = (gyaw - state.yaw + np.pi) % (2 * np.pi) - np.pi
        unwound_gyaw = state.yaw + diff

        X0 = np.zeros((self.N + 1, 4))
        for k in range(self.N + 1):
            alpha = k / self.N
            X0[k] = (1 - alpha) * x0 + alpha * np.array([gx, gy, unwound_gyaw, 0])
        U0 = np.zeros((self.N, 2))
        if self._last_X is not None: X0, U0 = self._last_X, self._last_U

        # Pack Params
        P = np.zeros(4 + 3 + 3 * self.max_obstacles)
        P[0:4] = x0
        P[4:7] = [gx, gy, gyaw]

        for j, o in enumerate(obstacles[:self.max_obstacles]):
            idx = 7 + 3 * j
            P[idx] = o.cx
            P[idx + 1] = o.cy
            # CRITICAL: Use min dimension for obstacle radius to allow tight fit
            # We assume the obstacle is roughly rectangular/circular.
            # Using the inscribed radius (min dimension) prevents "fat" obstacles.
            P[idx + 2] = min(o.hx, o.hy)

            # Bounds
        lbx = [-np.inf] * 4 * (self.N + 1) + [-0.52, -1.0] * self.N  # steer 30deg, acc 1.0
        ubx = [np.inf] * 4 * (self.N + 1) + [0.52, 1.0] * self.N

        try:
            res = self.solver(x0=ca.vertcat(X0.flatten(), U0.flatten()), lbx=lbx, ubx=ubx, lbg=0, ubg=0, p=P)
            res_x = np.array(res["x"]).flatten()
            X_sol = res_x[:4 * (self.N + 1)].reshape(self.N + 1, 4)
            U_sol = res_x[4 * (self.N + 1):].reshape(self.N, 2)
            self._last_X, self._last_U = X_sol, U_sol
            return MPCSolution(X_sol, U_sol, True, {"termination": "success"})
        except Exception as e:
            print(f"Solver failed: {e}")
            return MPCSolution(X0, U0, False, {})

    def first_control(self, sol: MPCSolution) -> np.ndarray:
        return sol.controls[0] if len(sol.controls) > 0 else np.zeros(2)