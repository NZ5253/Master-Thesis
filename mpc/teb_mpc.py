# mpc/teb_mpc.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Literal, Union

import os
import numpy as np
import yaml

try:
    import casadi as ca
except Exception:
    ca = None

ProfileType = Literal["perpendicular"]


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
    # axis-aligned rectangle, world frame (center + half-sizes)
    cx: float
    cy: float
    hx: float
    hy: float


@dataclass
class MPCSolution:
    states: np.ndarray     # (N+1, 4) [x, y, yaw, v]
    controls: np.ndarray   # (N, 2)   [steer, accel]
    success: bool
    info: Dict[str, Any]


class TEBMPC:
    """
    TEB-style MPC wrapper with a CasADi-based optimisation.

    - States  : [x, y, yaw, v]
    - Controls: [steer, accel]
    - Obstacles are modelled as inflated circular potentials.
    """

    def __init__(self, config_path: Optional[str] = None, max_obstacles: int = 8) -> None:
        if config_path is None:
            base_dir = os.path.dirname(os.path.dirname(__file__))
            config_path = os.path.join(base_dir, "mpc", "config_mpc.yaml")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Could not find MPC config at {config_path}")

        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        self.vehicle_cfg = cfg.get("vehicle", {})
        self.env_cfg = cfg.get("environment", {})
        self.mpc_cfg = cfg.get("mpc", cfg)

        # basic horizon settings
        self.dt = float(self.mpc_cfg.get("dt", 0.1))
        self.N = int(self.mpc_cfg.get("horizon_steps", 25))

        # we support up to this many obstacles; extra ones are ignored
        self.max_obstacles = max_obstacles

        # ---- weights (incl. velocity + boundary) ----
        self.w_goal_xy = float(self.mpc_cfg.get("w_goal_xy", 60.0))
        self.w_goal_theta = float(self.mpc_cfg.get("w_goal_theta", 40.0))
        self.w_goal_v = float(self.mpc_cfg.get("w_goal_v", 1.0))

        self.w_steer = float(self.mpc_cfg.get("w_steer", 1.0))
        self.w_accel = float(self.mpc_cfg.get("w_accel", 0.1))
        self.w_smooth_steer = float(self.mpc_cfg.get("w_smooth_steer", 0.05))
        self.w_smooth_accel = float(self.mpc_cfg.get("w_smooth_accel", 0.02))

        self.w_collision = float(self.mpc_cfg.get("w_collision", 8.0))
        self.alpha_obs = float(self.mpc_cfg.get("alpha_obs", 3.0))

        # environment / safety sizes
        self.world_w = float(self.env_cfg.get("size_x", 4.0))
        self.world_h = float(self.env_cfg.get("size_y", 4.0))
        self.safety_margin = float(self.env_cfg.get("safety_margin", 0.05))

        # soft boundary penalty
        self.w_boundary = float(self.mpc_cfg.get("w_boundary", 30.0))
        self.boundary_margin = float(self.mpc_cfg.get("boundary_margin", 0.2))
        self.x_min = -self.world_w / 2.0
        self.x_max = self.world_w / 2.0
        self.y_min = -self.world_h / 2.0
        self.y_max = self.world_h / 2.0

        # approximate vehicle radius for obstacle inflation
        length = float(self.vehicle_cfg.get("length", 0.36))
        width = float(self.vehicle_cfg.get("width", 0.26))
        self.vehicle_radius = 0.5 * float(np.sqrt(length ** 2 + width ** 2))

        # solver meta
        self.solver: Optional[ca.Function] = None
        self.n_states = 4
        self.n_controls = 2
        self.n_g = 0
        self.n_p = 0

        # warm-start buffers
        self._last_X: Optional[np.ndarray] = None
        self._last_U: Optional[np.ndarray] = None

        if ca is not None:
            self._build_solver()
        else:
            print("[TEBMPC] Warning: casadi not available; solver will not run.")

    # ------------------------------------------------------------------
    # Build optimisation problem
    # ------------------------------------------------------------------
    def _build_solver(self) -> None:
        N = self.N
        dt = self.dt
        L = float(self.vehicle_cfg.get("wheelbase", 0.21))

        # world bounds for penalty
        x_min = self.x_min
        x_max = self.x_max
        y_min = self.y_min
        y_max = self.y_max
        margin = self.boundary_margin
        w_boundary = self.w_boundary

        # state and control symbols
        x = ca.SX.sym("x")
        y = ca.SX.sym("y")
        yaw = ca.SX.sym("yaw")
        v = ca.SX.sym("v")
        states = ca.vertcat(x, y, yaw, v)
        n_states = states.size1()

        steer = ca.SX.sym("steer")
        accel = ca.SX.sym("accel")
        controls = ca.vertcat(steer, accel)
        n_controls = controls.size1()

        # decision variables
        X = ca.SX.sym("X", n_states, N + 1)
        U = ca.SX.sym("U", n_controls, N)

        # parameters:
        # P[0:4]   = initial state x0
        # P[4:7]   = goal (x, y, yaw)
        # P[7:]    = obstacles: (cx, cy, r) * max_obstacles
        n_p = 4 + 3 + 3 * self.max_obstacles
        P = ca.SX.sym("P", n_p)

        self.n_states = n_states
        self.n_controls = n_controls
        self.n_p = n_p

        goal_x = P[4]
        goal_y = P[5]
        goal_yaw = P[6]

        # slice obstacle params from P
        obs_cx = []
        obs_cy = []
        obs_r = []
        for j in range(self.max_obstacles):
            base = 7 + 3 * j
            obs_cx.append(P[base + 0])
            obs_cy.append(P[base + 1])
            obs_r.append(P[base + 2])

        obj = 0
        g = []

        # initial state constraint: X[:,0] == x0
        for i in range(n_states):
            g.append(X[i, 0] - P[i])

        def angle_wrap(a):
            # differentiable angle wrapping
            return ca.atan2(ca.sin(a), ca.cos(a))

        for k in range(N):
            st = X[:, k]
            u = U[:, k]

            # -------- goal tracking (x,y,yaw,v) --------
            ex = st[0] - goal_x
            ey = st[1] - goal_y
            yaw_err = angle_wrap(st[2] - goal_yaw)
            v_err = st[3] - 0.0          # want v -> 0 at goal

            obj += self.w_goal_xy * (ex * ex + ey * ey)
            obj += self.w_goal_theta * (yaw_err * yaw_err)
            obj += self.w_goal_v * (v_err * v_err)

            # control effort
            obj += self.w_steer * (u[0] * u[0]) + self.w_accel * (u[1] * u[1])

            # smoothness in controls
            if k > 0:
                du_steer = U[0, k] - U[0, k - 1]
                du_acc = U[1, k] - U[1, k - 1]
                obj += self.w_smooth_steer * (du_steer * du_steer)
                obj += self.w_smooth_accel * (du_acc * du_acc)

            if k == N - 1:
                obj += 5.0 * self.w_goal_xy * (ex * ex + ey * ey)
                obj += 5.0 * self.w_goal_theta * (yaw_err * yaw_err)

            # obstacle soft costs (radial potentials)
            for j in range(self.max_obstacles):
                dx = st[0] - obs_cx[j]
                dy = st[1] - obs_cy[j]
                dist = ca.sqrt(dx * dx + dy * dy + 1e-6)
                d_eff = dist - obs_r[j]  # >0 outside inflated obstacle
                obj += self.w_collision * ca.exp(-self.alpha_obs * d_eff)

            # boundary soft costs: keep inside [x_min, x_max] × [y_min, y_max]
            left_violation   = ca.fmax(0, (x_min + margin) - st[0])
            right_violation  = ca.fmax(0, st[0] - (x_max - margin))
            bottom_violation = ca.fmax(0, (y_min + margin) - st[1])
            top_violation    = ca.fmax(0, st[1] - (y_max - margin))
            obj += w_boundary * (
                left_violation * left_violation +
                right_violation * right_violation +
                bottom_violation * bottom_violation +
                top_violation * top_violation
            )

            # dynamics constraints (kinematic bicycle)
            x_next = st[0] + st[3] * ca.cos(st[2]) * dt
            y_next = st[1] + st[3] * ca.sin(st[2]) * dt
            yaw_next = st[2] + (st[3] / L) * ca.tan(u[0]) * dt
            v_next = st[3] + u[1] * dt

            g.append(X[0, k + 1] - x_next)
            g.append(X[1, k + 1] - y_next)
            g.append(X[2, k + 1] - yaw_next)
            g.append(X[3, k + 1] - v_next)

        self.n_g = len(g)

        OPT_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        nlp = {
            "f": obj,
            "x": OPT_vars,
            "g": ca.vertcat(*g),
            "p": P,
        }
        opts = {
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.max_iter": int(self.mpc_cfg.get("max_iter", 30)),
            "ipopt.tol": 1e-3,
            "ipopt.acceptable_tol": 1e-2,
            "ipopt.sb": "yes",  # silent
            "ipopt.hessian_approximation": "limited-memory",
        }
        self.solver = ca.nlpsol("solver", "ipopt", nlp, opts)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _state_to_array(self, state: Union[VehicleState, np.ndarray, List[float]]) -> np.ndarray:
        if isinstance(state, VehicleState):
            arr = np.array([state.x, state.y, state.yaw, state.v], dtype=float)
        else:
            arr = np.asarray(state, dtype=float).flatten()
        if arr.size != 4:
            raise ValueError("state must have 4 elements [x, y, yaw, v].")
        return arr

    def _goal_to_obj(self, goal: Union[ParkingGoal, np.ndarray, List[float]]) -> ParkingGoal:
        if isinstance(goal, ParkingGoal):
            return goal
        arr = np.asarray(goal, dtype=float).flatten()
        if arr.size != 3:
            raise ValueError("goal must have 3 elements [x, y, yaw].")
        return ParkingGoal(x=float(arr[0]), y=float(arr[1]), yaw=float(arr[2]))

    def _inflate_obstacles(self, obstacles: List[Obstacle]) -> np.ndarray:
        """
        Convert a list of Obstacle to an array of shape (max_obstacles, 3)
        with [cx, cy, r_inflated].
        """
        data = np.zeros((self.max_obstacles, 3), dtype=float)

        for i, obs in enumerate(obstacles[: self.max_obstacles]):
            cx, cy, hx, hy = obs.cx, obs.cy, obs.hx, obs.hy
            base_radius = float(np.sqrt(hx ** 2 + hy ** 2))
            r = base_radius + self.vehicle_radius + self.safety_margin
            data[i, 0] = cx
            data[i, 1] = cy
            data[i, 2] = r

        # remaining obstacles placed far away so they have negligible effect
        for j in range(len(obstacles), self.max_obstacles):
            data[j, 0] = 1e3
            data[j, 1] = 1e3
            data[j, 2] = 0.0

        return data

    def _pack_params(self, x0: np.ndarray, goal: ParkingGoal, obstacles: List[Obstacle]) -> np.ndarray:
        obs_data = self._inflate_obstacles(obstacles)
        p = np.zeros(self.n_p, dtype=float)
        p[0:4] = x0
        p[4] = goal.x
        p[5] = goal.y
        p[6] = goal.yaw
        idx = 7
        for j in range(self.max_obstacles):
            p[idx + 0] = obs_data[j, 0]
            p[idx + 1] = obs_data[j, 1]
            p[idx + 2] = obs_data[j, 2]
            idx += 3
        return p

    def _initial_guess(self, x0: np.ndarray, goal: ParkingGoal) -> np.ndarray:
        xs = np.linspace(x0[0], goal.x, self.N + 1)
        ys = np.linspace(x0[1], goal.y, self.N + 1)
        yaws = np.linspace(x0[2], goal.yaw, self.N + 1)
        vs = np.linspace(x0[3], 0.0, self.N + 1)
        return np.stack([xs, ys, yaws, vs], axis=1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def solve(
        self,
        state: Union[VehicleState, np.ndarray, List[float]],
        goal: Union[ParkingGoal, np.ndarray, List[float]],
        obstacles: List[Obstacle],
        profile: ProfileType = "parallel",
    ) -> MPCSolution:
        """
        Compute TEB-MPC trajectory from state towards goal around obstacles.

        Returns full horizon; apply the first control in closed loop.
        """
        if self.solver is None:
            raise RuntimeError("CasADi solver not built (casadi missing?)")

        x0 = self._state_to_array(state)
        g = self._goal_to_obj(goal)
        p = self._pack_params(x0, g, obstacles)

        # -------- initial guess (warm-start) --------
        if self._last_X is None or self._last_U is None:
            X0 = self._initial_guess(x0, g)           # (N+1, 4)
            U0 = np.zeros((self.N, self.n_controls))  # (N, 2)
        else:
            X_prev = self._last_X
            U_prev = self._last_U

            if X_prev.shape != (self.N + 1, self.n_states) or U_prev.shape != (self.N, self.n_controls):
                X0 = self._initial_guess(x0, g)
                U0 = np.zeros((self.N, self.n_controls))
            else:
                X0 = np.zeros_like(X_prev)
                U0 = np.zeros_like(U_prev)

                # shift states: drop first, repeat last
                X0[:-1, :] = X_prev[1:, :]
                X0[-1, :] = X_prev[-1, :]

                # shift controls: drop first, zero last
                U0[:-1, :] = U_prev[1:, :]
                U0[-1, :] = 0.0

                # force first state to current x0
                X0[0, :] = x0

        x_init = np.concatenate([X0.flatten(), U0.flatten()])
        zeros_g = np.zeros(self.n_g, dtype=float)

        res = self.solver(
            x0=x_init,
            p=p,
            lbg=zeros_g,
            ubg=zeros_g,
        )

        x_opt = np.array(res["x"]).flatten()
        nX = self.n_states * (self.N + 1)
        X = x_opt[:nX].reshape(self.N + 1, self.n_states)
        U = x_opt[nX:].reshape(self.N, self.n_controls)

        # store for warm-start next time
        self._last_X = X
        self._last_U = U

        info: Dict[str, Any] = {
            "status": self.solver.stats().get("return_status", ""),
            "profile": profile,
        }

        return MPCSolution(states=X, controls=U, success=True, info=info)
