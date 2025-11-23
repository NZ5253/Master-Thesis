# mpc_solver.py
# Simple nonlinear MPC for kinematic bicycle using CasADi.
# Install casadi in your environment: pip install casadi

try:
    import casadi as ca
except Exception:
    ca = None

import numpy as np


class MPCSolver:
    def __init__(self, vehicle_params, mpc_params):
        """
        vehicle_params : dict from env config_env.yaml["vehicle"]
        mpc_params     : either the whole config_mpc dict or just the 'mpc' subdict
        """
        # allow passing either whole config or just mpc subdict
        params = mpc_params.get("mpc", mpc_params)

        self.dt = float(params.get("dt", 0.1))
        self.N = int(params.get("N", params.get("horizon_steps", 30)))
        self.vehicle = vehicle_params
        self.solver = None

        # dimensions
        self.n_states = 4   # [x, y, yaw, v]
        self.n_controls = 2 # [steer, accel]
        self.n_g = 0        # number of equality constraints

        if ca is not None:
            self._build_solver()

    # ------------------------------------------------------------------
    # Internal model & NLP construction
    # ------------------------------------------------------------------
    def _build_solver(self):
        L = float(self.vehicle.get("wheelbase", 0.05))
        N = self.N
        dt = self.dt

        # state & control symbols
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

        # decision variables over horizon
        X = ca.SX.sym("X", n_states, N + 1)
        U = ca.SX.sym("U", n_controls, N)

        # parameters: initial state + goal state (4 + 4)
        P = ca.SX.sym("P", n_states + n_states)

        obj = 0
        g = []

        # simple quadratic weights (can tune later)
        Q = ca.diag(ca.SX([10.0, 10.0, 5.0, 0.1]))  # state error
        R = ca.diag(ca.SX([1.0, 0.1]))              # control effort

        # initial condition constraint: X[:,0] == x0
        g += [
            X[0, 0] - P[0],
            X[1, 0] - P[1],
            X[2, 0] - P[2],
            X[3, 0] - P[3],
        ]

        for k in range(N):
            st = X[:, k]
            con = U[:, k]
            goal = P[n_states:]  # [gx, gy, gyaw, gv]

            # cost: goal tracking + control effort
            obj += ca.mtimes([(st - goal).T, Q, (st - goal)]) \
                   + ca.mtimes([con.T, R, con])

            # kinematic bicycle dynamics (Euler)
            x_next = st[0] + st[3] * ca.cos(st[2]) * dt
            y_next = st[1] + st[3] * ca.sin(st[2]) * dt
            yaw_next = st[2] + (st[3] / L) * ca.tan(con[0]) * dt
            v_next = st[3] + con[1] * dt

            g += [
                X[0, k + 1] - x_next,
                X[1, k + 1] - y_next,
                X[2, k + 1] - yaw_next,
                X[3, k + 1] - v_next,
            ]

        self.n_states = n_states
        self.n_controls = n_controls
        self.n_g = len(g)

        OPT_variables = ca.vertcat(ca.reshape(X, -1, 1),
                                   ca.reshape(U, -1, 1))
        nlp_prob = {
            "f": obj,
            "x": OPT_variables,
            "g": ca.vertcat(*g),
            "p": P,
        }
        opts = {
            "ipopt.print_level": 0,
            "print_time": 0,
        }
        self.solver = ca.nlpsol("solver", "ipopt", nlp_prob, opts)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def solve(self, x0, goal):
        """
        Solve MPC for:
          x0   : [x, y, yaw, v]
          goal : [gx, gy, gyaw] or [gx, gy, gyaw, gv]
        """
        if self.solver is None:
            raise RuntimeError("CasADi not available - MPC solver not built.")

        x0 = np.asarray(x0, dtype=float).flatten()
        if x0.size != 4:
            raise ValueError("x0 must have length 4 [x, y, yaw, v].")

        goal = np.asarray(goal, dtype=float).flatten()
        if goal.size == 3:
            gx, gy, gyaw = goal
            goal = np.array([gx, gy, gyaw, 0.0], dtype=float)
        elif goal.size != 4:
            raise ValueError("goal must have length 3 or 4 [gx, gy, gyaw, (gv)].")

        # parameters: [x0, goal]
        p = np.concatenate([x0, goal])

        # initial guess: repeat x0 and zero controls
        X0 = np.tile(x0.reshape(-1, 1), (1, self.N + 1))
        U0 = np.zeros((self.n_controls, self.N), dtype=float)
        x_init = np.concatenate([X0.flatten(), U0.flatten()])

        zeros_g = np.zeros(self.n_g, dtype=float)

        res = self.solver(
            x0=x_init,
            p=p,
            lbg=zeros_g,
            ubg=zeros_g,
        )
        return res

    def extract_first_control(self, res):
        """
        Extract the first control [steer, accel] from the solver result.
        """
        x_opt = np.array(res["x"]).flatten()
        nX = self.n_states * (self.N + 1)
        U_flat = x_opt[nX:]
        U = U_flat.reshape(self.n_controls, self.N)
        u0 = U[:, 0]
        return float(u0[0]), float(u0[1])
