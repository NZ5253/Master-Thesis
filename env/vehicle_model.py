import numpy as np


class KinematicBicycle:
    def __init__(self, params):
        # Geometry
        # Use your config wheelbase (0.25) instead of the old 0.05 default
        self.L = params.get("wheelbase", 0.25)  # [m]

        # Steering limits
        self.max_steer = params.get("max_steer", 0.523)       # [rad]
        self.max_steer_rate = params.get("max_steer_rate", 0.5)  # [rad/s]

        # Speed / accel:
        # Your config uses max_vel / max_acc, so we prioritise those,
        # but still support older names max_speed / max_accel as a fallback.
        self.max_speed = params.get("max_vel", params.get("max_speed", 1.0))
        self.max_accel = params.get("max_acc", params.get("max_accel", 1.0))

    def step(self, state, action, dt=0.1):
        """
        Simple kinematic bicycle forward integration.

        state  = [x, y, yaw, v]
        action = [steer, accel]
        """
        x, y, yaw, v = state
        steer, accel = action
        steer = float(steer)
        accel = float(accel)

        # Clamp inputs
        steer = np.clip(steer, -self.max_steer, self.max_steer)
        accel = np.clip(accel, -self.max_accel, self.max_accel)

        # Integrate
        v_next = np.clip(v + accel * dt, -self.max_speed, self.max_speed)
        x_next = x + v_next * np.cos(yaw) * dt
        y_next = y + v_next * np.sin(yaw) * dt

        if abs(np.cos(steer)) < 1e-6:
            yaw_next = yaw
        else:
            yaw_next = yaw + (v_next / max(1e-4, self.L)) * np.tan(steer) * dt

        return np.array([x_next, y_next, yaw_next, v_next])
