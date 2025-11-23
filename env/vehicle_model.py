import numpy as np

class KinematicBicycle:
    def __init__(self, params):
        self.L = params.get('wheelbase', 0.05)  # meters (CRC small)
        self.max_steer = params.get('max_steer', 0.6)  # radians
        self.max_accel = params.get('max_accel', 1.0)
        self.max_speed = params.get('max_speed', 1.0)
        self.max_steer_rate = params.get('max_steer_rate', 1.0)

    def step(self, state, action, dt=0.1):
        x, y, yaw, v = state
        steer, accel = action
        steer = float(steer)
        accel = float(accel)

        # clamp
        steer = np.clip(steer, -self.max_steer, self.max_steer)
        accel = np.clip(accel, -self.max_accel, self.max_accel)

        # simplest Euler integration
        v_next = np.clip(v + accel*dt, -self.max_speed, self.max_speed)
        x_next = x + v_next * np.cos(yaw) * dt
        y_next = y + v_next * np.sin(yaw) * dt
        if abs(np.cos(steer)) < 1e-6:
            yaw_next = yaw
        else:
            yaw_next = yaw + (v_next / max(1e-4, self.L)) * np.tan(steer) * dt

        return np.array([x_next, y_next, yaw_next, v_next])
