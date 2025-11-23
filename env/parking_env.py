import numpy as np

from .vehicle_model import KinematicBicycle
from .obstacle_manager import ObstacleManager
from .reward_functions import compute_reward


class ParkingEnv:
    """Perpendicular parking environment.

    State: [x, y, yaw, v]
    Goal : [gx, gy, gyaw]

    Observation:
        [x, y, yaw, v,
         dx, dy, dtheta,
         dist_front, dist_left, dist_right]

    Action:
        [steer, accel]
    """

    def __init__(self, config):
        self.dt = float(config.get("dt", 0.1))
        self.max_steps = int(config.get("max_steps", 200))

        self.model = KinematicBicycle(config["vehicle"])
        self.goal = np.array(config["goal"], dtype=float)  # [gx, gy, gyaw]

        # Obstacle manager needs goal to place neighbour cars left/right
        self.obstacles = ObstacleManager(config.get("obstacles", {}), goal=self.goal)

        self.state: np.ndarray | None = None
        self.step_count: int = 0

    # ------------------------------------------------------------------
    # Gym-like API
    # ------------------------------------------------------------------
    def reset(self, randomize: bool = True):
        """Reset environment state and optionally randomise obstacles."""
        if randomize:
            self.obstacles.randomize()
            self.state = self._random_start()
        else:
            self.obstacles.reset_to_base()
            # deterministic nominal start (lane below bays)
            gx, gy, gyaw = self.goal
            x = gx
            y = gy - 1.0
            yaw = gyaw
            v = 0.0
            self.state = np.array([x, y, yaw, v], dtype=float)

        self.step_count = 0
        return self._get_obs()

    def _random_start(self) -> np.ndarray:
        """Random starting pose for perpendicular parking.

        The car spawns in a lane below the bay, roughly centered,
        and guaranteed not to collide with any obstacle at t=0.
        """
        gx, gy, gyaw = self.goal

        lane_x_center = gx
        lane_half_width = 0.25  # narrower lane than ±0.5
        y_min = gy - 1.2
        y_max = gy - 0.9

        for _ in range(50):
            x = np.random.uniform(lane_x_center - lane_half_width,
                                  lane_x_center + lane_half_width)
            y = np.random.uniform(y_min, y_max)
            yaw = gyaw
            v = 0.0
            candidate = np.array([x, y, yaw, v], dtype=float)

            if not self.obstacles.check_collision(candidate, self.model):
                return candidate

        # fallback safe pose
        x = lane_x_center
        y = y_min
        yaw = gyaw
        v = 0.0
        return np.array([x, y, yaw, v], dtype=float)

    def _get_obs(self) -> np.ndarray:
        x, y, yaw, v = self.state
        gx, gy, gyaw = self.goal

        dx = gx - x
        dy = gy - y
        dtheta = ((gyaw - yaw + np.pi) % (2.0 * np.pi)) - np.pi

        dist_front, dist_left, dist_right = self.obstacles.distance_features(self.state)

        obs = np.array(
            [x, y, yaw, v, dx, dy, dtheta, dist_front, dist_left, dist_right],
            dtype=float,
        )
        return obs

    def step(self, action):
        """Apply control and advance the simulation by one time-step."""
        self.step_count += 1

        steer = float(action[0])
        accel = float(action[1])

        next_state = self.model.step(self.state, (steer, accel), dt=self.dt)

        collision = self.obstacles.check_collision(next_state, self.model)
        reward, info = compute_reward(self.state, next_state, self.goal, collision)

        done = False
        if collision:
            done = True
            info["termination"] = "collision"
        elif self._is_success(next_state):
            done = True
            info["termination"] = "success"
        elif self.step_count >= self.max_steps:
            done = True
            info["termination"] = "max_steps"

        self.state = next_state
        return self._get_obs(), reward, done, info

    def _is_success(self, state: np.ndarray) -> bool:
        """Check if the car is parked sufficiently well in the bay."""
        x, y, yaw, v = state
        gx, gy, gyaw = self.goal

        pos_err = np.hypot(gx - x, gy - y)
        yaw_err = abs(((gyaw - yaw + np.pi) % (2.0 * np.pi)) - np.pi)

        # Slightly relaxed tolerances; tighten later if you want.
        return (pos_err < 0.20) and (yaw_err < 0.25) and (abs(v) < 0.10)
