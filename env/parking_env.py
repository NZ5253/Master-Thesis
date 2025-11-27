import numpy as np
from .vehicle_model import KinematicBicycle
from .obstacle_manager import ObstacleManager
from .reward_functions import compute_reward


class ParkingEnv:
    """Gym-like parking environment (perpendicular parking).

    Observation:
        [x, y, yaw, v,
         dx, dy, dtheta,
         dist_front, dist_left, dist_right]

    Action:
        [steer, accel]
    """

    def __init__(self, config):
        # --------------------------------------------------------------
        # Core settings
        # --------------------------------------------------------------
        self.dt = config.get("dt", 0.1)
        self.model = KinematicBicycle(config["vehicle"])

        # goal will be overwritten each episode when we sample the bay
        self.goal = np.array(config.get("goal", [0.0, 0.0, 0.0]), dtype=float)

        self.max_steps = config.get("max_steps", 200)
        self.state = None
        self.step_count = 0

        # world bounds for 4×4 m world (or fallback)
        self.world = config.get("world", None)
        if self.world is not None:
            width = float(self.world.get("width", 4.0))
            height = float(self.world.get("height", 4.0))
            half_w = width / 2.0
            half_h = height / 2.0
            self.x_min = -half_w
            self.x_max = half_w
            self.y_min = -half_h
            self.y_max = half_h
        else:
            # fallback to old [-1, 1] × [-1, 1] region
            self.x_min, self.x_max = -1.0, 1.0
            self.y_min, self.y_max = -1.0, 1.0

        # per-episode layout config
        self.parking_cfg = config.get("parking", {})
        self.spawn_cfg = config.get("spawn_lane", {})

        # Obstacle manager: walls + neighbours + random obstacles
        obs_cfg = config.get("obstacles", {})
        self.obstacles = ObstacleManager(obs_cfg, goal=self.goal)

    # ------------------------------------------------------------------
    # Episode setup
    # ------------------------------------------------------------------
    def reset(self, randomize=True):
        if randomize:
            # 1) sample new bay (goal pose) + obstacles
            self._sample_bay_and_obstacles()
            # 2) spawn ego relative to this bay, avoiding collisions
            self.state = self._random_start()
        else:
            # deterministic fallback: keep whatever goal is set in config
            self.obstacles.set_goal(self.goal)
            self.obstacles.reset_to_base()
            self.state = np.array([0.0, -1.0, self.goal[2], 0.0], dtype=float)

        self.step_count = 0
        return self._get_obs()

    def _sample_bay_and_obstacles(self):
        """Sample bay position (goal) and regenerate obstacle layout."""

        bay_cfg = self.parking_cfg.get("bay", {})

        # Bay centre range along X (inside world bounds by some margin)
        cx_min = float(bay_cfg.get("x_min", self.x_min + 0.5))
        cx_max = float(bay_cfg.get("x_max", self.x_max - 0.5))
        cy = float(bay_cfg.get("center_y", 0.0))
        yaw = float(bay_cfg.get("yaw", 0.0))

        # sample bay center along X
        cx = np.random.uniform(cx_min, cx_max)

        # update goal: center of the slot
        self.goal = np.array([cx, cy, yaw], dtype=float)

        # notify obstacle manager & randomise layout
        self.obstacles.set_goal(self.goal)
        self.obstacles.randomize()

    def _random_start(self):
        """Spawn the ego car in a lane below the bay, relative to the current goal."""
        gx, gy, gyaw = self.goal

        lane_y_min = float(self.spawn_cfg.get("y_min", gy - 1.6))
        lane_y_max = float(self.spawn_cfg.get("y_max", gy - 1.3))
        half_w = float(self.spawn_cfg.get("x_half_width", 0.8))

        for _ in range(50):
            x = np.random.uniform(gx - half_w, gx + half_w)
            y = np.random.uniform(lane_y_min, lane_y_max)
            yaw = gyaw  # facing roughly towards the bays
            v = 0.0
            candidate = np.array([x, y, yaw, v], dtype=float)

            if self._out_of_bounds(candidate):
                continue
            if self.obstacles.check_collision(candidate, self.model):
                continue
            return candidate

        # fallback: lane center below bay
        x = gx
        y = 0.5 * (lane_y_min + lane_y_max)
        return np.array([x, y, gyaw, 0.0], dtype=float)

    # ------------------------------------------------------------------
    # Core env API
    # ------------------------------------------------------------------
    def _get_obs(self):
        x, y, yaw, v = self.state
        gx, gy, gyaw = self.goal
        dx = gx - x
        dy = gy - y
        dtheta = (gyaw - yaw + np.pi) % (2 * np.pi) - np.pi

        dist_front, dist_left, dist_right = self.obstacles.distance_features(
            self.state
        )
        obs = np.array(
            [x, y, yaw, v, dx, dy, dtheta, dist_front, dist_left, dist_right],
            dtype=float,
        )
        return obs

    def step(self, action):
        self.step_count += 1
        steer, accel = action
        next_state = self.model.step(self.state, (steer, accel), dt=self.dt)

        collision = self.obstacles.check_collision(next_state, self.model)
        if self._out_of_bounds(next_state):
            collision = True

        reward, info = compute_reward(self.state, next_state, self.goal, collision)
        done = False
        success = self._is_success(next_state)

        if collision:
            done = True
            info["termination"] = "collision"
        if success:
            done = True
            info["termination"] = "success"
        if self.step_count >= self.max_steps:
            done = True
            info["termination"] = "max_steps"

        self.state = next_state
        return self._get_obs(), reward, done, info

    def _is_success(self, state):
        # Match success thresholds with reward_functions.compute_reward()
        x, y, yaw, v = state
        gx, gy, gyaw = self.goal
        pos_err = np.hypot(gx - x, gy - y)
        yaw_err = abs(((gyaw - yaw + np.pi) % (2 * np.pi)) - np.pi)
        return (pos_err < 0.08) and (yaw_err < 0.1) and (abs(v) < 0.05)

    def _out_of_bounds(self, state):
        """Treat leaving the world rectangle as a collision."""
        x, y, _, _ = state
        return (x < self.x_min) or (x > self.x_max) or (y < self.y_min) or (y > self.y_max)
