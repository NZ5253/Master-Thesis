import numpy as np
from .vehicle_model import KinematicBicycle
from .obstacle_manager import ObstacleManager
from .reward_functions import compute_reward


class ParkingEnv:
    """
    Gym-like parking environment (parallel or perpendicular, depending on config).

    Observation:
        [x, y, yaw, v,
         dx, dy, dtheta,
         dist_front, dist_left, dist_right]

    Action:
        [steer, accel]
    """

    def __init__(self, config):
        # ------------ core settings (from config_env.yaml) ------------
        self.vehicle_params = config["vehicle"]
        self.dt = config["dt"]
        self.max_steps = config["max_steps"]

        self.model = KinematicBicycle(self.vehicle_params)

        # This "goal" will be overwritten each episode when sampling the bay
        self.goal = np.array(config.get("goal", [0.0, 0.0, 0.0]), dtype=float)

        # world bounds for 4×4 m world (or fallback if missing)
        world_cfg = config.get("world", {"width": 4.0, "height": 4.0})
        width = float(world_cfg.get("width", 4.0))
        height = float(world_cfg.get("height", 4.0))
        half_w = width / 2.0
        half_h = height / 2.0
        self.x_min = -half_w
        self.x_max = half_w
        self.y_min = -half_h
        self.y_max = half_h

        self.state = None
        self.step_count = 0

        # per-episode layout config
        self.parking_cfg = config.get("parking", {})
        self.spawn_cfg = config.get("spawn_lane", {})

        # obstacles: walls + neighbours + optional random
        obs_cfg = config.get("obstacles", {})
        self.obstacles = ObstacleManager(
            obs_cfg,
            goal=self.goal,
            vehicle_params=self.vehicle_params,
        )
        self.bay_center = np.array([0.0, 0.0, 0.0], dtype=float)

    # ------------------------------------------------------------------
    # Episode setup
    # ------------------------------------------------------------------
    def reset(self, randomize: bool = True):
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
        bay_cfg = self.parking_cfg.get("bay", {})

        cx = 0.0
        center_y = float(bay_cfg.get("center_y", 0.0))
        yaw = float(bay_cfg.get("yaw", 0.0))

        self.bay_center = np.array([cx, center_y, yaw], dtype=float)

        L = float(self.vehicle_params.get("length", 0.36))
        dist_to_center = L / 2.0 - 0.05  # must match ObstacleManager.get_car_corners

        if abs(yaw) < 0.3:
            # PARALLEL:
            # We want CAR CENTER to be at the bay center (0, center_y).
            # Rear-axle goal is therefore shifted backwards along the car axis.
            goal_x = cx - dist_to_center * np.cos(yaw)
            goal_y = center_y - dist_to_center * np.sin(yaw)
        else:
            # PERPENDICULAR: allow config override
            default_goal_offset_y = -dist_to_center
            goal_offset_y = float(bay_cfg.get("goal_offset_y", default_goal_offset_y))
            goal_x = cx
            goal_y = center_y + goal_offset_y

        self.goal = np.array([goal_x, goal_y, yaw], dtype=float)

        self.obstacles.set_goal(self.bay_center)
        self.obstacles.randomize()

    def _random_start(self):
        """
        Spawn the ego car relative to the current goal, using spawn_lane config:

          spawn_lane:
            y_min, y_max
            yaw
            x_min_offset, x_max_offset  (for parallel)
            x_half_width               (for perpendicular)
        """
        gx, gy, gyaw = self.goal

        lane_y_min = float(self.spawn_cfg.get("y_min", gy - 1.6))
        lane_y_max = float(self.spawn_cfg.get("y_max", gy - 1.3))

        spawn_yaw = float(self.spawn_cfg.get("yaw", gyaw))

        # Asymmetric spawning (e.g. parallel) if x_min_offset is specified
        if "x_min_offset" in self.spawn_cfg:
            x_min = gx + float(self.spawn_cfg["x_min_offset"])
            x_max = gx + float(self.spawn_cfg.get("x_max_offset", 2.0))
        else:
            half_w = float(self.spawn_cfg.get("x_half_width", 0.8))
            x_min = gx - half_w
            x_max = gx + half_w

        for _ in range(50):
            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(lane_y_min, lane_y_max)
            yaw = spawn_yaw
            v = 0.0
            candidate = np.array([x, y, yaw, v], dtype=float)

            if self._out_of_bounds(candidate):
                continue
            if self.obstacles.check_collision(candidate, self.model):
                continue
            return candidate

        # Fallback if random placement fails
        return np.array(
            [x_max, 0.5 * (lane_y_min + lane_y_max), spawn_yaw, 0.0],
            dtype=float,
        )

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
        """
        Parallel: success if CAR CENTER is close to the slot center (between neighbours),
                  with small yaw and low speed.
        Perpendicular: keep a simple rear-axle based test.
        """
        x_ra, y_ra, yaw, v = state
        gx, gy, gyaw = self.goal

        # --- common yaw + speed check ---
        yaw_err = abs(((gyaw - yaw + np.pi) % (2 * np.pi)) - np.pi)
        if abs(v) > 0.05:  # must be (almost) stopped
            return False
        if yaw_err > 0.05:  # ~2.9 degrees
            return False

        bay_cfg = self.parking_cfg.get("bay", {})
        bay_yaw = float(bay_cfg.get("yaw", 0.0))

        # vehicle geometry
        L = float(self.vehicle_params.get("length", 0.36))
        dist_to_center = L / 2.0 - 0.05  # must match ObstacleManager / viz

        # car center in world frame
        cx = x_ra + dist_to_center * np.cos(yaw)
        cy = y_ra + dist_to_center * np.sin(yaw)

        if abs(bay_yaw) < 0.3:
            # -------- PARALLEL: center-based success --------
            slot_cx = 0.0
            slot_cy = float(bay_cfg.get("center_y", 0.13))

            along_err = abs(cx - slot_cx)
            lateral_err = abs(cy - slot_cy)

            # stricter tolerances → car ends really centred & deep
            return (along_err < 0.03) and (lateral_err < 0.03)

        else:
            # -------- PERPENDICULAR: keep simple rear-axle check --------
            pos_err = np.hypot(gx - x_ra, gy - y_ra)
            return pos_err < 0.08

    def _out_of_bounds(self, state):
        """Treat leaving the world rectangle as a collision."""
        x, y, _, _ = state
        return (x < self.x_min) or (x > self.x_max) or (y < self.y_min) or (y > self.y_max)
