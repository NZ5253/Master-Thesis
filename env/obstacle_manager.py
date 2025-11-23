import numpy as np
from typing import Dict, Any, List


class ObstacleManager:
    """Generate and manage rectangular obstacles for perpendicular parking.

    Obstacles are stored as dicts:
        {"x": cx, "y": cy, "w": width, "h": height}

    Layout is generated procedurally from a small set of parameters so that
    neighbour cars are placed to the *left and right* of the goal parking bay.
    Sizes and spacing are configured in config_env.yaml.
    """

    def __init__(self, config: Dict[str, Any], goal: np.ndarray) -> None:
        # Store goal pose [gx, gy, gyaw]
        self.goal = np.asarray(goal, dtype=float)

        # Variability (jitter) settings
        var_cfg = config.get("variability", {})
        self.pos_jitter = float(var_cfg.get("pos", 0.0))
        self.size_jitter = float(var_cfg.get("size", 0.0))

        # World walls (outer rectangle)
        world_cfg = config.get("world", {})
        x_min = float(world_cfg.get("x_min", -2.0))
        x_max = float(world_cfg.get("x_max", 2.0))
        y_min = float(world_cfg.get("y_min", -2.0))
        y_max = float(world_cfg.get("y_max", 2.0))
        thickness = float(world_cfg.get("thickness", 0.05))

        walls: List[Dict[str, float]] = [
            # bottom wall
            {"x": 0.5 * (x_min + x_max), "y": y_min,
             "w": x_max - x_min, "h": thickness},
            # top wall
            {"x": 0.5 * (x_min + x_max), "y": y_max,
             "w": x_max - x_min, "h": thickness},
            # left wall
            {"x": x_min, "y": 0.5 * (y_min + y_max),
             "w": thickness, "h": y_max - y_min},
            # right wall
            {"x": x_max, "y": 0.5 * (y_min + y_max),
             "w": thickness, "h": y_max - y_min},
        ]

        # Neighbour cars (left/right of the goal)
        neigh_cfg = config.get("neighbor", {})
        neigh_w = float(neigh_cfg.get("w", 0.26))
        neigh_h = float(neigh_cfg.get("h", 0.50))
        neigh_off = float(neigh_cfg.get("offset", 0.40))

        gx, gy, _ = self.goal
        neighbours: List[Dict[str, float]] = [
            {"x": gx - neigh_off, "y": gy, "w": neigh_w, "h": neigh_h},
            {"x": gx + neigh_off, "y": gy, "w": neigh_w, "h": neigh_h},
        ]

        self.wall_layout = walls
        self.neighbour_layout = neighbours

        # Base (non-randomised) layout
        self.base_layout: List[Dict[str, float]] = walls + neighbours
        self.obstacles: List[Dict[str, float]] = [dict(o) for o in self.base_layout]

    # ------------------------------------------------------------------
    # Layout management
    # ------------------------------------------------------------------
    def reset_to_base(self) -> None:
        """Reset obstacles to the non-randomised base layout."""
        self.obstacles = [dict(o) for o in self.base_layout]

    def randomize(self) -> None:
        """Apply small random perturbations to neighbour cars only."""
        obs: List[Dict[str, float]] = [dict(o) for o in self.wall_layout]  # walls unchanged

        for o in self.neighbour_layout:
            x = o["x"] + np.random.uniform(-self.pos_jitter, self.pos_jitter)
            y = o["y"] + np.random.uniform(-self.pos_jitter, self.pos_jitter)
            w = o["w"] * (1.0 + np.random.uniform(-self.size_jitter, self.size_jitter))
            h = o["h"] * (1.0 + np.random.uniform(-self.size_jitter, self.size_jitter))
            obs.append({"x": float(x), "y": float(y), "w": float(w), "h": float(h)})

        self.obstacles = obs

    # ------------------------------------------------------------------
    # Features for RL / control
    # ------------------------------------------------------------------
    def distance_features(self, state: np.ndarray, max_range: float = 5.0):
        """Approximate distances in front, left and right of the car.

        Each obstacle is approximated by its centre point. We cast three rays:
            front  – along yaw
            left   – yaw + pi/2
            right  – yaw - pi/2

        and report the minimum positive projection (distance) along each ray.
        """
        x, y, yaw, v = state
        dirs = [yaw, yaw + np.pi / 2.0, yaw - np.pi / 2.0]
        dists = []

        for ang in dirs:
            dx_dir = float(np.cos(ang))
            dy_dir = float(np.sin(ang))
            best = max_range
            for o in self.obstacles:
                cx = o["x"]
                cy = o["y"]
                rx = cx - x
                ry = cy - y
                proj = rx * dx_dir + ry * dy_dir
                if proj <= 0.0:
                    continue  # obstacle behind along this ray
                if proj < best:
                    best = proj
            dists.append(best)

        return dists[0], dists[1], dists[2]

    def check_collision(self, state: np.ndarray, vehicle_model) -> bool:
        """Simple circle vs axis-aligned-rectangle collision check."""
        x, y, yaw, v = state
        r = 0.06  # approximate vehicle radius; tune if needed

        for o in self.obstacles:
            cx = o["x"]
            cy = o["y"]
            w = o["w"]
            h = o["h"]

            dx = abs(x - cx)
            dy = abs(y - cy)
            if dx <= (w / 2.0 + r) and dy <= (h / 2.0 + r):
                return True

        return False
