import numpy as np
from typing import Dict, Any, List, Optional


class ObstacleManager:

    def __init__(self, config: Dict[str, Any], goal: Optional[np.ndarray] = None) -> None:
        self.config = config or {}
        self.goal = (
            np.array(goal, dtype=float)
            if goal is not None
            else np.zeros(3, dtype=float)
        )

        # legacy style
        self.base_layout: List[Dict[str, float]] = list(
            self.config.get("base_layout", [])
        )
        self.variability = self.config.get(
            "variability", {"pos": 0.0, "size": 0.0}
        )

        # new style
        self.world_cfg = self.config.get("world", None)
        self.neigh_cfg = self.config.get("neighbor", None)
        self.random_cfg = self.config.get("random", None)

        # current episode obstacles
        self.obstacles: List[Dict[str, float]] = []

        # initial layout
        self.reset_to_base()

    # ------------------------------------------------------------------
    # API used by ParkingEnv
    # ------------------------------------------------------------------
    def set_goal(self, goal: np.ndarray) -> None:
        """Update the bay / goal pose this manager should align to."""
        self.goal = np.array(goal, dtype=float)

    def reset_to_base(self) -> None:
        """For scenario_loader / deterministic reset."""
        self.randomize()

    def randomize(self) -> None:
        """Build a new obstacle layout for this episode."""
        if self.world_cfg is not None or self.neigh_cfg is not None or self.random_cfg is not None:
            self._build_from_goal()
        else:
            self._jitter_base_layout()

    # ------------------------------------------------------------------
    # Internal layout builders
    # ------------------------------------------------------------------
    def _jitter_base_layout(self) -> None:
        """Legacy path: jitter around a fixed base_layout."""
        pos_j = float(self.variability.get("pos", 0.0))
        size_j = float(self.variability.get("size", 0.0))

        obs: List[Dict[str, float]] = []
        for o in self.base_layout:
            x = o["x"] + np.random.uniform(-pos_j, pos_j)
            y = o["y"] + np.random.uniform(-pos_j, pos_j)
            w = o["w"] * (1.0 + np.random.uniform(-size_j, size_j))
            h = o["h"] * (1.0 + np.random.uniform(-size_j, size_j))
            theta = o.get("theta", 0.0)

            obs.append({
                "x": float(x), "y": float(y),
                "w": float(w), "h": float(h),
                "theta": float(theta),
            })

        self.obstacles = obs

    def _build_from_goal(self) -> None:
        """
        Builds walls, neighbour cars (Left/Right), and random obstacles.
        """
        gx, gy, gyaw = self.goal
        obs: List[Dict[str, float]] = []

        # ----- 1. World Walls (Fixed Room) -----
        if self.world_cfg is not None:
            x_min = float(self.world_cfg.get("x_min", -2.0))
            x_max = float(self.world_cfg.get("x_max", 2.0))
            y_min = float(self.world_cfg.get("y_min", -2.0))
            y_max = float(self.world_cfg.get("y_max", 2.0))
            t = float(self.world_cfg.get("thickness", 0.1))

            # Bottom, Top, Left, Right
            obs.append({"x": 0.5 * (x_min + x_max), "y": y_min + t / 2, "w": x_max - x_min, "h": t, "theta": 0.0})
            obs.append({"x": 0.5 * (x_min + x_max), "y": y_max - t / 2, "w": x_max - x_min, "h": t, "theta": 0.0})
            obs.append({"x": x_min + t / 2, "y": 0.5 * (y_min + y_max), "w": t, "h": y_max - y_min, "theta": 0.0})
            obs.append({"x": x_max - t / 2, "y": 0.5 * (y_min + y_max), "w": t, "h": y_max - y_min, "theta": 0.0})

        # ----- 2. Neighbour Cars (Left & Right) -----
        if self.neigh_cfg is not None:
            car_w = float(self.neigh_cfg.get("w", 0.26))
            car_h = float(self.neigh_cfg.get("h", 0.36))
            offset = float(self.neigh_cfg.get("offset", 0.40))
            pos_jitter = float(self.neigh_cfg.get("pos_jitter", 0.0))

            c, s = float(np.cos(gyaw)), float(np.sin(gyaw))

            # FIX: We want neighbors to the LEFT and RIGHT of the goal.
            # In the car frame, Left/Right is the Y-axis.
            # We iterate offsets +offset and -offset along the Y-axis.
            for sy in (-offset, offset):

                # Apply jitter to the "forward/backward" position (X-axis)
                lx_local = np.random.uniform(-pos_jitter, pos_jitter)
                # The main offset is along Y (lateral)
                ly_local = sy + np.random.uniform(-0.01, 0.01)

                # Rotate local (lx, ly) into world frame
                # World = Goal + Rotation * Local
                wx = gx + c * lx_local - s * ly_local
                wy = gy + s * lx_local + c * ly_local

                # FIX: Hitbox Rotation for Simple Collision
                # If the car is vertical (90 deg), its "World Width" is its Height,
                # and "World Height" is its Width.
                # We simply swap w/h for the hitbox if the angle is close to 90.
                if abs(abs(gyaw) - 1.5708) < 0.2:  # Approx 90 degrees
                    eff_w, eff_h = car_h, car_w
                else:
                    eff_w, eff_h = car_w, car_h

                obs.append({
                    "x": float(wx),
                    "y": float(wy),
                    "w": float(eff_w),  # Effective width for AABB
                    "h": float(eff_h),  # Effective height for AABB
                    "theta": float(gyaw),
                })

        # ----- 3. Random Obstacles -----
        if self.random_cfg is not None:
            num_min = int(self.random_cfg.get("num_min", 0))
            num_max = int(self.random_cfg.get("num_max", num_min))
            if num_max < num_min: num_max = num_min

            n = np.random.randint(num_min, num_max + 1) if num_max > 0 else 0

            # Default ranges
            x_range = self.random_cfg.get("x_range", [-1.5, 1.5])
            y_range = self.random_cfg.get("y_range", [-1.5, 1.5])
            w_range = self.random_cfg.get("w_range", [0.2, 0.4])
            h_range = self.random_cfg.get("h_range", [0.2, 0.4])

            for _ in range(n):
                # Try to spawn away from goal
                for _try in range(10):
                    ox = np.random.uniform(*x_range)
                    oy = np.random.uniform(*y_range)
                    # Check distance to goal center
                    if np.hypot(ox - gx, oy - gy) > 0.8:
                        break

                ow = np.random.uniform(*w_range)
                oh = np.random.uniform(*h_range)
                obs.append({
                    "x": float(ox), "y": float(oy),
                    "w": float(ow), "h": float(oh),
                    "theta": 0.0,
                })

        self.obstacles = obs

    # ------------------------------------------------------------------
    # Features for RL / Collision
    # ------------------------------------------------------------------
    def distance_features(self, state: np.ndarray, max_range: float = 5.0):
        # (Same as before)
        x, y, yaw, v = state
        dirs = [yaw, yaw + np.pi / 2.0, yaw - np.pi / 2.0]
        dists = []
        for ang in dirs:
            dx_dir = float(np.cos(ang))
            dy_dir = float(np.sin(ang))
            best = max_range
            for o in self.obstacles:
                # Approximate distance to rectangle center
                rx = o["x"] - x
                ry = o["y"] - y
                proj = rx * dx_dir + ry * dy_dir
                if proj <= 0.0: continue
                # Simple subtraction of radius approx
                dist = proj - min(o["w"], o["h"]) / 2
                if dist < best: best = dist
            dists.append(max(0.0, best))
        return dists[0], dists[1], dists[2]

    def get_car_corners(self, state, vehicle_params):
        x, y, yaw, _ = state
        L = vehicle_params.get('length', 0.36)
        W = vehicle_params.get('width', 0.26)
        # Center of rear axle is x,y.
        # Car extends forward by L, backward by 0.0 (simple) or small overhang
        rear_overhang = 0.05
        front_overhang = L - rear_overhang

        corners_local = np.array([
            [front_overhang, W / 2],
            [front_overhang, -W / 2],
            [-rear_overhang, -W / 2],
            [-rear_overhang, W / 2]
        ])

        c, s = np.cos(yaw), np.sin(yaw)
        rot = np.array([[c, -s], [s, c]])
        return (rot @ corners_local.T).T + np.array([x, y])

    def check_collision(self, state, vehicle_model) -> bool:
        """
        Robust check using point-in-AABB (Axis Aligned Bounding Box).
        Works because we swapped w/h for rotated neighbors in _build_from_goal.
        """
        corners = self.get_car_corners(state, {'length': 0.36, 'width': 0.26})

        for ox, oy in corners:
            for o in self.obstacles:
                # Check bounds
                dx = abs(ox - o["x"])
                dy = abs(oy - o["y"])

                # Use the effective W/H stored in obs
                if dx < o["w"] / 2.0 and dy < o["h"] / 2.0:
                    return True
        return False