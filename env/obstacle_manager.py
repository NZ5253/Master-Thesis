# env/obstacle_manager.py
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
        self.world_cfg = self.config.get("world", None)
        self.neigh_cfg = self.config.get("neighbor", None)
        self.random_cfg = self.config.get("random", None)
        self.obstacles: List[Dict[str, float]] = []
        self.reset_to_base()

    def set_goal(self, goal: np.ndarray) -> None:
        self.goal = np.array(goal, dtype=float)

    def reset_to_base(self) -> None:
        self.randomize()

    def randomize(self) -> None:
        self._build_from_goal()

    def _build_from_goal(self) -> None:
        gx, gy, gyaw = self.goal
        obs: List[Dict[str, float]] = []

        # 1. World Walls (Always keep these)
        if self.world_cfg is not None:
            x_min = float(self.world_cfg.get("x_min", -2.0))
            x_max = float(self.world_cfg.get("x_max", 2.0))
            y_min = float(self.world_cfg.get("y_min", -2.0))
            y_max = float(self.world_cfg.get("y_max", 2.0))
            t = float(self.world_cfg.get("thickness", 0.1))

            obs.append({"x": 0.5 * (x_min + x_max), "y": y_min + t / 2, "w": x_max - x_min, "h": t, "theta": 0.0})
            obs.append({"x": 0.5 * (x_min + x_max), "y": y_max - t / 2, "w": x_max - x_min, "h": t, "theta": 0.0})
            obs.append({"x": x_min + t / 2, "y": 0.5 * (y_min + y_max), "w": t, "h": y_max - y_min, "theta": 0.0})
            obs.append({"x": x_max - t / 2, "y": 0.5 * (y_min + y_max), "w": t, "h": y_max - y_min, "theta": 0.0})

        # --- HARDCODED LOGIC FOR PARALLEL PARKING ---
        # If yaw is close to 0, we assume Parallel Parking Scenario.
        # We FORCE the neighbors to be exactly at +0.52 and -0.52.
        if abs(gyaw) < 0.1:
            offset = float(self.neigh_cfg.get("offset", 0.43)) if self.neigh_cfg else 0.43
            # Neighbor Dimensions
            car_w = 0.36
            car_h = 0.26
            offset

            # 1. Rear Neighbor (Fixed at -0.52)
            obs.append({
                "x": gx - offset,
                "y": gy,
                "w": car_w,
                "h": car_h,
                "theta": 0.0
            })

            # 2. Front Neighbor (Fixed at +0.52)
            obs.append({
                "x": gx + offset,
                "y": gy,
                "w": car_w,
                "h": car_h,
                "theta": 0.0
            })

            # DONE. No random jitter, no calculation. Just fixed blocks.
            self.obstacles = obs
            return

            # --- PERPENDICULAR / OTHER LOGIC (Keep original for Perpendicular) ---
        if self.neigh_cfg is not None:
            car_w = float(self.neigh_cfg.get("w", 0.26))
            car_h = float(self.neigh_cfg.get("h", 0.36))
            offset = float(self.neigh_cfg.get("offset", 0.40))
            pos_jitter = float(self.neigh_cfg.get("pos_jitter", 0.0))

            c, s = float(np.cos(gyaw)), float(np.sin(gyaw))

            for sy in (-offset, offset):
                jitter_x = np.random.uniform(-pos_jitter, pos_jitter) if pos_jitter > 0 else 0.0
                jitter_y = np.random.uniform(-pos_jitter, pos_jitter) if pos_jitter > 0 else 0.0

                lx_local = jitter_x
                ly_local = sy + jitter_y

                wx = gx + c * lx_local - s * ly_local
                wy = gy + s * lx_local + c * ly_local

                obs.append({
                    "x": float(wx), "y": float(wy),
                    "w": float(car_h), "h": float(car_w),  # Swapped for vertical
                    "theta": float(gyaw),
                })

        self.obstacles = obs

    def distance_features(self, state: np.ndarray, max_range: float = 5.0):
        # ... (Keep this function exactly as it was) ...
        x, y, yaw, v = state
        dirs = [yaw, yaw + np.pi / 2.0, yaw - np.pi / 2.0]
        dists = []
        for ang in dirs:
            dx_dir = float(np.cos(ang))
            dy_dir = float(np.sin(ang))
            best = max_range
            for o in self.obstacles:
                rx = o["x"] - x
                ry = o["y"] - y
                proj = rx * dx_dir + ry * dy_dir
                if proj <= 0.0: continue
                dist = proj - min(o["w"], o["h"]) / 2
                if dist < best: best = dist
            dists.append(max(0.0, best))
        return dists[0], dists[1], dists[2]

    def get_car_corners(self, state, vehicle_params):
        # ... (Keep exactly as was) ...
        x, y, yaw, _ = state
        L = vehicle_params.get('length', 0.36)
        W = vehicle_params.get('width', 0.26)
        corners_local = np.array([
            [L / 2, W / 2], [L / 2, -W / 2], [-L / 2, -W / 2], [-L / 2, W / 2]
        ])
        c, s = np.cos(yaw), np.sin(yaw)
        rot = np.array([[c, -s], [s, c]])
        return (rot @ corners_local.T).T + np.array([x, y])

    def check_collision(self, state, vehicle_model) -> bool:
        # ... (Keep exactly as was) ...
        corners = self.get_car_corners(state, {'length': 0.36, 'width': 0.26})
        for ox, oy in corners:
            for o in self.obstacles:
                if abs(ox - o["x"]) < o["w"] / 2.0 and abs(oy - o["y"]) < o["h"] / 2.0:
                    return True
        return False