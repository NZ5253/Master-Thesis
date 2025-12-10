import numpy as np
from typing import Dict, Any, List, Optional


class ObstacleManager:

    def __init__(
        self,
        config: Dict[str, Any],
        goal: Optional[np.ndarray] = None,
        vehicle_params: Optional[Dict[str, float]] = None,
    ) -> None:
        self.config = config or {}
        self.goal = (
            np.array(goal, dtype=float)
            if goal is not None
            else np.zeros(3, dtype=float)
        )

        # obstacle layout config
        self.world_cfg = self.config.get("world", {}) or {}
        self.neigh_cfg = self.config.get("neighbor", {}) or {}
        self.random_cfg = self.config.get("random", {}) or {}

        # ego vehicle geometry (for collision)
        self.vehicle_params = vehicle_params or {}
        self.obstacles: List[Dict[str, float]] = []

        self.reset_to_base()

    # ---------------------- public API -----------------------

    def set_goal(self, goal: np.ndarray) -> None:
        self.goal = np.array(goal, dtype=float)

    def reset_to_base(self) -> None:
        self.randomize()

    def randomize(self) -> None:
        """Rebuild obstacle layout for current goal."""
        self._build_from_goal()

    def _build_from_goal(self) -> None:
        gx, gy, gyaw = self.goal
        obs: List[Dict[str, float]] = []

        # 1) World walls
        if self.world_cfg:
            x_min = float(self.world_cfg.get("x_min", -2.0))
            x_max = float(self.world_cfg.get("x_max", 2.0))
            y_min = float(self.world_cfg.get("y_min", -2.0))
            y_max = float(self.world_cfg.get("y_max", 2.0))
            t = float(self.world_cfg.get("thickness", 0.1))

            obs.append(
                {
                    "x": 0.5 * (x_min + x_max),
                    "y": y_min + t / 2,
                    "w": x_max - x_min,
                    "h": t,
                    "theta": 0.0,
                }
            )
            obs.append(
                {
                    "x": 0.5 * (x_min + x_max),
                    "y": y_max - t / 2,
                    "w": x_max - x_min,
                    "h": t,
                    "theta": 0.0,
                }
            )
            obs.append(
                {
                    "x": x_min + t / 2,
                    "y": 0.5 * (y_min + y_max),
                    "w": t,
                    "h": y_max - y_min,
                    "theta": 0.0,
                }
            )
            obs.append(
                {
                    "x": x_max - t / 2,
                    "y": 0.5 * (y_min + y_max),
                    "w": t,
                    "h": y_max - y_min,
                    "theta": 0.0,
                }
            )

        # 2) Neighbour cars (parallel vs perpendicular)
        if self.neigh_cfg:
            car_w = float(self.neigh_cfg.get("w", self.vehicle_params.get("length", 0.36)))
            car_h = float(self.neigh_cfg.get("h", self.vehicle_params.get("width", 0.26)))
            offset = float(self.neigh_cfg.get("offset", 0.4))
            pos_jitter = float(self.neigh_cfg.get("pos_jitter", 0.0))

            # We keep using gyaw to distinguish scenarios:
            # - parallel: goal yaw ≈ 0 -> neighbours in front/behind along bay axis
            # - perpendicular: yaw ≈ pi/2 -> neighbours left/right of bay
            c, s = float(np.cos(gyaw)), float(np.sin(gyaw))

            if abs(gyaw) < 0.1:
                # --------- PARALLEL PARKING: front & rear neighbours ---------
                # Place neighbours along bay's longitudinal axis
                for sx in (-offset, offset):
                    jx = np.random.uniform(-pos_jitter, pos_jitter) if pos_jitter > 0 else 0.0
                    jy = np.random.uniform(-pos_jitter, pos_jitter) if pos_jitter > 0 else 0.0

                    lx_local = sx + jx   # along bay x-axis
                    ly_local = jy        # small lateral jitter only

                    wx = gx + c * lx_local - s * ly_local
                    wy = gy + s * lx_local + c * ly_local

                    obs.append(
                        {
                            "x": float(wx),
                            "y": float(wy),
                            "w": float(car_w),
                            "h": float(car_h),
                            "theta": float(gyaw),
                        }
                    )

                # ----- NEW: soft curb just below both neighbours -----
                curb_gap = float(self.neigh_cfg.get("curb_gap", 0.05))         # distance from parked cars
                curb_thickness = float(self.neigh_cfg.get("curb_thickness", 0.04))

                # For your current config gyaw ≈ 0, so we place a thin bar under the neighbours
                curb_y = gy - car_h / 2.0 - curb_gap - curb_thickness / 2.0
                curb_width = 2.0 * offset + car_w  # span covering both neighbours + slot

                obs.append(
                    {
                        "x": float(gx),              # centred under the bay
                        "y": float(curb_y),
                        "w": float(curb_width),
                        "h": float(curb_thickness),
                        "theta": 0.0,
                        "kind": "curb",              # mark as soft curb
                    }
                )

            else:
                # --------- PERPENDICULAR / OTHER: neighbours side-by-side ----
                # Place neighbours offset along local y-axis of bay (left/right)
                for sy in (-offset, offset):
                    jx = np.random.uniform(-pos_jitter, pos_jitter) if pos_jitter > 0 else 0.0
                    jy = np.random.uniform(-pos_jitter, pos_jitter) if pos_jitter > 0 else 0.0

                    lx_local = jx
                    ly_local = sy + jy

                    wx = gx + c * lx_local - s * ly_local
                    wy = gy + s * lx_local + c * ly_local

                    # keep rectangles aligned with bay heading
                    obs.append(
                        {
                            "x": float(wx),
                            "y": float(wy),
                            "w": float(car_h),  # swapped for vertical arrangement
                            "h": float(car_w),
                            "theta": float(gyaw),
                        }
                    )

        # 3) (Optional) random obstacles – you can add a generator here later
        #    using self.random_cfg if you want.

        self.obstacles = obs


    # ---------------------- features / collision ----------------------

    def distance_features(self, state: np.ndarray, max_range: float = 5.0):
        """
        Simple distance features in ego heading frame:
        - distance straight ahead
        - distance to left (90°)
        - distance to right (-90°)
        """
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
                if proj <= 0.0:
                    continue
                dist = proj - min(o["w"], o["h"]) / 2.0
                if dist < best:
                    best = dist
            dists.append(max(0.0, best))
        return dists[0], dists[1], dists[2]

    def get_car_corners(self, state, vehicle_params=None):
        """
        Compute ego rectangle corners in world frame, using the same
        rear-axle -> center shift as your drawing/MPC.
        """
        if vehicle_params is None:
            vehicle_params = self.vehicle_params

        x_rear, y_rear, yaw, _ = state
        L = float(vehicle_params.get("length", 0.36))
        W = float(vehicle_params.get("width", 0.26))

        # Same center as draw_car / MPC:
        dist_to_center = L / 2.0 - 0.05  # 0.05 is the rear-overhang fudge
        cx = x_rear + dist_to_center * np.cos(yaw)
        cy = y_rear + dist_to_center * np.sin(yaw)

        corners_local = np.array(
            [
                [L / 2, W / 2],
                [L / 2, -W / 2],
                [-L / 2, -W / 2],
                [-L / 2, W / 2],
            ]
        )
        c, s = np.cos(yaw), np.sin(yaw)
        rot = np.array([[c, -s], [s, c]])
        return (rot @ corners_local.T).T + np.array([cx, cy])

    def check_collision(self, state, vehicle_model) -> bool:
        """
        Simple AABB-based collision check between ego corners and
        rectangular obstacles.
        """
        corners = self.get_car_corners(state)
        for ox, oy in corners:
            for o in self.obstacles:
                # soft curb: allowed to ride over, only used as soft cost in MPC
                if o.get("kind") == "curb":
                    continue
                if (
                    abs(ox - o["x"]) < o["w"] / 2.0
                    and abs(oy - o["y"]) < o["h"] / 2.0
                ):
                    return True
        return False
