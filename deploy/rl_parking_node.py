#!/usr/bin/env python3
"""
RL Parking Node -- Deploy trained PPO policy on ChronosCar hardware.

Receives car state via UDP from ros_bridge.py (running in CRS Docker),
computes bay-frame observations (including virtual obstacle distances),
runs PPO inference, and sends steering + torque commands back via UDP.

No ROS installation needed on the host.

Usage:
  # Dry-run (tests checkpoint loading + observation pipeline)
  python deploy/rl_parking_node.py --dry-run --checkpoint checkpoints/chronos/.../checkpoint_000XXX

  # Ghost mode (reads state via UDP, computes actions, does NOT send commands)
  python deploy/rl_parking_node.py --ghost --checkpoint ... --scene deploy/parking_scene.yaml

  # Live mode (actually controls the car)
  python deploy/rl_parking_node.py --checkpoint ... --scene deploy/parking_scene.yaml
"""

import sys
import os
import time
import csv
import argparse
import struct
import socket
import threading
import numpy as np
import yaml
from pathlib import Path

# Add project root to path so we can import env/ and rl/ modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from env.obstacle_manager import ObstacleManager
from env.vehicle_model import KinematicBicycle

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon


# ---------------------------------------------------------------------------
# Live top-down visualizer
# ---------------------------------------------------------------------------
class LiveVisualizer:
    """
    Real-time top-down matplotlib view showing:
      - Ego car (blue rectangle + heading arrow)
      - Virtual neighbor cars (skyblue rectangles)
      - Curb (gray bar)
      - Parking bay outline (green dashed rectangle)
      - Goal marker (red X)
      - Trajectory trail
    """

    def __init__(self, obs_builder: "ObservationBuilder", scene_cfg: dict):
        self.obs_builder = obs_builder
        veh = scene_cfg["vehicle"]
        self.car_length = float(veh["length"])
        self.car_width = float(veh["width"])
        self.rear_overhang = float(veh["rear_overhang"])
        self.dist_to_center = self.car_length / 2.0 - self.rear_overhang

        # Bay rectangle (same size as car)
        self.bay_x = obs_builder.bay_x
        self.bay_y = obs_builder.bay_y
        self.bay_yaw = obs_builder.bay_yaw

        # Trajectory history
        self.trail_x = []
        self.trail_y = []
        self.max_trail = 500

        # Set up matplotlib
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.fig.canvas.manager.set_window_title("ChronosCar RL Parking - Live View")
        self._first_draw = True

    def _rect_corners(self, cx, cy, w, h, theta):
        """Compute 4 corners of a rotated rectangle centered at (cx, cy)."""
        hw, hh = w / 2.0, h / 2.0
        corners = np.array([
            [hw, hh], [hw, -hh], [-hw, -hh], [-hw, hh]
        ])
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        return corners @ R.T + np.array([cx, cy])

    def _car_corners_from_rear(self, xr, yr, yaw):
        """Car rectangle corners given rear-axle position."""
        cx = xr + self.dist_to_center * np.cos(yaw)
        cy = yr + self.dist_to_center * np.sin(yaw)
        return self._rect_corners(cx, cy, self.car_length, self.car_width, yaw)

    def update(self, x, y, yaw, v, obs=None, action=None, steer=None, torque=None,
               step=0, parked=False):
        """Redraw the scene with current car state."""
        ax = self.ax
        ax.clear()

        # 1) Draw virtual obstacles
        for o in self.obs_builder.obstacle_mgr.obstacles:
            cx, cy = float(o["x"]), float(o["y"])
            w, h = float(o["w"]), float(o["h"])
            theta = float(o.get("theta", 0.0))
            kind = o.get("kind", "")

            corners = self._rect_corners(cx, cy, w, h, theta)

            if kind == "curb":
                color, alpha, zorder = "dimgray", 0.7, 4
            elif max(w, h) > 1.5:
                # World walls - skip drawing (too large, clutters view)
                continue
            else:
                color, alpha, zorder = "skyblue", 0.8, 5

            poly = MplPolygon(corners, closed=True, facecolor=color,
                              alpha=alpha, zorder=zorder)
            ax.add_patch(poly)
            border = MplPolygon(corners, closed=True, fill=False,
                                edgecolor="black", linewidth=1, zorder=zorder)
            ax.add_patch(border)

            # Label neighbor cars
            if kind not in ("curb",) and max(w, h) <= 1.5:
                ax.text(cx, cy, "P", ha="center", va="center",
                        fontsize=8, color="navy", zorder=6)

        # 2) Draw parking bay outline (green dashed)
        bay_corners = self._rect_corners(self.bay_x, self.bay_y,
                                         self.car_length, self.car_width,
                                         self.bay_yaw)
        bay_poly = MplPolygon(bay_corners, closed=True, fill=False,
                              edgecolor="green", linewidth=2, linestyle="--",
                              zorder=3)
        ax.add_patch(bay_poly)

        # Goal marker (bay center)
        ax.plot(self.bay_x, self.bay_y, "gx", markersize=10,
                markeredgewidth=2, zorder=10)

        # 3) Draw ego car (blue)
        ego_corners = self._car_corners_from_rear(x, y, yaw)
        ego_color = "limegreen" if parked else "royalblue"
        ego_poly = MplPolygon(ego_corners, closed=True, fill=True,
                              facecolor=ego_color, alpha=0.4,
                              edgecolor=ego_color, linewidth=2, zorder=8)
        ax.add_patch(ego_poly)

        # Heading arrow (from rear axle)
        arrow_len = self.car_length * 0.8
        ax.arrow(x, y,
                 arrow_len * np.cos(yaw), arrow_len * np.sin(yaw),
                 head_width=self.car_width * 0.3,
                 head_length=self.car_length * 0.15,
                 fc=ego_color, ec=ego_color, zorder=9)

        # Rear axle dot
        ax.plot(x, y, "ko", markersize=3, zorder=10)

        # 4) Trajectory trail
        self.trail_x.append(x)
        self.trail_y.append(y)
        if len(self.trail_x) > self.max_trail:
            self.trail_x = self.trail_x[-self.max_trail:]
            self.trail_y = self.trail_y[-self.max_trail:]
        if len(self.trail_x) > 1:
            ax.plot(self.trail_x, self.trail_y, "b-", linewidth=1,
                    alpha=0.4, zorder=2)

        # 5) Info text
        info_lines = [f"Step: {step}"]
        if obs is not None:
            info_lines.append(
                f"along={obs[0]:.3f}  lat={obs[1]:.3f}  "
                f"yaw_err={np.degrees(obs[2]):.1f}\u00b0"
            )
            info_lines.append(f"v={obs[3]:.3f}  dF={obs[4]:.3f}  dL={obs[5]:.3f}  dR={obs[6]:.3f}")
        if steer is not None and torque is not None:
            info_lines.append(f"steer={np.degrees(steer):.1f}\u00b0  torque={torque:.4f}")
        if parked:
            info_lines.append("PARKED!")
        ax.set_title("\n".join(info_lines), fontsize=9, family="monospace")

        # 6) Auto-scale view around the action area
        all_x = [self.bay_x, x]
        all_y = [self.bay_y, y]
        for o in self.obs_builder.obstacle_mgr.obstacles:
            if max(float(o["w"]), float(o["h"])) <= 1.5:
                all_x.append(float(o["x"]))
                all_y.append(float(o["y"]))

        cx_view = np.mean(all_x)
        cy_view = np.mean(all_y)
        spread = max(max(all_x) - min(all_x), max(all_y) - min(all_y))
        margin = max(spread * 0.6, 0.3)  # At least 0.3m margin
        ax.set_xlim(cx_view - margin, cx_view + margin)
        ax.set_ylim(cy_view - margin, cy_view + margin)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.2)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

        # Flush
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def close(self):
        plt.ioff()
        plt.close(self.fig)


# ---------------------------------------------------------------------------
# Policy loader (adapted from rl/eval_policy.py)
# ---------------------------------------------------------------------------
def load_policy(checkpoint_path: str):
    """Load PPO policy from checkpoint for inference."""
    import ray
    from ray.rllib.algorithms.ppo import PPO

    ray.init(ignore_reinit_error=True, include_dashboard=False, num_cpus=2)

    # Import and register the env so RLlib config resolves
    from rl.curriculum_env import create_env_for_rllib
    from ray.tune.registry import register_env
    register_env("curriculum_parking_env", create_env_for_rllib)

    # Load config from checkpoint
    from rl.eval_policy import _load_config_from_checkpoint, _patch_config_for_eval
    cfg = _load_config_from_checkpoint(checkpoint_path)
    if not cfg:
        raise RuntimeError(f"Cannot load config from checkpoint: {checkpoint_path}")

    cfg = _patch_config_for_eval(cfg, force_cpu=True)

    algo = PPO(config=cfg)
    algo.restore(checkpoint_path)
    policy = algo.get_policy()
    print(f"[rl_parking_node] Policy loaded from {checkpoint_path}")
    return policy


# ---------------------------------------------------------------------------
# Virtual obstacle + observation builder
# ---------------------------------------------------------------------------
class ObservationBuilder:
    """
    Builds the 7D observation vector matching the training environment.
    obs = [along, lateral, yaw_err, v, dist_front, dist_left, dist_right]

    Uses the real car state from MoCap + virtual obstacles from config.
    """

    def __init__(self, scene_cfg: dict):
        self.bay_x = float(scene_cfg["bay"]["center_x"])
        self.bay_y = float(scene_cfg["bay"]["center_y"])
        self.bay_yaw = float(scene_cfg["bay"]["yaw"])
        self.bay_center = np.array([self.bay_x, self.bay_y, self.bay_yaw])

        veh = scene_cfg["vehicle"]
        self.L = float(veh["length"])
        self.rear_overhang = float(veh["rear_overhang"])
        self.dist_to_center = self.L / 2.0 - self.rear_overhang

        # Compute rear-axle goal from bay center
        gx = self.bay_x - self.dist_to_center * np.cos(self.bay_yaw)
        gy = self.bay_y - self.dist_to_center * np.sin(self.bay_yaw)
        self.goal = np.array([gx, gy, self.bay_yaw])

        # Build virtual obstacles (same as training)
        obs_cfg = scene_cfg.get("obstacles", {})

        # Re-center world boundaries around the bay.
        # Training had the bay near origin and world at ±1.25m (symmetric).
        # Deployment bay can be anywhere in the MoCap frame, so we shift
        # the world walls to keep the same relative geometry.
        raw_world = scene_cfg.get("world", {})
        half_x = (raw_world.get("x_max", 1.25) - raw_world.get("x_min", -1.25)) / 2.0
        half_y = (raw_world.get("y_max", 1.25) - raw_world.get("y_min", -1.25)) / 2.0
        obs_cfg["world"] = {
            "x_min": self.bay_x - half_x,
            "x_max": self.bay_x + half_x,
            "y_min": self.bay_y - half_y,
            "y_max": self.bay_y + half_y,
        }
        print(f"[ObservationBuilder] World re-centered on bay: "
              f"x=[{obs_cfg['world']['x_min']:.3f}, {obs_cfg['world']['x_max']:.3f}] "
              f"y=[{obs_cfg['world']['y_min']:.3f}, {obs_cfg['world']['y_max']:.3f}]")

        self.obstacle_mgr = ObstacleManager(
            obs_cfg,
            goal=self.bay_center,
            vehicle_params=veh,
        )

        # Build kinematic model (needed for collision checks / distance features)
        self.vehicle_model = KinematicBicycle(veh)

        print(f"[ObservationBuilder] Bay center: ({self.bay_x:.3f}, {self.bay_y:.3f}, "
              f"yaw={np.degrees(self.bay_yaw):.1f} deg)")
        print(f"[ObservationBuilder] Goal (rear axle): ({self.goal[0]:.3f}, {self.goal[1]:.3f})")
        print(f"[ObservationBuilder] dist_to_center: {self.dist_to_center:.4f}m")
        print(f"[ObservationBuilder] Virtual obstacles: {len(self.obstacle_mgr.obstacles)}")
        self.inside_obstacle = False

    def build_obs(self, x: float, y: float, yaw: float, v: float) -> np.ndarray:
        """Build 7D observation from real-world state."""
        state = np.array([x, y, yaw, v], dtype=float)

        # Car center (rear axle -> geometric center)
        cx = x + self.dist_to_center * np.cos(yaw)
        cy = y + self.dist_to_center * np.sin(yaw)

        # Bay-frame errors
        bx, by, byaw = self.bay_center
        cos_b = np.cos(byaw)
        sin_b = np.sin(byaw)
        along = float((cx - bx) * cos_b + (cy - by) * sin_b)
        lateral = float(-(cx - bx) * sin_b + (cy - by) * cos_b)
        yaw_err = float((byaw - yaw + np.pi) % (2 * np.pi) - np.pi)

        # Virtual obstacle distances (ray-casting)
        dist_front, dist_left, dist_right = self.obstacle_mgr.distance_features(state)

        # Flag if car is inside an obstacle (all distances = 0)
        self.inside_obstacle = (dist_front < 0.001 and dist_left < 0.001
                                and dist_right < 0.001)

        obs = np.array([along, lateral, yaw_err, v,
                        dist_front, dist_left, dist_right], dtype=np.float32)
        return obs

    def is_at_goal(self, obs: np.ndarray, tol_along: float, tol_lat: float,
                   tol_yaw: float, tol_v: float) -> bool:
        """Check if car is within success tolerances."""
        along, lateral, yaw_err, v = obs[:4]
        return (abs(along) < tol_along and
                abs(lateral) < tol_lat and
                abs(yaw_err) < tol_yaw and
                abs(v) < tol_v)


# ---------------------------------------------------------------------------
# Action converter
# ---------------------------------------------------------------------------
class ActionConverter:
    """
    Convert policy output [steer, accel] in [-1, 1] to hardware commands.

    Since training used ChronosCar dimensions (max_steer=0.35, max_acc=0.5),
    policy outputs map DIRECTLY to hardware -- no scaling needed.

    Accel -> velocity: integrate accel to get v_target, send to CRS controller.
    The CRS ff_fb_controller handles torque internally (feedforward + feedback),
    including friction compensation. We just send velocity setpoints.
    """

    def __init__(self, scene_cfg: dict):
        veh = scene_cfg["vehicle"]
        self.max_steer = float(veh["max_steer"])    # 0.35 rad
        self.max_acc = float(veh["max_acc"])          # 0.5 m/s^2
        self.max_vel = float(veh["max_vel"])          # 0.5 m/s
        self.dt = 0.1  # 10 Hz control loop

        # Safety limits
        safety = scene_cfg.get("safety", {})
        self.safe_max_steer = float(safety.get("max_steer", 0.30))
        self.max_vel_rate = float(safety.get("max_vel_rate", 0.15))  # m/s per step
        self.max_steer_rate = float(safety.get("max_steer_rate", 0.30))

        # Hardware calibration: the car's physical response overshoots
        # commanded values. Divide commands by measured gains so the car
        # matches the kinematic model used during training.
        cal = scene_cfg.get("calibration", {})
        self.vel_gain = float(cal.get("velocity_gain", 1.0))
        self.steer_gain = float(cal.get("steer_gain", 1.0))

        # State
        self.v_target = 0.0
        self.v_model = 0.0  # Open-loop integrator (what training sees)
        self.last_steer = 0.0
        self.last_vel = 0.0
        self.last_accel_cmd = 0.0  # For logging

    def convert(self, action: np.ndarray, v_current: float) -> tuple:
        """
        Convert policy action to (steer_rad, velocity).

        Args:
            action: [steer, accel] in [-1, 1]
            v_current: current velocity from state estimate

        Returns:
            (steer_rad, velocity) clamped and rate-limited
        """
        # Clip raw actions to [-1, 1] (PPO Gaussian can exceed this range)
        raw_steer = float(np.clip(np.asarray(action[0]).item(), -1.0, 1.0))
        raw_accel = float(np.clip(np.asarray(action[1]).item(), -1.0, 1.0))
        # Scale from [-1, 1] to physical units
        steer_cmd = raw_steer * self.max_steer
        accel_cmd = raw_accel * self.max_acc  # [-0.5, 0.5] m/s²
        self.last_accel_cmd = accel_cmd

        # --- v_model: open-loop integrator (what training sees) ---
        # Training does: v_new = v + accel * dt with NO friction.
        # Track this to compare with hardware reality in logs.
        self.v_model += accel_cmd * self.dt
        self.v_model = float(np.clip(self.v_model, -self.max_vel, self.max_vel))

        # Drift cap: prevent v_model from diverging too far from hardware velocity.
        # Normal transient gap during approach: 0.08–0.18 m/s (car lags commands).
        # Stuck-car scenario (car against wall, friction): gap grows to 0.47–0.50 m/s.
        # At that point policy observes v_model=0.5 while car isn't moving → bad decisions.
        # Cap at 0.35 m/s: handles normal transients, prevents stuck-car divergence.
        MAX_V_DRIFT = 0.35
        self.v_model = float(np.clip(
            self.v_model, v_current - MAX_V_DRIFT, v_current + MAX_V_DRIFT
        ))

        # --- Velocity command: use v_model (open-loop, matches training) ---
        #
        # ROOT BUG with old closed-loop approach (v_target = v_current + accel*dt):
        #   When friction keeps the car nearly stationary (v_hw ≈ 0), v_target
        #   stays near 0 each step regardless of what the policy commands.
        #   Example: v_current=0.03, accel=-0.20 → v_target=0.03-0.02=+0.01 m/s
        #   ros_bridge sees vel_cmd=0.007 → abs < 0.01 → STOP → braking torque
        #   → car barely moves → v_hw stays near 0 → loop repeats forever.
        #
        # FIX: use v_model (same open-loop integrator training uses):
        #   v_model accumulates: -0.02, -0.04, ..., -0.40, -0.50 m/s
        #   ros_bridge receives a meaningful reverse target → applies full torque
        #   → car actually moves → matches training dynamics.
        #
        # Keep v_target for diagnostics only (still logged as v_target).
        self.v_target = v_current + accel_cmd * self.dt
        self.v_target = float(np.clip(self.v_target, -self.max_vel, self.max_vel))

        MIN_CMD = 0.05  # Safety net: minimum command when policy wants movement

        vel_cmd = self.v_model

        # Apply MIN_CMD whenever policy wants to move but command is too tiny
        # (handles very early steps before v_model has built up)
        if 0.0 < abs(vel_cmd) < MIN_CMD:
            vel_cmd = np.sign(vel_cmd) * MIN_CMD

        # Safety clamps
        steer_cmd = np.clip(steer_cmd, -self.safe_max_steer, self.safe_max_steer)
        vel_cmd = float(np.clip(vel_cmd, -self.max_vel, self.max_vel))

        # Rate limiting
        d_steer = steer_cmd - self.last_steer
        steer_cmd = self.last_steer + np.clip(d_steer, -self.max_steer_rate, self.max_steer_rate)

        self.last_steer = steer_cmd
        self.last_vel = vel_cmd

        # Apply hardware calibration correction so car dynamics match
        # the kinematic model used during training.
        steer_cmd = steer_cmd / self.steer_gain
        vel_cmd = vel_cmd / self.vel_gain

        return steer_cmd, vel_cmd

    def reset(self):
        """Reset state (call at episode start)."""
        self.v_target = 0.0
        self.v_model = 0.0
        self.last_steer = 0.0
        self.last_vel = 0.0
        self.last_accel_cmd = 0.0


# ---------------------------------------------------------------------------
# UDP car interface (communicates with ros_bridge.py running in Docker)
# ---------------------------------------------------------------------------
STATE_UDP_PORT = 5800   # Receive car state from bridge
CMD_UDP_PORT   = 5801   # Send commands to bridge


class UDPCarInterface:
    """
    UDP interface to the ROS bridge running in the CRS Docker container.

    No ROS installation needed on the host -- just plain sockets.
    Protocol matches deploy/ros_bridge.py:
      State:   7 doubles [x, y, yaw, v_tot, vx_b, vy_b, steer]
      Command: 3 doubles [velocity, steer, v_actual]
    """

    def __init__(self):
        # State receiver
        self.state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.state_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.state_sock.bind(("0.0.0.0", STATE_UDP_PORT))
        self.state_sock.settimeout(0.5)

        # Command sender
        self.cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.cmd_dest = ("127.0.0.1", CMD_UDP_PORT)

        self._lock = threading.Lock()
        self._latest = None       # dict with x, y, yaw, v_tot, ...
        self._latest_time = None  # time.time()
        self._running = True
        self._recv_count = 0

        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

    def _recv_loop(self):
        while self._running:
            try:
                data, _ = self.state_sock.recvfrom(1024)
                if len(data) == 56:  # 7 doubles
                    vals = struct.unpack("!7d", data)
                    with self._lock:
                        self._latest = {
                            "x": vals[0], "y": vals[1], "yaw": vals[2],
                            "v_tot": vals[3], "vx_b": vals[4],
                            "vy_b": vals[5], "steer": vals[6],
                        }
                        self._latest_time = time.time()
                    self._recv_count += 1
            except socket.timeout:
                continue
            except OSError:
                break  # Socket closed

    def get_state(self):
        """Return (state_dict, timestamp) or (None, None)."""
        with self._lock:
            return self._latest, self._latest_time

    def send_command(self, velocity: float, steer: float, v_actual: float = 0.0):
        """Send velocity + steer + actual velocity command to the bridge.

        v_actual lets ros_bridge distinguish 'accelerate' from 'brake':
        e.g. vel=-0.15 with v_actual=-0.20 means SLOW DOWN (brake),
        while vel=-0.15 with v_actual=0 means SPEED UP (accelerate reverse).
        """
        data = struct.pack("!3d", velocity, steer, v_actual)
        try:
            self.cmd_sock.sendto(data, self.cmd_dest)
        except OSError:
            pass

    def close(self):
        self._running = False
        try:
            self.state_sock.close()
        except OSError:
            pass
        try:
            self.cmd_sock.close()
        except OSError:
            pass


def run_live(policy, obs_builder: ObservationBuilder,
             action_converter: ActionConverter, scene_cfg: dict,
             ghost: bool = False, visualize: bool = True,
             log_dir: str = None):
    """
    Main control loop using UDP bridge (no ROS on host).

    Reads car state from UDP, runs policy inference, sends commands via UDP.
    If log_dir is set, writes a per-step CSV for post-run analysis.
    """
    car = UDPCarInterface()
    viz = LiveVisualizer(obs_builder, scene_cfg) if visualize else None

    # --- CSV logging setup ---
    csv_file = None
    csv_writer = None
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        csv_path = log_path / f"run_{timestamp}.csv"
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "step", "time", "x", "y", "yaw", "v_hw", "v_raw", "v_model",
            "v_target", "along", "lateral", "yaw_err", "dF", "dL", "dR",
            "raw_steer", "raw_accel", "steer_hw", "vel_hw", "v_gap",
            "collision", "settled", "parked",
        ])
        print(f"[rl_parking_node] CSV log: {csv_path}")

    # Success config
    success_cfg = scene_cfg.get("success", {})
    tol_along = float(success_cfg.get("along_tol", 0.043))
    tol_lat = float(success_cfg.get("lateral_tol", 0.043))
    tol_yaw = float(success_cfg.get("yaw_tol", 0.15))
    tol_v = float(success_cfg.get("v_tol", 0.05))
    settled_required = int(success_cfg.get("settled_steps", 5))

    # Safety
    safety_cfg = scene_cfg.get("safety", {})
    stale_timeout = float(safety_cfg.get("stale_timeout", 0.3))

    step = 0
    settled = 0
    parked = False
    obs = None
    v = 0.0
    v_filtered = 0.0
    v_alpha = 0.5  # Low-pass filter for MoCap-derived velocity (0.4 → 0.5: less lag)
    prev_x = None
    prev_y = None
    prev_time = None

    # Velocity source: "mocap" (default, computed from position deltas) or
    # "encoder" (uses vx_b from CRS estimator -- requires wheel encoders on car).
    # Encoder velocity is far superior: no delay, no noise from differentiation.
    # ChronosCar: use "mocap" (CRS vx_b is broken, reads ~28 m/s at standstill)
    # New car:    use "encoder" (wheel encoders give accurate direct measurement)
    vel_source = str(scene_cfg.get("velocity_source", "mocap")).lower()
    use_encoder_vel = (vel_source == "encoder")
    if use_encoder_vel:
        print(f"[rl_parking_node] Velocity source: ENCODER (vx_b from CRS estimator)")
        print(f"[rl_parking_node]   -> No MoCap position derivative needed")
        print(f"[rl_parking_node]   -> LP filter still applied (alpha={v_alpha})")
    else:
        print(f"[rl_parking_node] Velocity source: MOCAP (position deltas, LP filtered)")

    # Collision recovery state
    # In training, collision terminates the episode — the policy NEVER learned
    # to recover from being inside an obstacle. So we use a hand-coded reverse
    # maneuver to back the car out, then resume policy control.
    collision_recovery_steps = 0   # Countdown: >0 means we're in recovery
    COLLISION_REVERSE_STEPS = 15   # 1.5 seconds of reversing at 10Hz
    MAX_COLLISION_RECOVERIES = 5   # Give up after this many collisions
    collision_count = 0

    # Training observation ranges (for OOD detection)
    # Computed from config_env_chronos.yaml spawn_lane config:
    #   along:   [0.34, 1.02] at spawn, passes through 0 when parking
    #   lateral: [0.16, 0.78] at spawn, approaches 0 when parking
    #   yaw_err: [-0.52, 0.52] rad = [-30°, 30°]
    OOD_YAW_LIMIT = np.radians(60)    # Policy is lost beyond this
    OOD_LATERAL_LIMIT = 1.0            # Way beyond any training lateral

    mode = "GHOST (read-only)" if ghost else "LIVE"

    # ---- Comprehensive startup diagnostics ----
    print("=" * 70)
    print(f"  MODE: {mode}")
    print("=" * 70)

    veh = scene_cfg["vehicle"]
    print(f"\n--- Vehicle params (deployment) ---")
    print(f"  length={veh['length']}, width={veh['width']}, "
          f"wheelbase={veh['wheelbase']}, rear_overhang={veh['rear_overhang']}")
    print(f"  max_steer={veh['max_steer']} rad ({np.degrees(veh['max_steer']):.1f} deg), "
          f"max_vel={veh['max_vel']} m/s, max_acc={veh['max_acc']} m/s²")
    print(f"  dist_to_center={obs_builder.dist_to_center:.4f}m")

    print(f"\n--- Bay (from parking_scene.yaml) ---")
    print(f"  center: ({obs_builder.bay_x:.4f}, {obs_builder.bay_y:.4f})")
    print(f"  yaw: {np.degrees(obs_builder.bay_yaw):.1f} deg ({obs_builder.bay_yaw:.4f} rad)")
    print(f"  goal (rear axle): ({obs_builder.goal[0]:.4f}, {obs_builder.goal[1]:.4f})")

    print(f"\n--- Action converter ---")
    print(f"  max_steer={action_converter.max_steer:.3f} rad, "
          f"max_acc={action_converter.max_acc:.3f} m/s², "
          f"max_vel={action_converter.max_vel:.3f} m/s, dt={action_converter.dt}")
    print(f"  safety: max_steer={action_converter.safe_max_steer:.3f} rad, "
          f"steer_rate={action_converter.max_steer_rate:.3f} rad/step")
    if action_converter.steer_gain != 1.0 or action_converter.vel_gain != 1.0:
        print(f"  calibration: steer_gain={action_converter.steer_gain:.3f}x, "
              f"vel_gain={action_converter.vel_gain:.3f}x "
              f"(commands divided by these factors)")

    print(f"\n--- Success tolerances ---")
    print(f"  along={tol_along:.4f}m, lateral={tol_lat:.4f}m, "
          f"yaw={np.degrees(tol_yaw):.1f}deg, v={tol_v:.3f}m/s, "
          f"settled_steps={settled_required}")

    print(f"\n--- Obstacles ---")
    for i, o in enumerate(obs_builder.obstacle_mgr.obstacles):
        kind = o.get("kind", "wall" if max(o["w"], o["h"]) > 1.0 else "neighbor")
        print(f"  [{i}] {kind}: center=({o['x']:.3f},{o['y']:.3f}) "
              f"size=({o['w']:.3f}x{o['h']:.3f}) theta={np.degrees(o.get('theta',0)):.1f}deg")

    # Training reference (for sanity check)
    print(f"\n--- Training reference (config_env_chronos.yaml) ---")
    print(f"  max_steer=0.35 rad, max_vel=0.5 m/s, max_acc=0.5 m/s²")
    print(f"  NO steer rate limiting in training")
    print(f"  Deployment steer clamp={action_converter.safe_max_steer:.2f} "
          f"(should be 0.35 to match training)")
    print(f"  Deployment max_vel={action_converter.max_vel:.2f} "
          f"(should be 0.50 to match training)")
    if abs(action_converter.safe_max_steer - 0.35) > 0.01:
        print(f"  *** WARNING: steer clamp {action_converter.safe_max_steer} != training 0.35 ***")
    if abs(action_converter.max_vel - 0.5) > 0.01:
        print(f"  *** WARNING: max_vel {action_converter.max_vel} != training 0.50 ***")

    print("=" * 70)

    print(f"\n[rl_parking_node] Listening for state on UDP:{STATE_UDP_PORT}")
    print(f"[rl_parking_node] Sending commands to UDP:{CMD_UDP_PORT}")
    print(f"[rl_parking_node] Waiting for car state...")

    try:
        while True:
            loop_start = time.time()

            state, state_time = car.get_state()

            # Wait for first state
            if state is None:
                time.sleep(0.1)
                continue

            # Stale check
            age = time.time() - state_time
            if age > stale_timeout:
                if step % 20 == 0:
                    print(f"[WARN] Stale state ({age:.2f}s), sending stop")
                if not ghost:
                    car.send_command(0.0, 0.0)
                time.sleep(0.1)
                continue

            if parked:
                if not ghost:
                    car.send_command(0.0, 0.0)
                time.sleep(0.1)
                continue

            x = state["x"]
            y = state["y"]
            raw_yaw = state["yaw"]

            # Skip if MoCap lost tracking (NaN values)
            if any(np.isnan(v) for v in [x, y, raw_yaw]):
                if step % 20 == 0:
                    print("[WARN] NaN in state (MoCap lost tracking), skipping")
                if not ghost:
                    car.send_command(0.0, 0.0)
                time.sleep(0.1)
                continue

            # Normalize yaw to [-π, π] (MoCap can give unwrapped values)
            yaw = float((raw_yaw + np.pi) % (2 * np.pi) - np.pi)

            # --- Velocity estimation ---
            now = time.time()
            max_vel = action_converter.max_vel

            if use_encoder_vel:
                # ENCODER mode (new car with wheel encoders):
                # Use vx_b from CRS estimator directly — accurate, no latency.
                # Still apply a light LP filter to suppress encoder noise.
                v_raw = float(state.get("vx_b", 0.0))
                # Sanity check: if vx_b looks broken (ChronosCar shows ~28 m/s),
                # warn and fall back to MoCap derivative.
                if abs(v_raw) > max_vel * 2.0:
                    if step % 20 == 0:
                        print(f"[WARN] Encoder vx_b={v_raw:.1f} looks broken! "
                              f"Falling back to MoCap derivative. "
                              f"Check if new car CRS estimator is using wheel encoders.")
                    # Fall back
                    if prev_x is not None and prev_time is not None and (now - prev_time) > 0.001:
                        dx, dy = x - prev_x, y - prev_y
                        spd = np.sqrt(dx*dx + dy*dy) / (now - prev_time)
                        vb = dx * np.cos(yaw) + dy * np.sin(yaw)
                        v_raw = np.sign(vb) * spd if spd > 0.005 else 0.0
                    else:
                        v_raw = 0.0
                v_filtered = v_alpha * v_raw + (1.0 - v_alpha) * v_filtered
            else:
                # MOCAP mode (ChronosCar — CRS velocity is broken ~28 m/s idle):
                # Compute velocity from position derivative + LP filter.
                if prev_x is not None and prev_time is not None:
                    dt_real = now - prev_time
                    if dt_real > 0.001:
                        dx = x - prev_x
                        dy = y - prev_y
                        speed = np.sqrt(dx * dx + dy * dy) / dt_real
                        v_body = dx * np.cos(yaw) + dy * np.sin(yaw)
                        v_raw = np.sign(v_body) * speed if speed > 0.005 else 0.0
                    else:
                        v_raw = 0.0
                else:
                    v_raw = 0.0
                v_filtered = v_alpha * v_raw + (1.0 - v_alpha) * v_filtered

            prev_x = x
            prev_y = y
            prev_time = now
            v = float(np.clip(v_filtered, -max_vel, max_vel))

            # Build observation — use v_hw (MoCap velocity), NOT v_model.
            # v_hw and position (both from MoCap) are self-consistent: if the car
            # moved slowly, v_hw is small AND along barely changed. The policy sees
            # "I'm slow and far from bay → keep reversing harder" → sustained reverse.
            #
            # v_model creates an internal inconsistency: policy thinks it's going
            # -0.37 m/s (v_model) but MoCap position barely changed (car lagged due
            # to friction). Policy then "corrects" this by alternating accel sign
            # (braking then reversing), which produces the forward/backward oscillation.
            #
            # Note: vel_cmd still uses v_model (Fix 1) to give ros_bridge a real
            # reverse target even when v_hw ≈ 0 due to friction at standstill.
            obs = obs_builder.build_obs(x, y, yaw, v)

            # Guard against NaN in observation (can happen from edge cases)
            if np.any(np.isnan(obs)):
                print(f"[WARN] NaN in obs: {obs}, skipping step")
                if not ghost:
                    car.send_command(0.0, 0.0)
                time.sleep(0.1)
                continue

            step += 1

            # Log first step with full detail
            if step == 1:
                print(f"\n--- First state received ---")
                print(f"  Car position: x={x:.4f} y={y:.4f} yaw={np.degrees(yaw):.1f}deg")
                # Distance from car center to bay center
                cx = x + obs_builder.dist_to_center * np.cos(yaw)
                cy = y + obs_builder.dist_to_center * np.sin(yaw)
                dist_to_bay = np.sqrt((cx - obs_builder.bay_x)**2 +
                                      (cy - obs_builder.bay_y)**2)
                print(f"  Car center: cx={cx:.4f} cy={cy:.4f}")
                print(f"  Distance to bay center: {dist_to_bay:.3f}m")
                print(f"  Obs: along={obs[0]:.4f} lat={obs[1]:.4f} "
                      f"yaw_err={np.degrees(obs[2]):.1f}deg")
                print(f"  Dist features: dF={obs[4]:.3f} dL={obs[5]:.3f} dR={obs[6]:.3f}")
                print()

            # Log periodically (every 5 steps = 0.5s for better visibility)
            if step % 5 == 0:
                print(
                    f"[step {step:3d}] pos=({x:.3f},{y:.3f}) yaw={np.degrees(yaw):.1f}deg "
                    f"| along={obs[0]:+.3f} lat={obs[1]:+.3f} yaw_err={np.degrees(obs[2]):+.1f}deg"
                )
                print(
                    f"           v_hw={v:.3f} v_model={action_converter.v_model:.3f} "
                    f"v_target={action_converter.v_target:.3f} (v_raw={v_raw:.3f}) "
                    f"| dF={obs[4]:.3f} dL={obs[5]:.3f} dR={obs[6]:.3f}"
                )

            # --- Collision recovery ---
            # In training, collision = episode termination. The policy never
            # learned to handle post-collision states (dF=dL=dR=0 is undefined).
            # Instead of stopping forever, we use a hand-coded reverse maneuver
            # to back the car out, then resume policy control.
            if collision_recovery_steps > 0:
                # We're in recovery: reverse along current heading
                collision_recovery_steps -= 1
                if not ghost:
                    # Reverse slowly with zero steer to back straight out.
                    # Divide by vel_gain so car actually reverses at 0.12 m/s.
                    car.send_command(-0.12 / action_converter.vel_gain, 0.0)
                if collision_recovery_steps == 0:
                    print(f"[rl_parking_node] Recovery complete. Resuming policy.")
                    action_converter.reset()  # Reset v_target + v_model
                    v_filtered = 0.0          # Reset velocity filter
                elif collision_recovery_steps % 5 == 0:
                    print(f"[rl_parking_node] Recovering... {collision_recovery_steps} steps left")
                # Maintain loop timing
                elapsed = time.time() - loop_start
                sleep_time = 0.1 - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                continue

            if obs_builder.inside_obstacle:
                collision_count += 1
                if collision_count > MAX_COLLISION_RECOVERIES:
                    print(f"[rl_parking_node] Too many collisions ({collision_count}). Stopping.")
                    if not ghost:
                        car.send_command(0.0, 0.0)
                    break
                # Near-bay collision: car was essentially trying to park but clipped a
                # neighbor. Only nudge back briefly — don't undo all approach progress.
                # Normal collision (far from bay): full reversal to clear the obstacle.
                near_bay = abs(obs[0]) < 0.12 and abs(obs[2]) < np.radians(30)
                steps_to_reverse = 5 if near_bay else COLLISION_REVERSE_STEPS
                print(f"[rl_parking_node] COLLISION #{collision_count}: "
                      f"along={obs[0]:.3f} lat={obs[1]:.3f} yaw_err={np.degrees(obs[2]):.1f}deg. "
                      f"Auto-reversing for {steps_to_reverse} steps "
                      f"({'near-bay short' if near_bay else 'full'} recovery)...")
                collision_recovery_steps = steps_to_reverse
                if not ghost:
                    car.send_command(0.0, 0.0)  # Stop first
                continue

            # --- Out-of-distribution detection ---
            # The policy was trained with yaw_err in [-30°, 30°] and lateral
            # in [0.16, 0.78]. Beyond these ranges, policy output is undefined.
            yaw_err_deg = np.degrees(obs[2])
            if abs(obs[2]) > OOD_YAW_LIMIT:
                if step % 10 == 0:
                    print(f"[WARN] OUT OF DISTRIBUTION: yaw_err={yaw_err_deg:.1f}deg "
                          f"(training max: ±{np.degrees(OOD_YAW_LIMIT):.0f}deg). "
                          f"Policy output unreliable. Reposition car closer to bay heading.")
            if abs(obs[1]) > OOD_LATERAL_LIMIT:
                if step % 10 == 0:
                    print(f"[WARN] OUT OF DISTRIBUTION: lateral={obs[1]:.3f}m "
                          f"(training max: ~0.78m). Reposition car.")

            # Check success
            if obs_builder.is_at_goal(obs, tol_along, tol_lat, tol_yaw, tol_v):
                settled += 1
                if settled >= settled_required:
                    print("[rl_parking_node] PARKED SUCCESSFULLY!")
                    parked = True
                    if not ghost:
                        car.send_command(0.0, 0.0)
                    continue
                # In-tolerance but not yet settled: send stop, skip policy.
                # Critical: if policy runs here it fires a forward/reverse command
                # that kicks the car out of the bay and into a collision.
                if step % 5 == 0:
                    print(f"[rl_parking_node] Settling {settled}/{settled_required} — "
                          f"holding stop (along={obs[0]:+.3f} lat={obs[1]:+.3f} "
                          f"yaw={np.degrees(obs[2]):+.1f}deg v={v:+.3f})")
                if not ghost:
                    car.send_command(0.0, 0.0)
                action_converter.reset()  # Keep v_model=0 while stopped
                continue
            else:
                settled = 0

            # Run policy inference
            # compute_single_action returns (action_array, state_outs, info)
            action_tuple = policy.compute_single_action(obs, explore=False)
            action = action_tuple[0]  # Extract the action array (shape: (2,))

            # Convert to hardware commands
            steer, vel_cmd = action_converter.convert(action, v)

            # Log action with training-vs-hardware comparison
            raw0 = float(np.asarray(action[0]).item())
            raw1 = float(np.asarray(action[1]).item())
            accel_physical = float(np.clip(raw1, -1.0, 1.0)) * action_converter.max_acc
            if step % 5 == 0:
                tag = "ghost" if ghost else "cmd"
                # Show: what policy commanded, what we're sending, and the gap
                v_gap = action_converter.v_model - v
                steer_before_cal = action_converter.last_steer  # pre-calibration
                print(
                    f"  [{tag}] raw=[{raw0:+.3f},{raw1:+.3f}] "
                    f"accel={accel_physical:+.3f} steer_pol={np.degrees(steer_before_cal):.1f}deg "
                    f"→ steer_hw={np.degrees(steer):.1f}deg vel_hw={vel_cmd:.3f}"
                )
                if abs(v_gap) > 0.02:
                    print(
                        f"  [gap] v_model-v_hw={v_gap:+.3f} "
                        f"(training expects {action_converter.v_model:.3f}, "
                        f"hardware has {v:.3f})"
                    )

            # CSV logging (every step for post-run analysis)
            if csv_writer is not None:
                v_gap = action_converter.v_model - v
                csv_writer.writerow([
                    step, f"{time.time():.3f}",
                    f"{x:.5f}", f"{y:.5f}", f"{yaw:.4f}",
                    f"{v:.4f}", f"{v_raw:.4f}",
                    f"{action_converter.v_model:.4f}",
                    f"{action_converter.v_target:.4f}",
                    f"{obs[0]:.5f}", f"{obs[1]:.5f}",
                    f"{obs[2]:.4f}",
                    f"{obs[4]:.4f}", f"{obs[5]:.4f}", f"{obs[6]:.4f}",
                    f"{raw0:.4f}", f"{raw1:.4f}",
                    f"{steer:.4f}", f"{vel_cmd:.4f}",
                    f"{v_gap:.4f}",
                    int(obs_builder.inside_obstacle), settled,
                    int(parked),
                ])

            # Send command (include v_actual so ros_bridge can brake properly)
            if not ghost:
                car.send_command(vel_cmd, steer, v)

            # Update live visualization
            if viz is not None:
                viz.update(
                    x, y, yaw, v, obs=obs, action=action,
                    steer=steer, torque=vel_cmd, step=step, parked=parked,
                )

            # Maintain 10 Hz loop
            elapsed = time.time() - loop_start
            sleep_time = 0.1 - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[rl_parking_node] Stopped by user")
    finally:
        # Send stop command
        if not ghost:
            car.send_command(0.0, 0.0)
            time.sleep(0.05)
            car.send_command(0.0, 0.0)  # Send twice for safety
        car.close()
        if viz is not None:
            viz.close()

        # Episode summary
        print(f"\n{'=' * 60}")
        print(f"  EPISODE SUMMARY")
        print(f"{'=' * 60}")
        print(f"  Total steps:   {step}")
        print(f"  Duration:      {step * 0.1:.1f}s")
        print(f"  Collisions:    {collision_count}")
        print(f"  Parked:        {'YES' if parked else 'NO'}")
        if step > 0 and obs is not None:
            print(f"  Final errors:")
            print(f"    along:   {obs[0]:+.4f}m  (tol: {tol_along:.4f}m)")
            print(f"    lateral: {obs[1]:+.4f}m  (tol: {tol_lat:.4f}m)")
            print(f"    yaw_err: {np.degrees(obs[2]):+.1f}deg  (tol: {np.degrees(tol_yaw):.1f}deg)")
            print(f"    v:       {v:.4f}m/s  (tol: {tol_v:.4f}m/s)")
        print(f"{'=' * 60}")

        if csv_file is not None:
            csv_file.close()
            print(f"[rl_parking_node] CSV log saved: {csv_path}")


# ---------------------------------------------------------------------------
# Dry-run mode (no ROS needed)
# ---------------------------------------------------------------------------
def dry_run(checkpoint_path: str, scene_cfg: dict, visualize: bool = True):
    """Test checkpoint loading and observation computation without ROS."""
    print("=" * 60)
    print("DRY RUN: Testing policy + observation pipeline")
    print("=" * 60)

    # Load policy
    print("\n[1/3] Loading policy...")
    policy = load_policy(checkpoint_path)
    print("  Policy loaded OK")

    # Build observation builder
    print("\n[2/3] Building observation pipeline...")
    obs_builder = ObservationBuilder(scene_cfg)
    print("  ObservationBuilder OK")

    # Live visualizer
    viz = LiveVisualizer(obs_builder, scene_cfg) if visualize else None

    # Test observation at various positions
    print("\n[3/3] Testing observations...")
    test_poses = [
        (scene_cfg["bay"]["center_x"], scene_cfg["bay"]["center_y"],
         scene_cfg["bay"]["yaw"], 0.0, "At bay center"),
        (scene_cfg["bay"]["center_x"] + 0.5, scene_cfg["bay"]["center_y"] + 0.3,
         0.0, 0.0, "0.5m ahead, 0.3m lateral"),
        (scene_cfg["bay"]["center_x"] + 0.2, scene_cfg["bay"]["center_y"] + 0.1,
         0.3, 0.1, "Near bay, yaw=17deg, v=0.1"),
    ]

    action_conv = ActionConverter(scene_cfg)

    for i, (x, y, yaw, v, desc) in enumerate(test_poses):
        obs = obs_builder.build_obs(x, y, yaw, v)
        action_tuple = policy.compute_single_action(obs, explore=False)
        action = action_tuple[0]  # Extract action array from (action, state_outs, info)
        steer, vel_cmd = action_conv.convert(action, v)

        print(f"\n  Pose: {desc}")
        print(f"    State: x={x:.3f} y={y:.3f} yaw={np.degrees(yaw):.1f}deg v={v:.3f}")
        print(f"    Obs:   along={obs[0]:.3f} lat={obs[1]:.3f} yaw_err={np.degrees(obs[2]):.1f}deg")
        print(f"           dist: F={obs[4]:.3f} L={obs[5]:.3f} R={obs[6]:.3f}")
        a0 = float(np.asarray(action[0]).item())
        a1 = float(np.asarray(action[1]).item())
        print(f"    Action: raw=[{a0:.3f}, {a1:.3f}]")
        print(f"    Output: steer={np.degrees(steer):.1f}deg vel={vel_cmd:.3f}")

        at_goal = obs_builder.is_at_goal(obs, 0.043, 0.043, 0.15, 0.05)
        print(f"    At goal: {at_goal}")

        # Show each test pose in the visualizer
        if viz is not None:
            viz.update(x, y, yaw, v, obs=obs, action=action,
                       steer=steer, torque=vel_cmd, step=i,
                       parked=at_goal)
            plt.pause(1.5)  # Pause to let user see each pose

        action_conv.reset()

    print("\n" + "=" * 60)
    print("DRY RUN PASSED - Ready for hardware deployment")
    print("=" * 60)

    if viz is not None:
        print("\n[viz] Close the matplotlib window to exit.")
        plt.ioff()
        plt.show()  # Block until user closes window


# ---------------------------------------------------------------------------
# Calibrate mode: read car position from MoCap and save as bay center
# ---------------------------------------------------------------------------
def calibrate_bay(scene_path: str):
    """
    Read the car's current MoCap position via UDP and save it as the bay center.

    Workflow:
      1. Place the car at the desired parking spot
      2. Run with --calibrate
      3. Car's position + yaw become the bay center
      4. Move car away, then run ghost/live mode
    """
    print("=" * 60)
    print("CALIBRATE: Place the car at the desired parking spot")
    print("=" * 60)
    print()
    print("Reading car position from MoCap via UDP...")
    print(f"Listening on UDP port {STATE_UDP_PORT}...")

    car = UDPCarInterface()

    # Collect a few readings and average them for stability
    readings = []
    target_readings = 20  # 2 seconds at 10 Hz

    try:
        while len(readings) < target_readings:
            state, state_time = car.get_state()
            if state is None:
                time.sleep(0.1)
                continue

            age = time.time() - state_time
            if age > 0.5:
                time.sleep(0.1)
                continue

            readings.append((state["x"], state["y"], state["yaw"]))
            print(f"\r  Reading {len(readings)}/{target_readings}: "
                  f"x={state['x']:.4f}  y={state['y']:.4f}  "
                  f"yaw={np.degrees(state['yaw']):.1f}deg", end="", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        if not readings:
            print("\nNo readings collected. Exiting.")
            car.close()
            return
    finally:
        car.close()

    # Average the readings
    xs = [r[0] for r in readings]
    ys = [r[1] for r in readings]
    yaws = [r[2] for r in readings]
    bay_x = float(np.mean(xs))
    bay_y = float(np.mean(ys))
    bay_yaw_raw = float(np.mean(yaws))

    # Normalize yaw to [-π, π] (MoCap can give unwrapped values like -1252°)
    bay_yaw = float((bay_yaw_raw + np.pi) % (2 * np.pi) - np.pi)

    print(f"\n\nCRS position (averaged from {len(readings)} readings):")
    print(f"  x:   {bay_x:.4f} m")
    print(f"  y:   {bay_y:.4f} m")
    print(f"  yaw: {np.degrees(bay_yaw):.1f} deg ({bay_yaw:.4f} rad)")
    if abs(bay_yaw_raw - bay_yaw) > 0.01:
        print(f"  (raw yaw was {np.degrees(bay_yaw_raw):.1f} deg, normalized to [-180, 180])")

    # CRITICAL: Convert CRS position to geometric body center.
    # The CRS estimator outputs the rear-axle position, but in training
    # bay_center is the geometric center of the parking slot. The observation
    # computes cx = x + dist_to_center * cos(yaw), and along = cx - bay_center.
    # If bay_center = raw CRS position (rear axle), along = dist_to_center = 0.045
    # even when perfectly parked! This phantom 4.5cm offset prevents convergence.
    # Fix: shift bay_center forward by dist_to_center to match training semantics.
    scene_file = Path(scene_path)
    with open(scene_file) as f:
        scene_cfg = yaml.safe_load(f)

    veh = scene_cfg.get("vehicle", {})
    L = float(veh.get("length", 0.13))
    rear_overhang = float(veh.get("rear_overhang", 0.02))
    dist_to_center = L / 2.0 - rear_overhang

    bay_center_x = bay_x + dist_to_center * np.cos(bay_yaw)
    bay_center_y = bay_y + dist_to_center * np.sin(bay_yaw)

    print(f"\nBay center (body center = CRS pos + {dist_to_center:.4f}m forward):")
    print(f"  x:   {bay_center_x:.4f} m")
    print(f"  y:   {bay_center_y:.4f} m")
    print(f"  yaw: {np.degrees(bay_yaw):.1f} deg ({bay_yaw:.4f} rad)")

    # Load existing scene config and update bay values
    # IMPORTANT: convert to plain Python float (not numpy scalar)
    # or yaml.dump will serialize numpy binary blobs that safe_load can't read
    scene_cfg["bay"]["center_x"] = round(float(bay_center_x), 4)
    scene_cfg["bay"]["center_y"] = round(float(bay_center_y), 4)
    scene_cfg["bay"]["yaw"] = round(float(bay_yaw), 4)

    with open(scene_file, "w") as f:
        # Write with comments preserved as much as possible
        f.write(f"# parking_scene.yaml\n")
        f.write(f"# Bay calibrated from MoCap on {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Car was placed at bay center and position was averaged over {len(readings)} readings.\n")
        f.write(f"#\n")
        f.write(f"# To recalibrate: place car at parking spot, run:\n")
        f.write(f"#   python deploy/rl_parking_node.py --calibrate --checkpoint <any_checkpoint>\n\n")
        yaml.dump(scene_cfg, f, default_flow_style=False, sort_keys=False)

    print(f"\nSaved to {scene_file}")
    print()
    print("Next steps:")
    print("  1. Move the car ~50-100cm away from the bay")
    print("  2. Run ghost mode to verify:")
    print(f"     ./deploy/run_parking.sh --ghost --checkpoint <checkpoint>")
    print("  3. Check that along/lateral approach 0 when car is at bay")
    print()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="RL Parking Deployment Node")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to PPO checkpoint directory (required for ghost/live/dry-run)")
    parser.add_argument("--scene", default="deploy/parking_scene.yaml",
                        help="Path to parking scene config")
    parser.add_argument("--calibrate", action="store_true",
                        help="Read car position from MoCap and save as bay center")
    parser.add_argument("--dry-run", action="store_true",
                        help="Test checkpoint loading + observation pipeline (no hardware)")
    parser.add_argument("--ghost", action="store_true",
                        help="Read state via UDP but don't send commands")
    parser.add_argument("--no-viz", action="store_true",
                        help="Disable live visualization window")
    parser.add_argument("--log-dir", default=None,
                        help="Directory for CSV run logs (e.g. deploy/logs/)")
    args = parser.parse_args()

    # Calibrate mode: just read position and save, no checkpoint needed
    if args.calibrate:
        calibrate_bay(args.scene)
        return

    # Load scene config
    scene_path = Path(args.scene)
    if not scene_path.exists():
        print(f"ERROR: Scene config not found: {scene_path}")
        sys.exit(1)

    with open(scene_path) as f:
        scene_cfg = yaml.safe_load(f)

    visualize = not args.no_viz

    if not args.checkpoint:
        print("ERROR: --checkpoint is required for ghost/live/dry-run mode")
        sys.exit(1)

    if args.dry_run:
        dry_run(args.checkpoint, scene_cfg, visualize=visualize)
        return

    # Live / ghost mode via UDP bridge (no ROS needed on host)
    print(f"[rl_parking_node] Loading policy from {args.checkpoint}...")
    policy = load_policy(args.checkpoint)

    obs_builder = ObservationBuilder(scene_cfg)
    action_converter = ActionConverter(scene_cfg)

    run_live(
        policy=policy,
        obs_builder=obs_builder,
        action_converter=action_converter,
        scene_cfg=scene_cfg,
        ghost=args.ghost,
        visualize=visualize,
        log_dir=args.log_dir,
    )


if __name__ == "__main__":
    main()
