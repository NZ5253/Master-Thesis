#!/usr/bin/env python3
"""
ROS <-> UDP Bridge for RL Parking Deployment.

Runs INSIDE the CRS Docker container where ROS Noetic + crs_msgs are available.
Bridges two ROS topics to/from UDP so the RL node on the host needs no ROS.

Protocol:
  State  (bridge -> host):  7 doubles [x, y, yaw, v_tot, vx_b, vy_b, steer] = 56 bytes
  Command (host -> bridge): 3 doubles [velocity, steer, v_actual] = 24 bytes (backward compat: 2 doubles = 16 bytes)

Usage (inside Docker):
  source /opt/ros/noetic/setup.bash
  source /code/devel/setup.bash
  python3 /project/deploy/ros_bridge.py [--namespace BEN_CAR_WIFI]
"""

import argparse
import math
import struct
import socket
import threading
import signal
import sys

import rospy
from crs_msgs.msg import car_state_cart, car_input

STATE_PORT = 5800   # Bridge sends state here (host listens)
CMD_PORT   = 5801   # Bridge receives commands here (host sends)


class ROSUDPBridge:
    def __init__(self, namespace="BEN_CAR_WIFI"):
        self.ns = namespace

        # --- UDP sockets ---
        # State sender (non-blocking)
        self.state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.state_dest = ("127.0.0.1", STATE_PORT)

        # Command receiver (blocking with timeout)
        self.cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.cmd_sock.bind(("0.0.0.0", CMD_PORT))
        self.cmd_sock.settimeout(0.5)

        self.running = True
        self.state_count = 0
        self.cmd_count = 0
        self.last_vx_b = 0.0  # Track car velocity for active braking

        # Torque impulse state (Professor Frank's suggestion).
        # A brief high-torque burst breaks static friction at standstill.
        # More effective than raising the minimum velocity setpoint because
        # it acts directly at the motor level, bypassing the PID setpoint lag.
        self.impulse_remaining = 0      # Countdown: >0 means impulse active
        self.IMPULSE_TORQUE = 0.20      # Pulse torque (N*m, above static threshold)
        self.IMPULSE_STEPS = 2          # Duration: 2 steps = 0.2s at 10Hz
        self.STANDSTILL_THRESH = 0.02   # m/s: car is "stopped"

        # --- ROS setup ---
        rospy.init_node("rl_udp_bridge", anonymous=True)

        self.state_sub = rospy.Subscriber(
            f"/{self.ns}/estimation_node/best_state",
            car_state_cart,
            self._state_cb,
            queue_size=1,
        )

        self.cmd_pub = rospy.Publisher(
            f"/{self.ns}/control_input",
            car_input,
            queue_size=1,
        )

        # Start command receiver thread
        self.cmd_thread = threading.Thread(target=self._recv_commands, daemon=True)
        self.cmd_thread.start()

        rospy.loginfo(
            f"[ros_bridge] Ready. ns=/{self.ns}  "
            f"state->UDP:{STATE_PORT}  cmd<-UDP:{CMD_PORT}"
        )

    def _state_cb(self, msg):
        """Forward car state from ROS topic to UDP."""
        try:
            data = struct.pack(
                "!7d",
                msg.x, msg.y, msg.yaw,
                msg.v_tot, msg.vx_b, msg.vy_b,
                msg.steer,
            )
            self.state_sock.sendto(data, self.state_dest)
            self.last_vx_b = msg.vx_b  # Track for braking decisions
            self.state_count += 1
            if self.state_count % 100 == 1:
                rospy.loginfo(
                    f"[ros_bridge] state #{self.state_count}: "
                    f"x={msg.x:.3f} y={msg.y:.3f} yaw={msg.yaw:.2f} v={msg.v_tot:.3f}"
                )
        except Exception as e:
            rospy.logwarn(f"[ros_bridge] state send error: {e}")

    def _recv_commands(self):
        """Receive commands from UDP and publish to ROS.

        Protocol:
          Old: 2 doubles [velocity, steer]         = 16 bytes (backward compat)
          New: 3 doubles [velocity, steer, v_actual] = 24 bytes

        v_actual (actual car velocity from MoCap) lets us distinguish:
          - 'accelerate reverse' (v_actual=0, velocity=-0.15 → need reverse torque)
          - 'brake from reverse' (v_actual=-0.20, velocity=-0.10 → need FORWARD torque)
        Without v_actual, all negative velocity gets reverse torque, even braking.
        """
        while self.running and not rospy.is_shutdown():
            try:
                data, addr = self.cmd_sock.recvfrom(1024)

                # Parse packet (backward compatible)
                v_actual = None
                if len(data) == 24:  # New 3-double protocol
                    velocity, steer, v_actual = struct.unpack("!3d", data)
                elif len(data) == 16:  # Old 2-double protocol
                    velocity, steer = struct.unpack("!2d", data)
                else:
                    rospy.logwarn(f"[ros_bridge] unexpected packet size: {len(data)}")
                    continue

                # --- Torque computation ---
                # CRS ff_fb_controller supports two modes:
                #   - velocity = number  → PID active (setpoint tracking)
                #   - velocity = nan     → PID disabled (pure torque control)
                # We use nan (pure torque) during impulse and reverse phases
                # to prevent the PID from fighting our direct torque commands.
                # Steady-state forward uses PID for stable velocity tracking.
                A_TORQUE = 6.44026018
                B_TORQUE = 0.13732343
                MIN_TORQUE = 0.10   # Steady-state minimum to sustain motion
                BRAKE_TORQUE = 0.10  # Active braking force

                # Resolve actual velocity (use v_actual from host if available,
                # else fall back to CRS estimate which is unreliable for ChronosCar)
                v_est = v_actual if v_actual is not None else self.last_vx_b
                car_stopped = abs(v_est) < self.STANDSTILL_THRESH

                if abs(velocity) < 0.01:
                    # ---- STOP ----
                    # Cancel any pending impulse -- policy wants to stop.
                    self.impulse_remaining = 0
                    cmd_velocity = 0.0   # PID at 0 helps hold car still
                    if v_est > 0.03:
                        ff_torque = -BRAKE_TORQUE
                    elif v_est < -0.03:
                        ff_torque = BRAKE_TORQUE
                    else:
                        ff_torque = 0.0

                elif velocity > 0:
                    # ---- FORWARD ----
                    if car_stopped and velocity > 0.02:
                        # Car at standstill but policy wants forward motion.
                        # Fire torque impulse to overcome static friction.
                        # KEY: set velocity=nan to DISABLE CRS PID during impulse.
                        # If we keep cmd_velocity=velocity (e.g. 0.05), the PID
                        # fights our impulse burst: it sees car exceeding 0.05 m/s
                        # and immediately brakes back, killing the breakaway force.
                        # Insight from friend's code: velocity=nan → pure torque control.
                        if self.impulse_remaining <= 0:
                            self.impulse_remaining = self.IMPULSE_STEPS
                            rospy.loginfo(
                                f"[ros_bridge] IMPULSE FWD: vel={velocity:.3f} "
                                f"v_est={v_est:.3f} → burst {self.IMPULSE_TORQUE:.2f} "
                                f"for {self.IMPULSE_STEPS} steps (PID disabled)"
                            )
                        cmd_velocity = math.nan   # Disable PID, let torque act freely
                        ff_torque = self.IMPULSE_TORQUE
                        self.impulse_remaining -= 1
                    else:
                        # Steady-state forward: PID setpoint + feedforward.
                        # Calibration-corrected velocity setpoint is well-tuned;
                        # keep PID active for stable velocity tracking.
                        self.impulse_remaining = 0
                        cmd_velocity = velocity
                        ff_torque = (velocity - B_TORQUE) / A_TORQUE
                        ff_torque = max(ff_torque, MIN_TORQUE)

                else:
                    # ---- REVERSE ----
                    # CRS PID is forward-only: velocity=nan disables it completely.
                    # (Previously cmd_velocity=0.0; nan is more explicit/correct.)
                    # Direct torque gives clean reverse speed control.
                    cmd_velocity = math.nan
                    abs_vel = min(abs(velocity), 0.25)

                    if v_est < velocity - 0.02:
                        # Car is going FASTER in reverse than requested → BRAKE
                        self.impulse_remaining = 0
                        ff_torque = BRAKE_TORQUE
                    elif car_stopped:
                        # At standstill, fire reverse impulse to break static friction
                        if self.impulse_remaining <= 0:
                            self.impulse_remaining = self.IMPULSE_STEPS
                            rospy.loginfo(
                                f"[ros_bridge] IMPULSE REV: vel={velocity:.3f} "
                                f"v_est={v_est:.3f} → burst {-self.IMPULSE_TORQUE:.2f} "
                                f"for {self.IMPULSE_STEPS} steps (PID disabled)"
                            )
                        ff_torque = -self.IMPULSE_TORQUE
                        self.impulse_remaining -= 1
                    else:
                        # Moving in reverse: proportional torque to maintain speed
                        self.impulse_remaining = 0
                        ff_torque = -(MIN_TORQUE / 0.25) * abs_vel
                        ff_torque = min(ff_torque, -0.06)

                # Clamp torque to safe range
                ff_torque = max(-0.15, min(0.15, ff_torque))

                msg = car_input()
                msg.header.stamp = rospy.Time.now()
                msg.velocity = cmd_velocity
                msg.torque = ff_torque
                msg.steer = steer
                msg.steer_override = False
                self.cmd_pub.publish(msg)

                self.cmd_count += 1
                if self.cmd_count % 50 == 1:
                    imp_str = f" [IMPULSE {self.impulse_remaining}]" if self.impulse_remaining > 0 else ""
                    va_str = f" v_est={v_est:.3f}" if v_actual is not None else f" v_crs={self.last_vx_b:.3f}"
                    rospy.loginfo(
                        f"[ros_bridge] cmd #{self.cmd_count}: "
                        f"vel={velocity:.3f} torque={ff_torque:.4f} "
                        f"steer={math.degrees(steer):.1f}deg{va_str}{imp_str}"
                    )

            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    rospy.logwarn(f"[ros_bridge] cmd recv error: {e}")

    def spin(self):
        """Block until ROS shutdown."""
        rospy.spin()
        self.running = False
        self.state_sock.close()
        self.cmd_sock.close()


def main():
    parser = argparse.ArgumentParser(description="ROS <-> UDP Bridge")
    parser.add_argument(
        "--namespace", default="BEN_CAR_WIFI",
        help="ROS namespace for car topics",
    )
    args = parser.parse_args()

    bridge = ROSUDPBridge(namespace=args.namespace)

    def shutdown_handler(sig, frame):
        print("\n[ros_bridge] Shutting down...")
        bridge.running = False
        rospy.signal_shutdown("User interrupt")

    signal.signal(signal.SIGINT, shutdown_handler)
    bridge.spin()


if __name__ == "__main__":
    main()
