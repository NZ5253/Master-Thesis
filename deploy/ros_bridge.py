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

        # ---------------------------------------------------------------
        # Velocity P-controller (Professor Frank's recommendation).
        #
        # OLD approach (jerky):
        #   standstill → impulse burst (0.20 N·m) → drop to MIN_TORQUE (0.10)
        #   → car slows below threshold → impulse again → repeat cycle → JERKY
        #
        # NEW approach (smooth):
        #   torque = FRICTION_COMP + K_P * (v_target_hw - v_actual)
        #   At standstill: large vel_error → high torque (natural breakaway)
        #   Car accelerates: vel_error shrinks → torque decreases smoothly
        #   At target speed: vel_error ≈ 0 → FRICTION_COMP sustains motion
        #   No binary impulse/stop cycles. Mimics training dynamics.
        #
        # VEL_GAIN: car drives 1.503x commanded (measured by steer_calibration.py).
        #   ActionConverter divides velocity by this gain before sending.
        #   We undo it here to compare velocity target with actual hardware speed.
        #   MUST match calibration.velocity_gain in parking_scene.yaml.
        #   For new car: update to match its measured gain (default 1.0 until calibrated).
        # ---------------------------------------------------------------
        self.VEL_GAIN     = 1.503   # ChronosCar: measured 1.503x. New car: update after calibration.
        self.FRICTION_COMP = 0.08   # Always-on torque to overcome kinetic friction (N·m)
        self.K_P_VEL      = 0.45   # P gain: torque per (m/s) velocity error

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
                # All motion uses velocity=nan (PID disabled) + direct torque.
                # A P velocity controller replaces the old impulse/MIN_TORQUE cycle:
                #   torque = FRICTION_COMP + K_P * (v_target_hw - v_actual)
                # This provides smooth, continuous speed regulation that naturally
                # gives high torque at standstill and tapers off as speed builds.
                BRAKE_TORQUE = 0.10  # Active braking force (for overspeed)

                # Actual velocity: prefer v_actual from host (MoCap/encoder),
                # fall back to CRS estimate (unreliable for ChronosCar at idle)
                v_est = v_actual if v_actual is not None else self.last_vx_b

                if abs(velocity) < 0.01:
                    # ---- STOP ----
                    cmd_velocity = 0.0   # PID at 0 helps hold car still
                    if v_est > 0.03:
                        ff_torque = -BRAKE_TORQUE   # Moving forward → brake
                    elif v_est < -0.03:
                        ff_torque = BRAKE_TORQUE    # Moving reverse → brake
                    else:
                        ff_torque = 0.0

                elif velocity > 0:
                    # ---- FORWARD: P velocity controller ----
                    # velocity is gain-corrected (divided by VEL_GAIN in ActionConverter).
                    # Undo the correction to get hardware-space target for comparison
                    # with v_est (which is real hardware speed from MoCap/encoder).
                    #
                    # vel_error > 0: car slower than target → add torque
                    # vel_error = 0: car at target → FRICTION_COMP sustains motion
                    # vel_error < 0: car faster than target → coast (low torque)
                    v_target_hw = velocity * self.VEL_GAIN
                    vel_error   = v_target_hw - v_est

                    cmd_velocity = math.nan   # Bypass PID entirely
                    ff_torque = self.FRICTION_COMP + self.K_P_VEL * vel_error
                    if vel_error < -0.10:
                        # Car going >10 cm/s faster than target (overshoot).
                        # Apply active braking — without this the car coasts 2-3×
                        # the commanded speed (observed: target=0.11, v_hw=0.278).
                        ff_torque = -BRAKE_TORQUE
                    else:
                        ff_torque = max(0.0, min(0.15, ff_torque))  # forward: non-negative

                    if self.cmd_count % 50 == 1 and abs(vel_error) > 0.05:
                        rospy.loginfo(
                            f"[ros_bridge] FWD P-ctrl: v_tgt={v_target_hw:.3f} "
                            f"v_est={v_est:.3f} err={vel_error:+.3f} "
                            f"torque={ff_torque:.3f}"
                        )

                else:
                    # ---- REVERSE: P velocity controller ----
                    # Use absolute speeds; braking guard prevents overspeed.
                    # NOTE: no 0.25 m/s cap — training uses max_vel=0.5 m/s and the
                    # policy sometimes needs the car at 0.35-0.45 m/s in reverse.
                    # The old 0.25 cap caused premature deceleration: at v_hw=0.255
                    # speed_error went negative → only friction comp (-0.08) → car
                    # coasted down to ~0.13 m/s even during commanded reverse.
                    abs_target = abs(velocity) * self.VEL_GAIN
                    abs_actual = abs(v_est)
                    speed_error = abs_target - abs_actual  # +ve = need more reverse

                    cmd_velocity = math.nan   # Bypass PID (forward-only anyway)

                    if v_est < velocity - 0.15:
                        # Car going MUCH FASTER in reverse than requested → BRAKE.
                        # Threshold is 0.15 m/s (not 0.02): policy makes small accel
                        # adjustments (e.g. v_model -0.37 → -0.30) that should NOT
                        # trigger active braking. Only brake for real overspeed.
                        ff_torque = BRAKE_TORQUE
                    else:
                        ff_torque = -(self.FRICTION_COMP + self.K_P_VEL * max(0.0, speed_error))
                        ff_torque = max(-0.15, min(0.0, ff_torque))  # reverse: non-positive

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
                    va_str = f" v_est={v_est:.3f}" if v_actual is not None else f" v_crs={self.last_vx_b:.3f}"
                    rospy.loginfo(
                        f"[ros_bridge] cmd #{self.cmd_count}: "
                        f"vel={velocity:.3f} torque={ff_torque:.4f} "
                        f"steer={math.degrees(steer):.1f}deg{va_str}"
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
