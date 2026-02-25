#!/usr/bin/env python3
"""
Steering Calibration Test for ChronosCar.

Commands a constant steer angle + constant forward velocity and measures
the actual turning radius from MoCap position data. Compares to the
kinematic bicycle model prediction to find the steering gain factor.

If the real car turns faster than the model, the effective steering angle
is larger than commanded -- this explains yaw rate mismatches during parking.

Usage (car must be on flat open area, Docker + ros_bridge running):
  python deploy/steer_calibration.py

The test runs for ~8 seconds, then prints results.
Press Ctrl+C to abort at any time (sends stop command).
"""

import sys
import os
import time
import struct
import socket
import signal
import numpy as np

# UDP ports (must match ros_bridge.py)
STATE_UDP_PORT = 5800
CMD_UDP_PORT = 5801

# Test parameters
TEST_VELOCITY = 0.15       # m/s forward (slow, safe)
TEST_STEER_RAD = 0.20      # rad (~11.5 deg) -- moderate turn
TEST_DURATION = 8.0         # seconds
WHEELBASE = 0.09            # m (from config)
SETTLE_TIME = 1.0           # seconds to reach steady state before measuring


def main():
    print("=" * 60)
    print("  ChronosCar Steering Calibration Test")
    print("=" * 60)
    print()
    print(f"  Commanded steer:    {TEST_STEER_RAD:.3f} rad ({np.degrees(TEST_STEER_RAD):.1f} deg)")
    print(f"  Commanded velocity: {TEST_VELOCITY:.3f} m/s")
    print(f"  Duration:           {TEST_DURATION:.0f} s (first {SETTLE_TIME:.0f}s ignored)")
    print(f"  Wheelbase:          {WHEELBASE:.3f} m")
    print()
    print(f"  Expected turning radius: R = {WHEELBASE}/tan({TEST_STEER_RAD:.3f})")
    R_expected = WHEELBASE / np.tan(TEST_STEER_RAD)
    print(f"                         R = {R_expected:.4f} m")
    print(f"  Expected yaw rate at v={TEST_VELOCITY}: "
          f"{np.degrees(TEST_VELOCITY / R_expected):.2f} deg/s")
    print()
    print("  Place the car on a flat, open area with ~1m clearance.")
    print("  The car will drive forward in a circle.")
    print()

    # Check for port conflicts FIRST
    print("Checking for other processes on UDP port 5800...")
    import subprocess
    try:
        result = subprocess.run(
            ["lsof", "-i", f"UDP:{STATE_UDP_PORT}", "-t"],
            capture_output=True, text=True, timeout=3
        )
        pids = result.stdout.strip().split()
        my_pid = str(os.getpid())
        other_pids = [p for p in pids if p and p != my_pid]
        if other_pids:
            print(f"\n  WARNING: Other processes using port {STATE_UDP_PORT}: PIDs {other_pids}")
            print(f"  These will steal UDP packets! Kill them first:")
            for pid in other_pids:
                try:
                    cmd_result = subprocess.run(
                        ["ps", "-p", pid, "-o", "comm="],
                        capture_output=True, text=True, timeout=2
                    )
                    name = cmd_result.stdout.strip()
                    print(f"    kill {pid}  # {name}")
                except Exception:
                    print(f"    kill {pid}")
            print()
            print("  Run those kill commands, then re-run this script.")
            return
    except FileNotFoundError:
        pass  # lsof not available, skip check
    except Exception:
        pass

    # Setup UDP
    state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    state_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    state_sock.bind(("0.0.0.0", STATE_UDP_PORT))
    state_sock.settimeout(0.5)

    cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    cmd_dest = ("127.0.0.1", CMD_UDP_PORT)

    def send_cmd(vel, steer):
        data = struct.pack("!3d", vel, steer, 0.0)
        try:
            cmd_sock.sendto(data, cmd_dest)
        except OSError:
            pass

    def stop():
        send_cmd(0.0, 0.0)
        time.sleep(0.05)
        send_cmd(0.0, 0.0)

    def drain_and_get_latest():
        """Read ALL buffered packets, return only the newest one.

        ros_bridge sends state at ~100Hz. Our main loop runs at 20Hz.
        Without draining, recvfrom returns the OLDEST buffered packet,
        so we'd always be reading stale data from seconds ago.
        """
        latest = None
        state_sock.setblocking(False)
        while True:
            try:
                data, _ = state_sock.recvfrom(1024)
                if len(data) == 56:
                    latest = data
            except BlockingIOError:
                break
            except OSError:
                break
        state_sock.setblocking(True)
        state_sock.settimeout(0.5)
        return latest

    # Signal handler for clean stop
    running = [True]
    def handler(sig, frame):
        print("\n[ABORT] Stopping car...")
        running[0] = False
    signal.signal(signal.SIGINT, handler)

    # Wait for first state
    print("Waiting for car state from UDP...")
    while running[0]:
        try:
            data, _ = state_sock.recvfrom(1024)
            if len(data) == 56:
                break
        except socket.timeout:
            continue
    if not running[0]:
        stop()
        return

    # Drain any old buffered packets before starting
    print("State received. Draining buffer...")
    drain_and_get_latest()

    print("Starting test in 2 seconds...")
    time.sleep(2.0)
    drain_and_get_latest()  # Drain again after sleep

    if not running[0]:
        stop()
        return

    # Collect data
    positions = []  # (time, x, y, yaw)
    t_start = time.time()

    print(f"\nDriving: steer={np.degrees(TEST_STEER_RAD):.1f} deg, vel={TEST_VELOCITY:.2f} m/s")
    print("  time   |   x      y      yaw")
    print("  -------|---------------------")

    while running[0]:
        t_now = time.time()
        elapsed = t_now - t_start

        if elapsed > TEST_DURATION:
            break

        # Send constant command
        send_cmd(TEST_VELOCITY, TEST_STEER_RAD)

        # Read LATEST state (drain buffer to avoid reading stale data)
        data = drain_and_get_latest()
        if data is not None and len(data) == 56:
            vals = struct.unpack("!7d", data)
            x, y, yaw = vals[0], vals[1], vals[2]
            positions.append((elapsed, x, y, yaw))

            if len(positions) % 10 == 0:
                print(f"  {elapsed:5.1f}s | {x:+.4f} {y:+.4f} {np.degrees(yaw):+7.1f} deg")

        time.sleep(0.05)  # 20 Hz command rate

    # Stop the car
    print("\nStopping car...")
    stop()

    # Close sockets
    state_sock.close()
    cmd_sock.close()

    if len(positions) < 20:
        print("ERROR: Not enough data points collected. Check UDP connection.")
        return

    # --- Analyze ---
    print("\n" + "=" * 60)
    print("  ANALYSIS")
    print("=" * 60)

    # Convert to arrays
    ts = np.array([p[0] for p in positions])
    xs = np.array([p[1] for p in positions])
    ys = np.array([p[2] for p in positions])
    yaws = np.array([p[3] for p in positions])

    # Only use data after settle time
    mask = ts >= SETTLE_TIME
    if np.sum(mask) < 10:
        print("ERROR: Not enough data after settle time.")
        return

    ts_m = ts[mask]
    xs_m = xs[mask]
    ys_m = ys[mask]
    yaws_m = yaws[mask]

    # Unwrap yaw for continuous measurement
    yaws_m = np.unwrap(yaws_m)

    # Method 1: Yaw rate
    # Linear fit yaw vs time to get yaw_rate (rad/s)
    coeffs = np.polyfit(ts_m, yaws_m, 1)
    yaw_rate_measured = coeffs[0]  # rad/s

    # Expected yaw rate from kinematic model
    yaw_rate_expected = TEST_VELOCITY / R_expected  # = (v/L) * tan(steer)

    gain_from_yaw = yaw_rate_measured / yaw_rate_expected

    print(f"\n  Method 1: Yaw Rate")
    print(f"    Measured yaw rate:  {np.degrees(yaw_rate_measured):+.2f} deg/s "
          f"({yaw_rate_measured:+.4f} rad/s)")
    print(f"    Expected yaw rate:  {np.degrees(yaw_rate_expected):+.2f} deg/s "
          f"({yaw_rate_expected:+.4f} rad/s)")
    print(f"    Steering gain:      {gain_from_yaw:.3f}x")

    # Method 2: Turning radius from position
    # Fit a circle to the (x, y) positions
    # Using algebraic circle fit: (x-a)^2 + (y-b)^2 = R^2
    # Rewrite as: x^2 + y^2 = 2*a*x + 2*b*y + (R^2 - a^2 - b^2)
    # Linear system: [2x, 2y, 1] @ [a, b, c] = x^2 + y^2 where c = R^2 - a^2 - b^2
    A = np.column_stack([2 * xs_m, 2 * ys_m, np.ones(len(xs_m))])
    b_vec = xs_m**2 + ys_m**2
    result, residuals, rank, sv = np.linalg.lstsq(A, b_vec, rcond=None)
    a_circ, b_circ, c_circ = result
    R_measured = np.sqrt(c_circ + a_circ**2 + b_circ**2)

    gain_from_radius = R_expected / R_measured  # Smaller radius = larger effective steer

    print(f"\n  Method 2: Turning Radius (circle fit)")
    print(f"    Measured radius:    {R_measured:.4f} m")
    print(f"    Expected radius:    {R_expected:.4f} m")
    print(f"    Steering gain:      {gain_from_radius:.3f}x")
    print(f"    Circle center:      ({a_circ:.4f}, {b_circ:.4f})")

    # Method 3: Velocity from position deltas
    dt_arr = np.diff(ts_m)
    dx_arr = np.diff(xs_m)
    dy_arr = np.diff(ys_m)
    v_arr = np.sqrt(dx_arr**2 + dy_arr**2) / dt_arr
    v_avg = np.mean(v_arr)

    print(f"\n  Method 3: Actual Velocity")
    print(f"    Commanded velocity: {TEST_VELOCITY:.3f} m/s")
    print(f"    Measured velocity:  {v_avg:.3f} m/s (avg)")
    print(f"    Velocity ratio:     {v_avg / TEST_VELOCITY:.3f}x")

    # Summary
    avg_gain = (gain_from_yaw + gain_from_radius) / 2.0
    print(f"\n  " + "-" * 40)
    print(f"  AVERAGE STEERING GAIN: {avg_gain:.3f}x")
    print(f"  " + "-" * 40)

    if abs(avg_gain - 1.0) < 0.15:
        print(f"\n  Steering gain is close to 1.0 -- model matches hardware.")
        print(f"  The yaw drift during parking is likely from other causes")
        print(f"  (tire slip, velocity dynamics, friction timing).")
    elif avg_gain > 1.15:
        steer_correction = 1.0 / avg_gain
        print(f"\n  Physical steer is {avg_gain:.2f}x commanded steer!")
        print(f"  To fix: multiply steer commands by {steer_correction:.3f}")
        print(f"  OR retrain with effective max_steer = {0.35 * avg_gain:.3f} rad")
        print(f"     (in config_env_chronos.yaml vehicle.max_steer)")
    elif avg_gain < 0.85:
        print(f"\n  Physical steer is {avg_gain:.2f}x commanded steer (under-steering).")
        print(f"  The car turns LESS than the model predicts.")

    # Total yaw change during measurement
    total_yaw = yaws_m[-1] - yaws_m[0]
    total_time = ts_m[-1] - ts_m[0]
    print(f"\n  Total yaw change: {np.degrees(total_yaw):.1f} deg in {total_time:.1f}s")
    print(f"  Data points used: {np.sum(mask)} (of {len(positions)} total)")
    print()


if __name__ == "__main__":
    main()
