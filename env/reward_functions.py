import numpy as np
from .geometry_utils import pose_error_goal_frame  # adjust import if needed


def compute_reward(state, next_state, goal, collision):
    # unpack
    x, y, yaw, v = next_state
    gx, gy, gyaw = goal

    # --- 1) Distance & progress towards goal ---
    pos_err_prev = np.hypot(gx - state[0], gy - state[1])
    pos_err      = np.hypot(gx - x,       gy - y)

    # positive if we moved closer
    progress = pos_err_prev - pos_err

    # --- 2) Geometry in goal frame (slot-aligned) ---
    ex, ey, yaw_err = pose_error_goal_frame(next_state, goal)
    # ex: forward/back error along slot
    # ey: lateral offset from slot centre
    # yaw_err: heading misalignment

    # --- 3) Base shaping reward ---
    # progress term: encourages continuous movement toward goal
    reward = 0.5 * progress          # >0 if getting closer

    # lateral alignment: strong penalty for being off the centerline
    reward -= 0.2 * abs(ey)

    # heading alignment: smaller penalty than lateral
    reward -= 0.05 * abs(yaw_err)

    # small time penalty so it prefers faster parking
    reward -= 0.01

    info = {
        "pos_err": float(pos_err),
        "progress": float(progress),
        "lat_err": float(ey),
        "lon_err": float(ex),
        "yaw_err": float(yaw_err),
    }

    # --- 4) Collision override ---
    if collision:
        reward -= 5.0   # increase magnitude if you want stronger penalty
        info["collision"] = True
        info["success"] = False
        return float(reward), info

    info["collision"] = False

    # --- 5) Anti-stall term (avoid plateau far from goal) ---
    # If still far (>0.5 m) and basically not moving, give a tiny penalty.
    if pos_err > 0.5 and abs(v) < 0.02:
        reward -= 0.02
        info["stall_penalty"] = True
    else:
        info["stall_penalty"] = False

    # --- 6) Success bonus (same thresholds as env) ---
    if pos_err < 0.08 and abs(yaw_err) < 0.1:
        reward += 10.0
        info["success"] = True
    else:
        info["success"] = False

    return float(reward), info
