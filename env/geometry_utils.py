import math

def wrap_angle(angle):
    """Wrap angle to [-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi

def pose_error_goal_frame(state, goal):
    """
    Express pose error in the goal's coordinate frame.

    state: [x, y, yaw, v]
    goal:  [gx, gy, gyaw]

    Returns:
        ex: longitudinal error (forward/back along goal heading)
        ey: lateral error (left/right relative to goal)
        yaw_err: heading error (state.yaw - goal.yaw, wrapped)
    """
    x, y, yaw, _ = state
    gx, gy, gyaw = goal

    dx = x - gx
    dy = y - gy

    cos_g = math.cos(gyaw)
    sin_g = math.sin(gyaw)

    # rotate world error into goal frame
    ex =  cos_g * dx + sin_g * dy      # along goal x-axis
    ey = -sin_g * dx + cos_g * dy      # left/right from slot centerline

    yaw_err = wrap_angle(yaw - gyaw)
    return ex, ey, yaw_err


def vehicle_footprint_collision(state, vehicle_model, obstacle):
    # placeholder: implement rectangle vs rectangle or polygon collision later
    return False
