# simulation/scenario_loader.py
from dataclasses import dataclass
from typing import List

import yaml
import os

from mpc.teb_mpc import VehicleState, ParkingGoal, Obstacle
from env.obstacle_manager import ObstacleManager


@dataclass
class ParkingScenario:
    """A simple scenario wrapper that gives TEB:
       - initial state
       - goal pose
       - static obstacle list (converted to MPC Obstacle format)
    """
    init_state: VehicleState
    goal: ParkingGoal
    obstacles: List[Obstacle]


def _load_env_cfg() -> dict:
    """Load the single perpendicular config_env.yaml."""
    cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config_env.yaml")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def _convert_to_mpc_obstacles(env_obstacles) -> List[Obstacle]:
    """Convert rectangular obstacles (dicts) into MPC Obstacle objects with half sizes."""
    obs_list = []
    for o in env_obstacles:
        obs_list.append(
            Obstacle(
                cx=float(o["x"]),
                cy=float(o["y"]),
                hx=float(o["w"]) / 2.0,
                hy=float(o["h"]) / 2.0,
            )
        )
    return obs_list


def make_perpendicular_scenario() -> ParkingScenario:
    """Build the perpendicular scenario:
       - loads config_env.yaml
       - automatically constructs the neighbour obstacles using ObstacleManager
       - defines a reasonable initial pose
    """

    cfg = _load_env_cfg()
    gx, gy, gyaw = cfg["goal"]

    # Initial pose in the lane below the bay
    init = VehicleState(
        x=gx,             # aligned horizontally with bay
        y=gy - 1.0,       # 1 meter below bay
        yaw=gyaw,         # facing the slot (pi/2)
        v=0.0,
    )

    # Let ObstacleManager generate left/right neighbours + walls
    om = ObstacleManager(cfg["obstacles"], goal=cfg["goal"])
    om.reset_to_base()

    # Convert to MPC Obstacle format
    obstacles = _convert_to_mpc_obstacles(om.obstacles)

    goal = ParkingGoal(x=gx, y=gy, yaw=gyaw)

    return ParkingScenario(init, goal, obstacles)
