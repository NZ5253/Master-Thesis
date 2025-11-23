# simulation/simulate_teb_mpc.py

import argparse
import yaml
import numpy as np

from mpc.teb_mpc import TEBMPC, VehicleState, ParkingGoal, Obstacle
from simulation.scenario_loader import make_perpendicular_scenario
from env.parking_env import ParkingEnv


def main():

    # -----------------------------
    # load perpendicular env config
    # -----------------------------
    with open("config_env.yaml", "r") as f:
        cfg_env = yaml.safe_load(f)

    # build environment
    env = ParkingEnv(cfg_env)

    # load scenario (initial pose + static obstacles)
    scenario = make_perpendicular_scenario()

    # build TEB-MPC solver
    teb = TEBMPC()

    # initial state
    state = scenario.init_state
    step_idx = 0
    done = False

    print("Starting simulation...")
    print("Initial state:", state)

    while not done and step_idx < cfg_env["max_steps"]:

        # solve TEB for this state
        sol = teb.solve(
            state=state,
            goal=scenario.goal,
            obstacles=scenario.obstacles,
            profile="perpendicular",
        )

        # TEB returns controls as [steer, accel]
        steer = float(sol.controls[0, 0])
        accel = float(sol.controls[0, 1])

        # apply to environment (env.step expects [steer, accel])
        obs, reward, done, info = env.step(np.array([steer, accel], dtype=float))

        # update VehicleState from observation
        x, y, yaw, v = obs[:4]
        state = VehicleState(x=x, y=y, yaw=yaw, v=v)

        step_idx += 1

        if "termination" in info:
            print(f"Terminated: {info['termination']} at step {step_idx}")

    print("Simulation finished in", step_idx, "steps.")


if __name__ == "__main__":
    main()
