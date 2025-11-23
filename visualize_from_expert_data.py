import pickle
import yaml
import matplotlib.pyplot as plt

from env.parking_env import ParkingEnv
from visualize_mpc_episode import draw_car, draw_obstacles, draw_goal


def visualize_episode(pkl_path: str, cfg_env: dict):
    env = ParkingEnv(cfg_env)
    # initialise obstacles; randomize=False so they are at the base layout
    env.reset(randomize=False)

    with open(pkl_path, "rb") as f:
        traj = pickle.load(f)

    fig, ax = plt.subplots()
    draw_obstacles(ax, env)
    draw_goal(ax, env)

    for state, action in traj:
        draw_car(ax, state)

    ax.set_aspect("equal")
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 2.2)
    ax.set_title(pkl_path)
    plt.show()


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episode",
        type=int,
        default=0,
        help="Episode index (episode_XXXX.pkl)",
    )
    args = parser.parse_args()

    with open("config_env.yaml") as f:
        cfg_env = yaml.safe_load(f)

    pkl_path = os.path.join(
        "data",
        "expert_trajectories",
        f"episode_{args.episode:04d}.pkl",
    )

    visualize_episode(pkl_path, cfg_env)
