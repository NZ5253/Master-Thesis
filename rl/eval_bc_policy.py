# rl/eval_bc_policy.py

import argparse
import numpy as np
import torch
import yaml

from env.parking_env import ParkingEnv
from rl.networks import MLP


def load_policy(path: str, device: str):
    ckpt = torch.load(path, map_location=device)
    state_dim = ckpt["state_dim"]
    act_dim = ckpt["act_dim"]
    hidden = tuple(ckpt.get("hidden", [128, 128]))

    policy = MLP(input_dim=state_dim, output_dim=act_dim, hidden=hidden).to(device)
    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()
    return policy, state_dim, act_dim


def run_episode(env: ParkingEnv, policy: MLP, device: str, max_steps: int = None):
    """Run one episode using the BC policy. Returns (success, steps, termination)."""
    obs = env.reset(randomize=True)
    done = False
    steps = 0
    info = {}

    if max_steps is None:
        max_steps = env.max_steps

    while not done and steps < max_steps:
        # We trained on raw state [x, y, yaw, v], not full obs.
        s = env.state.astype(np.float32)  # shape (4,)

        with torch.no_grad():
            s_t = torch.from_numpy(s).unsqueeze(0).to(device)  # [1, state_dim]
            a_t = policy(s_t)                                  # [1, act_dim]
        action = a_t.cpu().numpy().squeeze()                   # [2,] -> [steer, accel]

        obs, reward, done, info = env.step(action)
        steps += 1

        if done:
            break

    term = info.get("termination", None)
    success = (term == "success")
    return success, steps, term


def main():
    parser = argparse.ArgumentParser(description="Evaluate a behavior-cloned policy in the parking env.")
    parser.add_argument(
        "--policy",
        type=str,
        default="data/bc_policies/bc_policy.pt",
        help="Path to the trained BC policy (.pt).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=50,
        help="Number of evaluation episodes.",
    )
    parser.add_argument(
        "--cfg-env",
        type=str,
        default="config_env.yaml",
        help="Path to env config.",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[EVAL] Using device: {device}")

    # load env
    with open(args.cfg_env, "r") as f:
        cfg_env = yaml.safe_load(f)
    env = ParkingEnv(cfg_env)

    # load policy
    policy, state_dim, act_dim = load_policy(args.policy, device)
    print(f"[EVAL] Loaded BC policy with state_dim={state_dim}, act_dim={act_dim}")

    n_success = 0
    term_counts = {}
    steps_list = []

    for ep in range(args.episodes):
        success, steps, term = run_episode(env, policy, device)
        steps_list.append(steps)
        term_counts[term] = term_counts.get(term, 0) + 1
        if success:
            n_success += 1
        print(f"[EVAL] Episode {ep}: termination={term}, steps={steps}")

    success_rate = n_success / args.episodes
    avg_steps = float(np.mean(steps_list)) if steps_list else 0.0

    print("-------------------------------------------------")
    print(f"[EVAL] Episodes:      {args.episodes}")
    print(f"[EVAL] Success rate:  {success_rate:.3f}")
    print(f"[EVAL] Avg steps:     {avg_steps:.1f}")
    print(f"[EVAL] Terminations:  {term_counts}")
    print("-------------------------------------------------")


if __name__ == "__main__":
    main()
