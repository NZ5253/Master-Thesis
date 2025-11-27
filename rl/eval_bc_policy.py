# rl/eval_bc_policy.py

import argparse
import numpy as np
import torch
import yaml

from env.parking_env import ParkingEnv
from rl.networks import MLP


def load_policy(path: str, device: str):
    ckpt = torch.load(path, map_location=device)
    input_dim = ckpt["input_dim"]
    act_dim = ckpt["act_dim"]
    hidden = tuple(ckpt.get("hidden", [128, 128]))

    policy = MLP(input_dim=input_dim, output_dim=act_dim, hidden=hidden).to(device)
    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()
    return policy, input_dim, act_dim


def run_episode(env: ParkingEnv, policy: MLP, device: str, max_steps: int = None):
    obs = env.reset(randomize=True)
    done = False
    steps = 0
    info = {}

    if max_steps is None:
        max_steps = env.max_steps

    while not done and steps < max_steps:
        o_np = np.asarray(obs, dtype=np.float32)
        with torch.no_grad():
            o_t = torch.from_numpy(o_np).unsqueeze(0).to(device)
            a_t = policy(o_t)
        action = a_t.cpu().numpy().squeeze()

        obs, reward, done, info = env.step(action)
        steps += 1

        if done:
            break

    term = info.get("termination", None)
    success = (term == "success")
    return success, steps, term


def main():
    parser = argparse.ArgumentParser(description="Evaluate BC policy in random parking env.")
    parser.add_argument("--policy", type=str, default="data/bc_policies/bc_policy.pt")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--cfg-env", type=str, default="config_env.yaml")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[EVAL] Using device: {device}")

    with open(args.cfg-env, "r") as f:
        cfg_env = yaml.safe_load(f)
    env = ParkingEnv(cfg_env)

    policy, input_dim, act_dim = load_policy(args.policy, device)
    print(f"[EVAL] Loaded BC policy with input_dim={input_dim}, act_dim={act_dim}")

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
