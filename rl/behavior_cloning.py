# rl/behavior_cloning.py

import os
import pickle
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl.networks import MLP


def load_expert_trajectories(folder):
    """Load all (state, action) pairs from expert .pkl trajectories."""
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.pkl')]
    data = []
    for f in sorted(files):
        with open(f, "rb") as fh:
            traj = pickle.load(fh)  # list of (state, action)
            data.extend(traj)
    return data


def build_dataset(expert_folder):
    """Return arrays X (states) and Y (actions) from expert folder."""
    data = load_expert_trajectories(expert_folder)
    if len(data) == 0:
        raise RuntimeError(f"No expert data found in {expert_folder}")

    states = []
    actions = []
    for s, a in data:
        s = np.asarray(s, dtype=np.float32)
        a = np.asarray(a, dtype=np.float32)
        # here we assume s is [x,y,yaw,v] and a is [steer, accel]
        states.append(s)
        actions.append(a)

    X = np.stack(states, axis=0)
    Y = np.stack(actions, axis=0)
    return X, Y


def train_bc(
    expert_folder: str,
    policy_save_path: str,
    hidden=(128, 128),
    batch_size: int = 256,
    epochs: int = 50,
    lr: float = 1e-3,
    device: str = None,
):
    # device selection
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # build dataset
    X, Y = build_dataset(expert_folder)
    print(f"[BC] Loaded {X.shape[0]} expert samples from {expert_folder}")

    X_t = torch.from_numpy(X).to(device)  # [N, state_dim]
    Y_t = torch.from_numpy(Y).to(device)  # [N, act_dim]

    state_dim = X.shape[1]
    act_dim = Y.shape[1]

    # build policy network
    policy = MLP(input_dim=state_dim, output_dim=act_dim, hidden=hidden).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # simple manual minibatch loop
    N = X_t.shape[0]
    indices = np.arange(N)

    for ep in range(epochs):
        np.random.shuffle(indices)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, N, batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]

            xb = X_t[batch_idx]
            yb = Y_t[batch_idx]

            optimizer.zero_grad()
            pred = policy(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        epoch_loss /= max(n_batches, 1)
        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            print(f"[BC] Epoch {ep + 1}/{epochs}, loss={epoch_loss:.6f}")

    # save trained policy
    os.makedirs(os.path.dirname(policy_save_path), exist_ok=True)
    torch.save({"state_dict": policy.state_dict(),
                "state_dim": state_dim,
                "act_dim": act_dim,
                "hidden": list(hidden)},
               policy_save_path)
    print(f"[BC] Saved policy to {policy_save_path}")


def main():
    parser = argparse.ArgumentParser(description="Behavior Cloning training script.")
    parser.add_argument(
        "--expert-dir",
        type=str,
        default="data/expert_success",
        help="Directory with filtered successful expert episodes (.pkl).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/bc_policies/bc_policy.pt",
        help="Path to save the trained BC policy.",
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Number of training epochs."
    )
    parser.add_argument(
        "--batch-size", type=int, default=256,
        help="Mini-batch size."
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate."
    )
    args = parser.parse_args()

    train_bc(
        expert_folder=args.expert_dir,
        policy_save_path=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()

