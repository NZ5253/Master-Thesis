"""
Policy network architecture for parallel parking RL agent.

Network design:
- Input: 7D observation (along, lateral, yaw_err, v, dist_front, dist_left, dist_right)
- Output: 2D continuous action (steer, accel)
- Architecture: MLP with layer normalization and tanh activations
"""

import torch
import torch.nn as nn
import numpy as np


class ParkingPolicyNetwork(nn.Module):
    """
    Actor-Critic policy network for parking task.

    Uses separate heads for policy (actor) and value (critic).
    """

    def __init__(
        self,
        obs_dim=7,
        action_dim=2,
        hidden_sizes=[256, 256],
        activation="tanh",
        use_layer_norm=True,
    ):
        """
        Args:
            obs_dim: Observation dimension (default 10)
            action_dim: Action dimension (default 2: steer, accel)
            hidden_sizes: List of hidden layer sizes
            activation: Activation function ('tanh', 'relu', 'elu')
            use_layer_norm: Whether to use layer normalization
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Activation function
        if activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "elu":
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # ========== Shared Feature Extractor ==========
        layers = []
        prev_size = obs_dim

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_size))
            layers.append(self.activation)
            prev_size = hidden_size

        self.shared_net = nn.Sequential(*layers)

        # ========== Policy Head (Actor) ==========
        # Outputs mean and log_std for Gaussian policy
        self.policy_mean = nn.Linear(prev_size, action_dim)
        self.policy_log_std = nn.Parameter(
            torch.zeros(action_dim)
        )  # Learnable log std

        # Initialize policy mean with small weights for stability
        nn.init.orthogonal_(self.policy_mean.weight, gain=0.01)
        nn.init.constant_(self.policy_mean.bias, 0.0)

        # ========== Value Head (Critic) ==========
        self.value_head = nn.Linear(prev_size, 1)

        # Initialize value head
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.constant_(self.value_head.bias, 0.0)

    def forward(self, obs):
        """
        Forward pass through network.

        Args:
            obs: Observation tensor [batch, obs_dim]

        Returns:
            policy_mean: Mean of action distribution [batch, action_dim]
            policy_std: Std of action distribution [batch, action_dim]
            value: State value estimate [batch, 1]
        """
        # Shared features
        features = self.shared_net(obs)

        # Policy (actor)
        policy_mean = self.policy_mean(features)
        policy_std = torch.exp(self.policy_log_std).expand_as(policy_mean)

        # Value (critic)
        value = self.value_head(features)

        return policy_mean, policy_std, value

    def get_action(self, obs, deterministic=False):
        """
        Sample action from policy.

        Args:
            obs: Observation tensor [batch, obs_dim] or [obs_dim]
            deterministic: If True, return mean action (no sampling)

        Returns:
            action: Sampled action [batch, action_dim] or [action_dim]
            log_prob: Log probability of action [batch] or scalar
            value: Value estimate [batch] or scalar
        """
        # Handle single observation (add batch dimension)
        single_obs = obs.dim() == 1
        if single_obs:
            obs = obs.unsqueeze(0)

        policy_mean, policy_std, value = self.forward(obs)

        if deterministic:
            action = policy_mean
            log_prob = None
        else:
            # Sample from Gaussian distribution
            dist = torch.distributions.Normal(policy_mean, policy_std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)

        # Remove batch dimension if input was single observation
        if single_obs:
            action = action.squeeze(0)
            if log_prob is not None:
                log_prob = log_prob.squeeze(0)
            value = value.squeeze(0)

        return action, log_prob, value

    def evaluate_actions(self, obs, actions):
        """
        Evaluate log probability and value for given obs-action pairs.

        Used during PPO updates.

        Args:
            obs: Observation tensor [batch, obs_dim]
            actions: Action tensor [batch, action_dim]

        Returns:
            log_probs: Log probability of actions [batch]
            values: Value estimates [batch]
            entropy: Entropy of policy [batch]
        """
        policy_mean, policy_std, value = self.forward(obs)

        # Compute log probabilities
        dist = torch.distributions.Normal(policy_mean, policy_std)
        log_probs = dist.log_prob(actions).sum(dim=-1)

        # Compute entropy (for entropy bonus)
        entropy = dist.entropy().sum(dim=-1)

        return log_probs, value.squeeze(-1), entropy


class MLPNetwork(nn.Module):
    """
    Simple MLP for behavior cloning pre-training.

    This is a simpler version without the actor-critic split,
    used for supervised learning from expert demonstrations.
    """

    def __init__(
        self,
        obs_dim=7,
        action_dim=2,
        hidden_sizes=[256, 256],
        activation="tanh",
        use_layer_norm=True,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Activation
        if activation == "tanh":
            act_fn = nn.Tanh()
        elif activation == "relu":
            act_fn = nn.ReLU()
        elif activation == "elu":
            act_fn = nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build MLP
        layers = []
        prev_size = obs_dim

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_size))
            layers.append(act_fn)
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, action_dim))

        self.network = nn.Sequential(*layers)

        # Initialize output layer with small weights
        nn.init.orthogonal_(self.network[-1].weight, gain=0.01)
        nn.init.constant_(self.network[-1].bias, 0.0)

    def forward(self, obs):
        """
        Forward pass.

        Args:
            obs: Observation tensor [batch, obs_dim]

        Returns:
            action: Predicted action [batch, action_dim]
        """
        return self.network(obs)


def create_policy_network(config=None):
    """
    Factory function to create policy network with config dict.

    Args:
        config: Dict with network parameters (optional)

    Returns:
        policy_network: ParkingPolicyNetwork instance
    """
    if config is None:
        config = {}

    return ParkingPolicyNetwork(
        obs_dim=config.get("obs_dim", 7),
        action_dim=config.get("action_dim", 2),
        hidden_sizes=config.get("hidden_sizes", [256, 256]),
        activation=config.get("activation", "tanh"),
        use_layer_norm=config.get("use_layer_norm", True),
    )


if __name__ == "__main__":
    # Test network
    print("Testing ParkingPolicyNetwork...")

    net = ParkingPolicyNetwork(obs_dim=7, action_dim=2, hidden_sizes=[256, 256])

    # Test forward pass
    obs = torch.randn(4, 7)  # Batch of 4 observations
    policy_mean, policy_std, value = net(obs)

    print(f"Observation shape: {obs.shape}")
    print(f"Policy mean shape: {policy_mean.shape}")
    print(f"Policy std shape: {policy_std.shape}")
    print(f"Value shape: {value.shape}")

    # Test action sampling
    action, log_prob, value = net.get_action(obs)
    print(f"Action shape: {action.shape}")
    print(f"Log prob shape: {log_prob.shape}")
    print(f"Value shape: {value.shape}")

    # Test single observation
    single_obs = torch.randn(7)
    action, log_prob, value = net.get_action(single_obs)
    print(f"\nSingle observation action shape: {action.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in net.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    print("\n✓ All tests passed!")
