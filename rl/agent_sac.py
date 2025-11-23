import numpy as np
import torch

class SACAgent:
    def __init__(self, obs_dim, act_dim, cfg):
        # placeholders: implement actor/critic networks in networks.py
        self.obs_dim = obs_dim
        self.act_dim = act_dim

    def select_action(self, obs, deterministic=False):
        # placeholder: random policy for template
        return np.array([0.0, 0.0])

    def store(self, s, a, r, s2, done):
        pass

    def update(self):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass
