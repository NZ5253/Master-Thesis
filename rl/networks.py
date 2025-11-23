import torch.nn as nn
import torch

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=(128,128), activation=nn.ReLU):
        super().__init__()
        layers = []
        last = input_dim
        for h in hidden:
            layers.append(nn.Linear(last,h))
            layers.append(activation())
            last = h
        layers.append(nn.Linear(last, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
