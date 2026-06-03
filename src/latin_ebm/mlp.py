"""Small MLP head used as a residual on top of the linear EBM energy."""
from __future__ import annotations

import torch
import torch.nn as nn


class PerFootMLP(nn.Module):
    """Small MLP over per-foot dense features. One shared network across feet."""

    def __init__(self, input_dim: int, hidden_dim: int = 32, dropout: float = 0.0):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, foot_features: torch.Tensor) -> torch.Tensor:
        per_foot = self.net(foot_features).squeeze(-1)
        return per_foot.sum(dim=-1)
