"""Flat baseline encoder for ablation studies (matches published paper Table II baseline)."""

from typing import Union
import numpy as np
import torch
import torch.nn as nn

from phd_ludo_drl.encoding.raw_observation import RAW_STATE_DIMENSION


class FlatEncoder(nn.Module):
    """Processes flat observation vector through 2 Fully-Connected layers to 128-D.

    Published Architecture (IEEE Paper Section III-C):
    - Input: 60-D raw observation vector
    - Layer 1: Linear(60, 256) + ReLU
    - Layer 2: Linear(256, 128) + ReLU
    - Output: 128-D latent state vector
    """

    def __init__(self, input_dim: int = RAW_STATE_DIMENSION, hidden_dim: int = 256, latent_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )

    def forward(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Forward pass mapping input vector(s) to 128-D representation.

        Args:
            x: Input of shape (input_dim,) or (batch_size, input_dim).

        Returns:
            Latent representation tensor of shape (batch_size, 128) or (128,).
        """
        is_single = False
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        if x.dim() == 1:
            is_single = True
            x = x.unsqueeze(0)

        # Move to model device
        device = next(self.parameters()).device
        x = x.to(device)

        out = self.network(x)

        if is_single:
            return out.squeeze(0)
        return out
