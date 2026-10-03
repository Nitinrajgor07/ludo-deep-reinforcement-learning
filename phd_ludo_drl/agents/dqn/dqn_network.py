"""Dueling Deep Q-Network with shared feature encoder for Ludo."""

from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

from phd_ludo_drl.environment.constants import ACTION_SPACE_SIZE
from phd_ludo_drl.encoding.cnn_encoder import StructuredCNNEncoder


class DuelingQNetwork(nn.Module):
    """Double Dueling Q-Network architecture matching published IEEE paper Section IV-A.

    Decomposes state-action values into state value V(s) and advantage A(s, a):
        Q(s, a) = V(s) + (A(s, a) - (1 / |A|) * sum_a' A(s, a'))

    Input:
        - Either raw spatial board tensor (B, 4, 52) [+ context (B, 12)] via internal StructuredCNNEncoder,
        - Or pre-encoded latent features (B, 128).
    """

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_units: int = 256,
        action_dim: int = ACTION_SPACE_SIZE,
        use_internal_encoder: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_units = hidden_units
        self.action_dim = action_dim
        self.use_internal_encoder = use_internal_encoder

        if self.use_internal_encoder:
            self.encoder = StructuredCNNEncoder(latent_dim=latent_dim)
        else:
            self.encoder = None

        # Value stream V(s): Linear(128, 256) -> ReLU -> Linear(256, 1)
        self.value_stream = nn.Sequential(
            nn.Linear(latent_dim, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, 1),
        )

        # Advantage stream A(s, a): Linear(128, 256) -> ReLU -> Linear(256, 4)
        self.advantage_stream = nn.Sequential(
            nn.Linear(latent_dim, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, action_dim),
        )

    def forward(
        self,
        x: Union[np.ndarray, torch.Tensor],
        context: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Compute Q-values for all 4 discrete actions.

        Args:
            x: Spatial tensor (B, 4, 52) or pre-encoded latent tensor (B, 128).
            context: Optional context tensor (B, 12) if x is spatial.

        Returns:
            Q-values tensor of shape (batch_size, 4).
        """
        device = next(self.parameters()).device

        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        x = x.to(device)

        if self.use_internal_encoder and self.encoder is not None:
            # If x is spatial (B, 4, 52) or (4, 52)
            if x.dim() in (2, 3) and x.shape[-2:] == (4, 52):
                features = self.encoder(x, context)
            else:
                features = x
        else:
            features = x

        # Ensure features is rank 2 (B, 128)
        if features.dim() == 1:
            features = features.unsqueeze(0)

        values = self.value_stream(features)           # (B, 1)
        advantages = self.advantage_stream(features)   # (B, 4)

        # Dueling aggregation: Q(s, a) = V(s) + (A(s, a) - mean_a' A(s, a'))
        q_values = values + (advantages - advantages.mean(dim=-1, keepdim=True))
        return q_values
