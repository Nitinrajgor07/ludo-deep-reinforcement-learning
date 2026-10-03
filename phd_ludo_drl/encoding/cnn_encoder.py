"""Structured CNN encoder for the primary 4-model research comparison."""

from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

from phd_ludo_drl.environment.constants import TRACK_LENGTH


class StructuredCNNEncoder(nn.Module):
    """Encodes 4-channel spatial board state into a standardized 128-D latent vector.

    Architecture (Published IEEE Paper Section III-C):
    - Input: 4-channel spatial tensor of shape (batch, 4, 52)
    - Conv Layer 1: Conv1d(in_channels=4, out_channels=32, kernel_size=3, padding=1, padding_mode='circular')
    - Activation 1: ReLU
    - Conv Layer 2: Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1, padding_mode='circular')
    - Activation 2: ReLU
    - Spatial Pooling / Flatten: AdaptiveAvgPool1d(8) -> (batch, 64 * 8 = 512)
    - Spatial Projection: Linear(512, 128) + ReLU
    - Context Fusion (Optional Context Vector 12-D): Linear(128 + 12, 128) + ReLU
    - Standardized Output Dimension: 128-D
    """

    def __init__(
        self,
        in_channels: int = 4,
        context_dim: int = 12,
        latent_dim: int = 128,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.context_dim = context_dim
        self.latent_dim = latent_dim

        # Circular padding captures the closed 52-square loop of the Ludo track
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
        )
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
        )
        self.relu2 = nn.ReLU()

        self.pool = nn.AdaptiveAvgPool1d(8)
        self.spatial_fc = nn.Sequential(
            nn.Linear(64 * 8, 128),
            nn.ReLU(),
        )

        # Context fusion projection
        self.fusion_fc = nn.Sequential(
            nn.Linear(128 + context_dim, latent_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        spatial: Union[np.ndarray, torch.Tensor],
        context: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Forward pass through Structured CNN Encoder.

        Args:
            spatial: Tensor of shape (4, 52) or (batch_size, 4, 52).
            context: Optional context of shape (12,) or (batch_size, 12).

        Returns:
            Latent feature tensor of shape (batch_size, 128) or (128,).
        """
        is_single = False

        if isinstance(spatial, np.ndarray):
            spatial = torch.from_numpy(spatial).float()

        if spatial.dim() == 2:
            is_single = True
            spatial = spatial.unsqueeze(0)  # (1, 4, 52)

        device = next(self.parameters()).device
        spatial = spatial.to(device)
        batch_size = spatial.shape[0]

        # Handle context vector
        if context is None:
            context_t = torch.zeros(batch_size, self.context_dim, device=device)
        else:
            if isinstance(context, np.ndarray):
                context_t = torch.from_numpy(context).float()
            else:
                context_t = context.float()
            if context_t.dim() == 1:
                context_t = context_t.unsqueeze(0)
            context_t = context_t.to(device)

        # Convolutional forward
        h = self.relu1(self.conv1(spatial))  # (B, 32, 52)
        h = self.relu2(self.conv2(h))        # (B, 64, 52)
        h = self.pool(h)                     # (B, 64, 8)
        h_flat = h.view(batch_size, -1)      # (B, 512)

        h_spatial = self.spatial_fc(h_flat)  # (B, 128)

        # Fuse with scalar context (dice, player, yard/goal stats)
        fused = torch.cat([h_spatial, context_t], dim=1)  # (B, 140)
        out = self.fusion_fc(fused)                       # (B, 128)

        if is_single:
            return out.squeeze(0)
        return out
