"""Authoritative Shared Structured CNN Encoder for PhD Ludo DRL.

Conforms strictly to FINAL_EXPERIMENT_PROTOCOL.md Section 4 (Version 1.0.0 FROZEN):
1. Input spatial tensor: 4 x 52
2. Conv1D: 4 -> 32 (kernel_size=3, padding=1, padding_mode='circular') + ReLU()
3. Conv1D: 32 -> 64 (kernel_size=3, padding=1, padding_mode='circular') + ReLU()
4. AdaptiveAvgPool1D: output length = 1 -> (B, 64, 1)
5. Flatten: 64 features -> (B, 64)
6. Concatenate with: 12-dimensional context -> (B, 64 + 12) = (B, 76)
7. Fully connected projection: Linear(76, 128) + ReLU()
8. Output: Standardized 128-dimensional latent state feature vector h(s) in R^128.
"""

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

from phd_ludo_drl.environment.constants import TRACK_LENGTH

SPATIAL_IN_CHANNELS: int = 4
SPATIAL_LENGTH: int = TRACK_LENGTH  # 52
CONV1_OUT_CHANNELS: int = 32
CONV2_OUT_CHANNELS: int = 64
POOL_OUTPUT_SIZE: int = 1
SPATIAL_FEATURE_DIM: int = CONV2_OUT_CHANNELS * POOL_OUTPUT_SIZE  # 64
CONTEXT_INPUT_DIM: int = 12
FUSED_INPUT_DIM: int = SPATIAL_FEATURE_DIM + CONTEXT_INPUT_DIM     # 76
LATENT_OUTPUT_DIM: int = 128


class StructuredCNNEncoder(nn.Module):
    """Encodes 4-channel spatial board state and 12-D context into a standardized 128-D latent vector.

    Architecture (Protocol Section 4):
    - Input Spatial: (B, 4, 52)
    - Conv1D Layer 1: Conv1d(4, 32, kernel_size=3, padding=1, padding_mode='circular') + ReLU
    - Conv1D Layer 2: Conv1d(32, 64, kernel_size=3, padding=1, padding_mode='circular') + ReLU
    - Adaptive Pooling: AdaptiveAvgPool1d(1) -> (B, 64, 1)
    - Spatial Flatten: (B, 64)
    - Context Fusion: Concat with (B, 12) context -> (B, 76)
    - Linear Projection: Linear(76, 128) + ReLU
    - Latent Representation: (B, 128)
    """

    def __init__(
        self,
        in_channels: int = SPATIAL_IN_CHANNELS,
        context_dim: int = CONTEXT_INPUT_DIM,
        latent_dim: int = LATENT_OUTPUT_DIM,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.context_dim = context_dim
        self.latent_dim = latent_dim

        # Circular padding captures the closed 52-square loop of the Ludo track
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=CONV1_OUT_CHANNELS,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
        )
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(
            in_channels=CONV1_OUT_CHANNELS,
            out_channels=CONV2_OUT_CHANNELS,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
        )
        self.relu2 = nn.ReLU()

        self.pool = nn.AdaptiveAvgPool1d(POOL_OUTPUT_SIZE)
        self.spatial_feature_dim = CONV2_OUT_CHANNELS * POOL_OUTPUT_SIZE  # 64
        self.fused_dim = self.spatial_feature_dim + context_dim           # 76

        # Fully connected projection: 76 -> 128
        self.fc = nn.Linear(self.fused_dim, latent_dim)
        self.relu3 = nn.ReLU()

    @classmethod
    def verify_dimensions(cls) -> Dict[str, Any]:
        """Programmatically verify layer dimensions against the frozen protocol contract."""
        inst = cls()
        spec = {
            "spatial_input_shape": (SPATIAL_IN_CHANNELS, SPATIAL_LENGTH),
            "conv1_in_channels": inst.conv1.in_channels,
            "conv1_out_channels": inst.conv1.out_channels,
            "conv1_kernel_size": inst.conv1.kernel_size[0],
            "conv1_padding_mode": inst.conv1.padding_mode,
            "conv2_in_channels": inst.conv2.in_channels,
            "conv2_out_channels": inst.conv2.out_channels,
            "conv2_kernel_size": inst.conv2.kernel_size[0],
            "conv2_padding_mode": inst.conv2.padding_mode,
            "pool_output_size": POOL_OUTPUT_SIZE,
            "spatial_flatten_features": inst.spatial_feature_dim,
            "context_dimension": inst.context_dim,
            "total_fused_features": inst.fused_dim,
            "fc_in_features": inst.fc.in_features,
            "fc_out_features": inst.fc.out_features,
            "latent_output_dimension": inst.latent_dim,
        }

        assert spec["spatial_input_shape"] == (4, 52), "Spatial input shape must be (4, 52)"
        assert spec["conv1_in_channels"] == 4 and spec["conv1_out_channels"] == 32, "Conv1 must be 4 -> 32"
        assert spec["conv1_padding_mode"] == "circular", "Conv1 padding mode must be circular"
        assert spec["conv2_in_channels"] == 32 and spec["conv2_out_channels"] == 64, "Conv2 must be 32 -> 64"
        assert spec["conv2_padding_mode"] == "circular", "Conv2 padding mode must be circular"
        assert spec["pool_output_size"] == 1, "AdaptiveAvgPool1D output length must be 1"
        assert spec["spatial_flatten_features"] == 64, "Spatial flatten features must be 64"
        assert spec["context_dimension"] == 12, "Context dimension must be 12"
        assert spec["total_fused_features"] == 76, "Fused features must be 64 + 12 = 76"
        assert spec["fc_in_features"] == 76 and spec["fc_out_features"] == 128, "FC projection must be 76 -> 128"
        assert spec["latent_output_dimension"] == 128, "Final output dimension must be 128"

        return spec

    def forward(
        self,
        spatial: Union[np.ndarray, torch.Tensor],
        context: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Forward pass through the Structured CNN Encoder.

        Args:
            spatial: Tensor of shape (4, 52) or (batch_size, 4, 52).
            context: Optional context of shape (12,) or (batch_size, 12).
                     Defaults to zeros if omitted.

        Returns:
            Latent feature tensor of shape (128,) for single input or
            (batch_size, 128) for batched input.

        Raises:
            ValueError: If spatial or context tensor shapes violate dimensional contracts.
        """
        is_single = False

        if isinstance(spatial, np.ndarray):
            spatial = torch.from_numpy(spatial).float()
        elif not isinstance(spatial, torch.Tensor):
            raise TypeError(f"Expected numpy.ndarray or torch.Tensor for spatial, got {type(spatial).__name__}")

        # Validate spatial tensor rank and dimensions
        if spatial.dim() == 2:
            is_single = True
            if spatial.shape != (self.in_channels, SPATIAL_LENGTH):
                raise ValueError(
                    f"Expected single spatial tensor shape ({self.in_channels}, {SPATIAL_LENGTH}), got {tuple(spatial.shape)}"
                )
            spatial = spatial.unsqueeze(0)  # (1, 4, 52)
        elif spatial.dim() == 3:
            if spatial.shape[1] != self.in_channels or spatial.shape[2] != SPATIAL_LENGTH:
                raise ValueError(
                    f"Expected batched spatial tensor shape (B, {self.in_channels}, {SPATIAL_LENGTH}), got {tuple(spatial.shape)}"
                )
        else:
            raise ValueError(
                f"Invalid spatial tensor rank {spatial.dim()}. Expected rank 2 (4, 52) or rank 3 (B, 4, 52)."
            )

        device = next(self.parameters()).device
        spatial = spatial.to(device)
        batch_size = spatial.shape[0]

        # Handle and validate context vector
        if context is None:
            context_t = torch.zeros(batch_size, self.context_dim, device=device, dtype=spatial.dtype)
        else:
            if isinstance(context, np.ndarray):
                context_t = torch.from_numpy(context).float()
            elif isinstance(context, torch.Tensor):
                context_t = context.float()
            else:
                raise TypeError(f"Expected numpy.ndarray or torch.Tensor for context, got {type(context).__name__}")

            if context_t.dim() == 1:
                if context_t.shape[0] != self.context_dim:
                    raise ValueError(
                        f"Expected single context vector dimension {self.context_dim}, got {context_t.shape[0]}"
                    )
                context_t = context_t.unsqueeze(0)  # (1, 12)
            elif context_t.dim() == 2:
                if context_t.shape[1] != self.context_dim:
                    raise ValueError(
                        f"Expected batched context feature dimension {self.context_dim}, got {context_t.shape[1]}"
                    )
            else:
                raise ValueError(
                    f"Invalid context tensor rank {context_t.dim()}. Expected rank 1 (12,) or rank 2 (B, 12)."
                )

            if context_t.shape[0] != batch_size:
                raise ValueError(
                    f"Batch size mismatch: spatial batch size is {batch_size}, but context batch size is {context_t.shape[0]}"
                )

            context_t = context_t.to(device)

        # 1. First circular convolution: 4 -> 32
        h = self.relu1(self.conv1(spatial))      # (B, 32, 52)

        # 2. Second circular convolution: 32 -> 64
        h = self.relu2(self.conv2(h))            # (B, 64, 52)

        # 3. Adaptive average pooling to output length 1
        h = self.pool(h)                         # (B, 64, 1)

        # 4. Flatten: exactly 64 spatial features
        h_flat = h.view(batch_size, self.spatial_feature_dim)  # (B, 64)

        # 5. Concatenate with 12-dimensional context: 64 + 12 = 76
        fused = torch.cat([h_flat, context_t], dim=1)          # (B, 76)

        # 6. Dense projection: 76 -> 128
        out = self.relu3(self.fc(fused))                       # (B, 128)

        if is_single:
            return out.squeeze(0)
        return out
