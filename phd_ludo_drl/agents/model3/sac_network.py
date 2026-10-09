"""Neural network architectures for Discrete Soft Actor-Critic (Model 3)."""

from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from phd_ludo_drl.environment.constants import ACTION_SPACE_SIZE
from phd_ludo_drl.encoding.cnn_encoder import StructuredCNNEncoder
from phd_ludo_drl.environment.action_mask import mask_action_logits


class DiscreteCritic(nn.Module):
    """Twin Q-network architecture for Discrete Soft Actor-Critic.

    Computes action values Q_1(s, a) and Q_2(s, a) for all discrete actions a in [0..3].
    Uses the common Structured CNN Encoder for spatial board representation.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_units: int = 128,
        action_dim: int = ACTION_SPACE_SIZE,
        encoder1: Optional[nn.Module] = None,
        encoder2: Optional[nn.Module] = None,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim

        # Encoder for Q1
        self.encoder1 = encoder1 if encoder1 is not None else StructuredCNNEncoder(latent_dim=latent_dim)
        self.q1_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, action_dim),
        )

        # Encoder for Q2
        self.encoder2 = encoder2 if encoder2 is not None else StructuredCNNEncoder(latent_dim=latent_dim)
        self.q2_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, action_dim),
        )

        if freeze_encoder:
            self.freeze_encoder()

    def freeze_encoder(self) -> None:
        """Freeze both encoder networks to prevent gradient updates."""
        for p in self.encoder1.parameters():
            p.requires_grad = False
        for p in self.encoder2.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self) -> None:
        """Unfreeze both encoder networks to allow gradient updates."""
        for p in self.encoder1.parameters():
            p.requires_grad = True
        for p in self.encoder2.parameters():
            p.requires_grad = True

    def forward(
        self,
        spatial: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Q1 and Q2 values for all actions.

        Returns:
            Tuple of (q1_values, q2_values) each of shape (B, 4).
        """
        if spatial.dim() == 2:
            spatial = spatial.unsqueeze(0)
        if context is not None and context.dim() == 1:
            context = context.unsqueeze(0)

        f1 = self.encoder1(spatial, context)
        q1 = self.q1_head(f1)

        f2 = self.encoder2(spatial, context)
        q2 = self.q2_head(f2)

        return q1, q2

    def q1(
        self,
        spatial: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute only Q1 values."""
        if spatial.dim() == 2:
            spatial = spatial.unsqueeze(0)
        if context is not None and context.dim() == 1:
            context = context.unsqueeze(0)

        f1 = self.encoder1(spatial, context)
        return self.q1_head(f1)


class DiscreteActor(nn.Module):
    """Actor (policy) network for Discrete Soft Actor-Critic with Action Masking.

    Outputs action probabilities pi(a|s) and log-probabilities log pi(a|s).
    Invalid actions are masked with -1e9 before softmax, ensuring zero probability.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_units: int = 128,
        action_dim: int = ACTION_SPACE_SIZE,
        encoder: Optional[nn.Module] = None,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim

        self.encoder = encoder if encoder is not None else StructuredCNNEncoder(latent_dim=latent_dim)
        self.policy_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, action_dim),
        )

        if freeze_encoder:
            self.freeze_encoder()

    def freeze_encoder(self) -> None:
        """Freeze actor encoder to prevent gradient updates."""
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self) -> None:
        """Unfreeze actor encoder to allow gradient updates."""
        for p in self.encoder.parameters():
            p.requires_grad = True

    def forward(
        self,
        spatial: torch.Tensor,
        context: Optional[torch.Tensor],
        action_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute action probabilities and log-probabilities with action masking.

        Args:
            spatial: Spatial board tensor (B, 4, 52) or (4, 52).
            context: Context tensor (B, 12), (12,) or None.
            action_mask: Action mask tensor (B, 4) or (4,).

        Returns:
            Tuple of (action_probs, log_action_probs) each of shape (B, 4).
        """
        if spatial.dim() == 2:
            spatial = spatial.unsqueeze(0)
        if context is not None and context.dim() == 1:
            context = context.unsqueeze(0)
        if action_mask.dim() == 1:
            action_mask = action_mask.unsqueeze(0)

        device = next(self.parameters()).device
        if spatial.device != device:
            spatial = spatial.to(device)
        if context is not None and context.device != device:
            context = context.to(device)
        if action_mask.device != device:
            action_mask = action_mask.to(device)
        if action_mask.dtype != torch.float32:
            action_mask = action_mask.float()

        features = self.encoder(spatial, context)
        logits = self.policy_head(features)
        masked_logits = mask_action_logits(logits, action_mask)

        # Compute stable probabilities and log-probabilities
        probs = torch.softmax(masked_logits, dim=-1)
        log_probs = torch.log_softmax(masked_logits, dim=-1)

        # For strictly masked illegal actions, ensure log_prob is clamped for numerical stability
        # when multiplying by 0.0 probability: 0 * -1e9 -> 0
        clamped_log_probs = torch.where(
            action_mask.bool(),
            log_probs,
            torch.zeros_like(log_probs),
        )

        return probs, clamped_log_probs

    def select_action(
        self,
        spatial: torch.Tensor,
        context: Optional[torch.Tensor],
        action_mask: torch.Tensor,
        deterministic: bool = False,
    ) -> int:
        """Select a strictly legal discrete action.

        Args:
            spatial: Spatial tensor (1, 4, 52) or (4, 52).
            context: Context tensor (1, 12) or None.
            action_mask: Binary mask (1, 4) or (4,).
            deterministic: If True, selects argmax action.

        Returns:
            Legal action index [0..3].
        """
        if spatial.dim() == 2:
            spatial = spatial.unsqueeze(0)
        if context is not None and context.dim() == 1:
            context = context.unsqueeze(0)
        if action_mask.dim() == 1:
            action_mask = action_mask.unsqueeze(0)

        with torch.no_grad():
            probs, _ = self.forward(spatial, context, action_mask)
            if deterministic:
                action = torch.argmax(probs, dim=-1)
            else:
                dist = Categorical(probs=probs)
                action = dist.sample()

        return int(action.item())
