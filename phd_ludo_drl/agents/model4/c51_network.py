"""Categorical Q-network architecture (C51 / Distributional RL) for Model 4."""

from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from phd_ludo_drl.environment.constants import ACTION_SPACE_SIZE
from phd_ludo_drl.encoding.cnn_encoder import StructuredCNNEncoder
from phd_ludo_drl.environment.action_mask import mask_q_values


class CategoricalQNetwork(nn.Module):
    """Categorical Q-network (C51) matching Bellemare et al. (2017).

    Instead of estimating scalar Q(s, a), models the complete probability distribution
    over 51 discrete atoms: Z(s, a) on support [V_min, V_max].

    Key properties:
    - Utilizes common StructuredCNNEncoder mapping (4, 52) board + (12,) context to 128-D.
    - Distributional head: Linear(128, hidden_units) -> ReLU -> Linear(hidden_units, 4 * 51).
    - Expected Q-values: Q(s, a) = sum_i z_i * p_i(s, a).
    - Strict action masking applied during decision making.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_units: int = 128,
        action_dim: int = ACTION_SPACE_SIZE,
        num_atoms: int = 51,
        v_min: float = -20.0,
        v_max: float = 30.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_units = hidden_units
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (num_atoms - 1)

        # Support atoms: shape (51,)
        support = torch.linspace(v_min, v_max, num_atoms)
        self.register_buffer("support", support)

        # Shared CNN Encoder
        self.encoder = StructuredCNNEncoder(latent_dim=latent_dim)

        # Distributional value head
        self.head = nn.Sequential(
            nn.Linear(latent_dim, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, action_dim * num_atoms),
        )

    def forward(
        self,
        spatial: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute atom probabilities and expected Q-values.

        Args:
            spatial: Spatial board tensor of shape (B, 4, 52).
            context: Context tensor of shape (B, 12) or None.

        Returns:
            Tuple of:
            - probs: Atom probabilities of shape (B, action_dim, num_atoms)
            - q_values: Expected return values Q(s, a) of shape (B, action_dim)
        """
        features = self.encoder(spatial, context)  # (B, 128)
        logits = self.head(features)  # (B, action_dim * num_atoms)
        logits = logits.view(-1, self.action_dim, self.num_atoms)

        probs = F.softmax(logits, dim=-1)  # (B, action_dim, num_atoms)

        # Expected Q-values: dot product with support
        # probs: (B, A, N), support: (N,) -> q_values: (B, A)
        q_values = torch.sum(probs * self.support, dim=-1)

        return probs, q_values

    def get_q_values(
        self,
        spatial: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute only expected Q-values for efficient decision making."""
        _, q_values = self.forward(spatial, context)
        return q_values

    def select_action(
        self,
        spatial: torch.Tensor,
        context: Optional[torch.Tensor],
        action_mask: torch.Tensor,
        deterministic: bool = False,
        epsilon: float = 0.0,
    ) -> int:
        """Select action via masked expected Q-values or legal epsilon-greedy exploration.

        Args:
            spatial: Tensor of shape (1, 4, 52) or (4, 52).
            context: Tensor of shape (1, 12) or None.
            action_mask: Binary mask of shape (1, 4) or (4,).
            deterministic: If True, greedy selection.
            epsilon: Exploration probability.

        Returns:
            Selected legal action index [0..3].
        """
        if spatial.dim() == 2:
            spatial = spatial.unsqueeze(0)
        if context is not None and context.dim() == 1:
            context = context.unsqueeze(0)
        if action_mask.dim() == 1:
            action_mask = action_mask.unsqueeze(0)

        mask_np = action_mask.squeeze(0).detach().cpu().numpy()
        legal_actions = [i for i, m in enumerate(mask_np) if m == 1.0]

        if not legal_actions:
            raise ValueError("CategoricalQNetwork cannot select action with zero legal moves.")

        if not deterministic and np.random.rand() < epsilon:
            return int(np.random.choice(legal_actions))

        with torch.no_grad():
            q_values = self.get_q_values(spatial, context).squeeze(0)
            masked_q = mask_q_values(q_values, action_mask.squeeze(0))
            best_action = int(torch.argmax(masked_q).item())

        return best_action
