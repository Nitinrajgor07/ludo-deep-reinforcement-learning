"""Actor-Critic network architecture with shared representation for PPO."""

from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from phd_ludo_drl.environment.constants import ACTION_SPACE_SIZE
from phd_ludo_drl.encoding.cnn_encoder import StructuredCNNEncoder
from phd_ludo_drl.environment.action_mask import mask_action_logits


class ActorCriticNetwork(nn.Module):
    """Shared-representation Actor-Critic network matching published IEEE paper Section IV-B.

    Features:
    - Shared Structured CNN Encoder mapping board spatial tensor to 128-D latent representation.
    - Policy Head pi(a|s): Linear(128, 128) -> ReLU -> Linear(128, 4)
    - Value Head V(s): Linear(128, 128) -> ReLU -> Linear(128, 1)
    - Action Masking: Substitutes invalid action logits with -1e9 before softmax.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_units: int = 128,
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

        # Policy Head (Actor)
        self.policy_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, action_dim),
        )

        # Value Head (Critic)
        self.value_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, 1),
        )

    def extract_features(
        self,
        x: Union[np.ndarray, torch.Tensor],
        context: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Extract standardized 128-D feature representation."""
        device = next(self.parameters()).device
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        x = x.to(device)

        if self.use_internal_encoder and self.encoder is not None:
            if x.dim() in (2, 3) and x.shape[-2:] == (4, 52):
                features = self.encoder(x, context)
            else:
                features = x
        else:
            features = x

        if features.dim() == 1:
            features = features.unsqueeze(0)
        return features

    def get_value(
        self,
        x: Union[np.ndarray, torch.Tensor],
        context: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Compute state value estimate V(s)."""
        features = self.extract_features(x, context)
        return self.value_head(features).squeeze(-1)  # (B,)

    def get_action_and_value(
        self,
        spatial: Union[np.ndarray, torch.Tensor],
        context: Optional[Union[np.ndarray, torch.Tensor]],
        action_mask: Union[np.ndarray, torch.Tensor],
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute action, log_prob, entropy, and value estimate with strict action masking.

        Args:
            spatial: Tensor of shape (4, 52) or (B, 4, 52).
            context: Context tensor of shape (12,) or (B, 12).
            action_mask: Binary mask of shape (4,) or (B, 4).
            action: Optional action tensor if evaluating log_probs for update.
            deterministic: If True, selects argmax action.

        Returns:
            Tuple of (action, log_prob, entropy, value).
        """
        device = next(self.parameters()).device
        features = self.extract_features(spatial, context)  # (B, 128)

        if isinstance(action_mask, np.ndarray):
            mask_t = torch.from_numpy(action_mask).float().to(device)
        else:
            mask_t = action_mask.float().to(device)
        if mask_t.dim() == 1:
            mask_t = mask_t.unsqueeze(0)  # (1, 4)

        raw_logits = self.policy_head(features)  # (B, 4)
        masked_logits = mask_action_logits(raw_logits, mask_t)

        probs = torch.softmax(masked_logits, dim=-1)
        dist = Categorical(probs=probs)

        if action is None:
            if deterministic:
                action = torch.argmax(masked_logits, dim=-1)
            else:
                action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.value_head(features).squeeze(-1)

        return action, log_prob, entropy, value
