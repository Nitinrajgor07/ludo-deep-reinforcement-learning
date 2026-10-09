"""Authoritative PPO Agent for Ludo with Structured CNN Encoder and Action Masking."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from phd_ludo_drl.config.config_manager import PPOConfig
from phd_ludo_drl.agents.base_agent import BaseAgent
from phd_ludo_drl.agents.ppo.ppo_network import ActorCriticNetwork
from phd_ludo_drl.agents.ppo.rollout_buffer import PPORolloutBuffer
from phd_ludo_drl.encoding.structured_representation import (
    StructuredLudoState,
    build_structured_representation,
)


class PPOAgent(BaseAgent):
    """Authoritative PPO Agent matching published IEEE research paper specification.

    Key Features:
    - Shared Structured CNN Encoder representation (128-D).
    - Actor policy head and Critic value head with strict action masking.
    - Clipped surrogate policy objective (epsilon=0.2).
    - Generalized Advantage Estimation (gamma=0.99, lambda=0.95).
    - Clipped value loss and entropy regularization term.
    - Gradient norm clipping (0.5).
    """

    def __init__(
        self,
        config: Optional[PPOConfig] = None,
        latent_dim: int = 128,
        encoder: Optional[nn.Module] = None,
        freeze_encoder: bool = False,
        device: Optional[torch.device] = None,
        name: str = "PPO",
    ):
        super().__init__(name=name, latent_dim=latent_dim, device=device)
        self.config = config or PPOConfig()

        # Shared Actor-Critic Network
        self.network = ActorCriticNetwork(
            latent_dim=latent_dim,
            hidden_units=self.config.hidden_units,
            encoder=encoder,
            freeze_encoder=freeze_encoder,
        ).to(self.device)

        # Optimizer: Adam with published lr=3e-4, eps=1e-5 (only trainable parameters)
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.network.parameters()),
            lr=self.config.learning_rate,
            eps=1e-5,
        )

        self.total_steps = 0
        self.update_count = 0

    def freeze_encoder(self) -> None:
        """Freeze shared encoder parameters and reinitialize optimizer."""
        self.network.freeze_encoder()
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.network.parameters()),
            lr=self.config.learning_rate,
            eps=1e-5,
        )

    def unfreeze_encoder(self) -> None:
        """Unfreeze shared encoder parameters and reinitialize optimizer."""
        self.network.unfreeze_encoder()
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.config.learning_rate,
            eps=1e-5,
        )

    def seed(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    def _prepare_inputs(
        self,
        state: Union[Dict[str, Any], StructuredLudoState, np.ndarray, torch.Tensor],
        context: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Convert state input to standard spatial and context tensors."""
        if hasattr(state, "spatial_tensor") and hasattr(state, "context_vector"):
            spatial_t = torch.from_numpy(state.spatial_tensor).float().to(self.device)
            context_t = torch.from_numpy(state.context_vector).float().to(self.device)
            if spatial_t.dim() == 2:
                spatial_t = spatial_t.unsqueeze(0)
            if context_t.dim() == 1:
                context_t = context_t.unsqueeze(0)
            return spatial_t, context_t

        if isinstance(state, dict):
            struct_state = build_structured_representation(state)
            spatial_t = torch.from_numpy(struct_state.spatial_tensor).float().unsqueeze(0).to(self.device)
            context_t = torch.from_numpy(struct_state.context_vector).float().unsqueeze(0).to(self.device)
            return spatial_t, context_t

        if isinstance(state, np.ndarray):
            spatial_t = torch.from_numpy(state).float().to(self.device)
        else:
            spatial_t = state.float().to(self.device)

        if spatial_t.dim() == 2:
            spatial_t = spatial_t.unsqueeze(0)

        context_t = None
        if context is not None:
            if isinstance(context, np.ndarray):
                context_t = torch.from_numpy(context).float().to(self.device)
            else:
                context_t = context.float().to(self.device)
            if context_t.dim() == 1:
                context_t = context_t.unsqueeze(0)

        return spatial_t, context_t

    def select_action(
        self,
        state: Union[Dict[str, Any], StructuredLudoState, np.ndarray, torch.Tensor],
        action_mask: Union[np.ndarray, torch.Tensor],
        context: Optional[Union[np.ndarray, torch.Tensor]] = None,
        deterministic: bool = False,
    ) -> int:
        """Select a strictly legal action using the masked policy.

        Args:
            state: Game state dict, StructuredLudoState, or spatial tensor.
            action_mask: Binary legality mask of shape (4,).
            context: Optional context tensor (12,) if state is spatial tensor.
            deterministic: If True, selects greedy argmax action.

        Returns:
            Selected legal action index [0..3].
        """
        spatial_t, context_t = self._prepare_inputs(state, context=context)
        with torch.no_grad():
            action, _, _, _ = self.network.get_action_and_value(
                spatial=spatial_t,
                context=context_t,
                action_mask=action_mask,
                deterministic=deterministic,
            )
        return int(action.item())

    def step(
        self,
        state: Union[Dict[str, Any], StructuredLudoState, np.ndarray, torch.Tensor],
        action_mask: Union[np.ndarray, torch.Tensor],
        context: Optional[Union[np.ndarray, torch.Tensor]] = None,
        deterministic: bool = False,
    ) -> Tuple[int, float, float]:
        """Compute action, log_prob, and value estimate for rollout collection.

        Args:
            state: Game state dict, StructuredLudoState, or spatial tensor.
            action_mask: Binary legality mask of shape (4,).
            context: Optional context tensor (12,) if state is spatial tensor.
            deterministic: If True, selects greedy argmax action.

        Returns:
            Tuple of (action_idx, log_prob, value).
        """
        spatial_t, context_t = self._prepare_inputs(state, context=context)
        with torch.no_grad():
            action, log_prob, _, value = self.network.get_action_and_value(
                spatial=spatial_t,
                context=context_t,
                action_mask=action_mask,
                deterministic=deterministic,
            )
        return int(action.item()), float(log_prob.item()), float(value.item())

    def get_value(
        self,
        state: Union[Dict[str, Any], StructuredLudoState, np.ndarray, torch.Tensor],
        context: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> float:
        """Compute value estimate V(s) for a given state."""
        spatial_t, context_t = self._prepare_inputs(state, context=context)
        with torch.no_grad():
            val = self.network.get_value(spatial_t, context_t)
        return float(val.item())

    def update(self, rollout_buffer: PPORolloutBuffer) -> Dict[str, float]:
        """Perform PPO policy and value optimization over collected rollouts.

        Implements:
        - Clipped surrogate policy objective
        - Clipped value loss
        - Entropy bonus
        - Gradient norm clipping

        Args:
            rollout_buffer: Buffer containing trajectory data with GAE advantages.

        Returns:
            Dict containing optimization metrics.
        """
        if len(rollout_buffer) == 0:
            return {}

        policy_losses: List[float] = []
        value_losses: List[float] = []
        entropy_losses: List[float] = []
        total_losses: List[float] = []
        clip_fractions: List[float] = []
        approx_kls: List[float] = []

        num_epochs = self.config.num_epochs
        mini_batch_size = self.config.mini_batch_size

        for _ in range(num_epochs):
            for batch in rollout_buffer.get_mini_batches(mini_batch_size, self.device):
                states = batch["states"]
                contexts = batch["contexts"]
                actions = batch["actions"]
                action_masks = batch["action_masks"]
                old_log_probs = batch["old_log_probs"]
                old_values = batch["old_values"]
                returns = batch["returns"]
                advantages = batch["advantages"]

                # Evaluate current policy on mini-batch transitions
                _, new_log_probs, entropy, new_values = self.network.get_action_and_value(
                    spatial=states,
                    context=contexts,
                    action_mask=action_masks,
                    action=actions,
                )

                # Policy Loss (Clipped Surrogate Objective)
                log_ratio = new_log_probs - old_log_probs
                ratio = torch.exp(log_ratio)

                surr1 = ratio * advantages
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_epsilon,
                    1.0 + self.config.clip_epsilon,
                ) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value Loss (with optional clipping)
                v_loss_unclipped = (new_values - returns) ** 2
                v_clipped = old_values + torch.clamp(
                    new_values - old_values,
                    -self.config.clip_epsilon,
                    self.config.clip_epsilon,
                )
                v_loss_clipped = (v_clipped - returns) ** 2
                value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                # Entropy Bonus (negative loss for maximizing entropy)
                entropy_loss = -entropy.mean()

                # Total Loss
                total_loss = (
                    policy_loss
                    + self.config.value_coeff * value_loss
                    + self.config.entropy_coeff * entropy_loss
                )

                # Gradient Step
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=self.config.max_grad_norm)
                self.optimizer.step()

                # Track metrics
                total_losses.append(float(total_loss.item()))
                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropy_losses.append(float(entropy.mean().item()))

                # Diagnostic metrics
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
                    approx_kls.append(float(approx_kl))
                    clip_fraction = ((ratio - 1.0).abs() > self.config.clip_epsilon).float().mean().item()
                    clip_fractions.append(float(clip_fraction))

        self.update_count += 1
        self.total_steps += len(rollout_buffer)

        return {
            "loss": float(np.mean(total_losses)),
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropy_losses)),
            "approx_kl": float(np.mean(approx_kls)),
            "clip_fraction": float(np.mean(clip_fractions)),
            "updates": self.update_count,
            "total_steps": self.total_steps,
        }

    def save_checkpoint(self, path: Path) -> None:
        """Save model weights and optimizer state to file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "update_count": self.update_count,
        }, path)

    def load_checkpoint(self, path: Path) -> None:
        """Load model weights and optimizer state from file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.total_steps = checkpoint.get("total_steps", 0)
        self.update_count = checkpoint.get("update_count", 0)
