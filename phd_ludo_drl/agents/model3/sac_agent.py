"""Discrete Soft Actor-Critic (SAC) Agent for Ludo (Model 3)."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from phd_ludo_drl.config.config_manager import DiscreteSACConfig
from phd_ludo_drl.agents.base_agent import BaseAgent
from phd_ludo_drl.agents.model3.sac_network import DiscreteActor, DiscreteCritic
from phd_ludo_drl.agents.model3.replay_buffer import DiscreteSACReplayBuffer
from phd_ludo_drl.encoding.structured_representation import (
    StructuredLudoState,
    build_structured_representation,
)


class DiscreteSACAgent(BaseAgent):
    """Discrete Soft Actor-Critic agent matching Christodoulou (2019) / Haarnoja et al.

    Features:
    - Twin Critic Q-networks to eliminate maximization bias.
    - Soft Polyak target network averaging (tau = 0.005).
    - Exact expectation calculation for discrete target state values:
      V(s') = sum_a pi(a|s') * [ min_j Q_target,j(s', a) - alpha * log pi(a|s') ]
    - Action masking: illegal moves masked before softmax and excluded from expectations.
    - Automatic entropy temperature adjustment (alpha) via dual gradient descent.
    """

    def __init__(
        self,
        config: Optional[DiscreteSACConfig] = None,
        latent_dim: int = 128,
        encoder: Optional[nn.Module] = None,
        freeze_encoder: bool = False,
        device: Optional[torch.device] = None,
        name: str = "DiscreteSAC",
    ):
        super().__init__(name=name, latent_dim=latent_dim, device=device)
        self.config = config or DiscreteSACConfig()

        # Twin Critic networks & Target Critic networks
        self.critic = DiscreteCritic(
            latent_dim=latent_dim,
            hidden_units=self.config.hidden_units,
            freeze_encoder=freeze_encoder,
        ).to(self.device)

        self.critic_target = DiscreteCritic(
            latent_dim=latent_dim,
            hidden_units=self.config.hidden_units,
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.eval()

        # Actor network
        self.actor = DiscreteActor(
            latent_dim=latent_dim,
            hidden_units=self.config.hidden_units,
            encoder=encoder,
            freeze_encoder=freeze_encoder,
        ).to(self.device)

        # Optimizers (trainable parameters only)
        self.critic_optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.critic.parameters()),
            lr=self.config.critic_lr,
        )
        self.actor_optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.actor.parameters()),
            lr=self.config.actor_lr,
        )

        # Replay Buffer
        self.buffer = DiscreteSACReplayBuffer(
            capacity=self.config.buffer_capacity,
        )

        # Hyperparameters
        self.gamma = self.config.gamma
        self.tau = self.config.tau
        self.tune_alpha = self.config.tune_alpha

        # Target entropy: -log(1 / |A|) * target_entropy_ratio
        # For |A| = 4, log(4) ~ 1.3863. With ratio 0.98, target ~ 1.3586
        self.target_entropy = float(np.log(4) * self.config.target_entropy_ratio)
        self.log_alpha = torch.tensor(
            np.log(self.config.initial_alpha),
            dtype=torch.float32,
            requires_grad=True,
            device=self.device,
        )
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.config.alpha_lr)

        self.total_steps = 0
        self.update_count = 0

    @property
    def alpha(self) -> torch.Tensor:
        """Current entropy temperature parameter."""
        return self.log_alpha.exp()

    def freeze_encoder(self) -> None:
        """Freeze encoder parameters in actor and critic networks."""
        self.actor.freeze_encoder()
        self.critic.freeze_encoder()
        self.critic_optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.critic.parameters()),
            lr=self.config.critic_lr,
        )
        self.actor_optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.actor.parameters()),
            lr=self.config.actor_lr,
        )

    def unfreeze_encoder(self) -> None:
        """Unfreeze encoder parameters in actor and critic networks."""
        self.actor.unfreeze_encoder()
        self.critic.unfreeze_encoder()
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=self.config.critic_lr,
        )
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=self.config.actor_lr,
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
            action_mask: Binary legality mask (4,).
            context: Optional context tensor (12,) if state is spatial tensor.
            deterministic: If True, selects argmax probability action.

        Returns:
            Legal action index [0..3].
        """
        spatial_t, context_t = self._prepare_inputs(state, context=context)
        if isinstance(action_mask, np.ndarray):
            mask_t = torch.from_numpy(action_mask).float().to(self.device)
        else:
            mask_t = action_mask.float().to(self.device)

        return self.actor.select_action(
            spatial=spatial_t,
            context=context_t,
            action_mask=mask_t,
            deterministic=deterministic,
        )

    def compute_target(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        next_states: torch.Tensor,
        next_contexts: Optional[torch.Tensor],
        next_action_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Compute exact expectation value target for Discrete SAC.

        Formula:
            V(s') = sum_{a in A_legal} pi(a|s') * [ min_j Q_target,j(s', a) - alpha * log pi(a|s') ]
            target = r + (1 - d) * gamma * V(s')

        Args:
            rewards: Batch tensor of step rewards (B,).
            dones: Batch tensor of done flags (B,).
            next_states: Batch tensor of next spatial states (B, 4, 52).
            next_contexts: Batch tensor of next contexts (B, 12).
            next_action_masks: Batch tensor of next valid action masks (B, 4).

        Returns:
            Target Q values tensor of shape (B,).
        """
        with torch.no_grad():
            next_probs, next_log_probs = self.actor(next_states, next_contexts, next_action_masks)
            q1_target, q2_target = self.critic_target(next_states, next_contexts)
            min_q_target = torch.min(q1_target, q2_target)

            # V(s') = sum_a pi(a|s') * [ min Q(s', a) - alpha * log pi(a|s') ]
            next_v = (next_probs * (min_q_target - self.alpha * next_log_probs)).sum(dim=-1)
            target_q = rewards + (1.0 - dones) * self.gamma * next_v
        return target_q

    def train_step(self, batch_size: int = 64) -> Optional[Dict[str, float]]:
        """Execute a single off-policy Soft Actor-Critic optimization step."""
        if len(self.buffer) < batch_size:
            return None

        batch = self.buffer.sample(batch_size, self.device)
        states = batch["states"]
        contexts = batch["contexts"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_states = batch["next_states"]
        next_contexts = batch["next_contexts"]
        dones = batch["dones"]
        action_masks = batch["action_masks"]
        next_action_masks = batch["next_action_masks"]

        # -------------------------------------------------------------
        # 1. Critic Update with Exact Expectation Value Target
        # -------------------------------------------------------------
        target_q = self.compute_target(rewards, dones, next_states, next_contexts, next_action_masks)

        q1, q2 = self.critic(states, contexts)
        q1_a = q1.gather(1, actions.unsqueeze(1)).squeeze(1)
        q2_a = q2.gather(1, actions.unsqueeze(1)).squeeze(1)

        critic_loss = 0.5 * (F.mse_loss(q1_a, target_q) + F.mse_loss(q2_a, target_q))

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10.0)
        self.critic_optimizer.step()

        # -------------------------------------------------------------
        # 2. Actor Policy Update
        # -------------------------------------------------------------
        probs, log_probs = self.actor(states, contexts, action_masks)
        with torch.no_grad():
            q1_curr, q2_curr = self.critic(states, contexts)
            min_q_curr = torch.min(q1_curr, q2_curr)

        # Policy loss: sum_a pi(a|s) * [ alpha * log pi(a|s) - min Q(s, a) ]
        actor_loss = (probs * (self.alpha.detach() * log_probs - min_q_curr)).sum(dim=-1).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
        self.actor_optimizer.step()

        # -------------------------------------------------------------
        # 3. Temperature Parameter (Alpha) Tuning
        # -------------------------------------------------------------
        alpha_loss_val = 0.0
        if self.tune_alpha:
            entropy_diff = (probs.detach() * (log_probs.detach() + self.target_entropy)).sum(dim=-1)
            alpha_loss = -(self.log_alpha * entropy_diff).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha_loss_val = float(alpha_loss.item())

        # -------------------------------------------------------------
        # 4. Soft Target Update: Polyak averaging
        # -------------------------------------------------------------
        for target_p, online_p in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_p.data.copy_(self.tau * online_p.data + (1.0 - self.tau) * target_p.data)

        self.total_steps += 1
        self.update_count += 1

        with torch.no_grad():
            entropy = -(probs * log_probs).sum(dim=-1).mean().item()

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": alpha_loss_val,
            "alpha": float(self.alpha.item()),
            "mean_q1": float(q1_a.mean().item()),
            "mean_q2": float(q2_a.mean().item()),
            "entropy": float(entropy),
            "updates": self.update_count,
        }

    def save_checkpoint(self, path: Path) -> None:
        """Save network parameters and optimizer states to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor": self.actor.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "total_steps": self.total_steps,
            "update_count": self.update_count,
        }, path)

    def load_checkpoint(self, path: Path) -> None:
        """Load network parameters and optimizer states from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.log_alpha.data.copy_(checkpoint["log_alpha"].to(self.device))
        self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])
        self.total_steps = checkpoint.get("total_steps", 0)
        self.update_count = checkpoint.get("update_count", 0)
