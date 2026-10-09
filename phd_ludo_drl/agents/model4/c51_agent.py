"""Categorical DQN (C51 / Distributional RL) Agent for Ludo (Model 4)."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from phd_ludo_drl.config.config_manager import CategoricalDQNConfig
from phd_ludo_drl.agents.base_agent import BaseAgent
from phd_ludo_drl.agents.model4.c51_network import CategoricalQNetwork
from phd_ludo_drl.agents.model4.replay_buffer import C51ReplayBuffer
from phd_ludo_drl.environment.action_mask import mask_q_values
from phd_ludo_drl.encoding.structured_representation import (
    StructuredLudoState,
    build_structured_representation,
)


class CategoricalDQNAgent(BaseAgent):
    """Categorical DQN (C51) Agent matching Bellemare et al. (2017).

    Features:
    - Distributional value representation over 51 discrete atoms.
    - Categorical Bellman projection with linear interpolation.
    - Cross-entropy / KL divergence loss.
    - Action masking: illegal actions masked out during Q-argmax decision making.
    - Soft Polyak target updates (tau = 0.005).
    - Epsilon-greedy exploration over legal action subset.
    """

    def __init__(
        self,
        config: Optional[CategoricalDQNConfig] = None,
        latent_dim: int = 128,
        encoder: Optional[nn.Module] = None,
        freeze_encoder: bool = False,
        device: Optional[torch.device] = None,
        name: str = "CategoricalDQN_C51",
    ):
        super().__init__(name=name, latent_dim=latent_dim, device=device)
        self.config = config or CategoricalDQNConfig()

        # Online Network & Target Network
        self.online_net = CategoricalQNetwork(
            latent_dim=latent_dim,
            hidden_units=self.config.hidden_units,
            num_atoms=self.config.num_atoms,
            v_min=self.config.v_min,
            v_max=self.config.v_max,
            encoder=encoder,
            freeze_encoder=freeze_encoder,
        ).to(self.device)

        self.target_net = CategoricalQNetwork(
            latent_dim=latent_dim,
            hidden_units=self.config.hidden_units,
            num_atoms=self.config.num_atoms,
            v_min=self.config.v_min,
            v_max=self.config.v_max,
        ).to(self.device)

        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        # Optimizer: Adam with lr=1e-4 (trainable parameters only)
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.online_net.parameters()),
            lr=self.config.learning_rate,
        )

        # Replay Buffer
        self.buffer = C51ReplayBuffer(
            capacity=self.config.buffer_capacity,
        )

        # Hyperparameters
        self.gamma = self.config.gamma
        self.tau = self.config.tau
        self.num_atoms = self.config.num_atoms
        self.v_min = self.config.v_min
        self.v_max = self.config.v_max
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        self.support = self.online_net.support

        self.epsilon = self.config.epsilon_start
        self.epsilon_start = self.config.epsilon_start
        self.epsilon_end = self.config.epsilon_end
        self.epsilon_decay_steps = self.config.epsilon_decay_steps

        self.total_steps = 0
        self.update_count = 0

    def freeze_encoder(self) -> None:
        """Freeze shared encoder parameters."""
        self.online_net.freeze_encoder()
        self.target_net.freeze_encoder()
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.online_net.parameters()),
            lr=self.config.learning_rate,
        )

    def unfreeze_encoder(self) -> None:
        """Unfreeze shared encoder parameters."""
        self.online_net.unfreeze_encoder()
        self.target_net.unfreeze_encoder()
        self.optimizer = optim.Adam(
            self.online_net.parameters(),
            lr=self.config.learning_rate,
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
        """Select a strictly legal action via masked expected Q-values.

        Args:
            state: Game state dict, StructuredLudoState, or spatial tensor.
            action_mask: Binary legality mask of shape (4,).
            context: Optional context tensor (12,) if state is spatial tensor.
            deterministic: If True, executes pure greedy argmax without epsilon exploration.

        Returns:
            Selected legal action index [0..3].
        """
        spatial_t, context_t = self._prepare_inputs(state, context=context)
        if isinstance(action_mask, np.ndarray):
            mask_t = torch.from_numpy(action_mask).float().to(self.device)
        else:
            mask_t = action_mask.float().to(self.device)

        eps = 0.0 if deterministic else self.epsilon
        return self.online_net.select_action(
            spatial=spatial_t,
            context=context_t,
            action_mask=mask_t,
            deterministic=deterministic,
            epsilon=eps,
        )

    def project_distribution(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        p_next: torch.Tensor,
    ) -> torch.Tensor:
        """Categorical Bellman projection onto the discrete support [V_min, V_max].

        Formula:
            T z_j = clamp(r + (1 - d) * gamma * z_j, V_min, V_max)
            b_j = (T z_j - V_min) / delta_z
            m_l += p_next * (u - b_j)
            m_u += p_next * (b_j - l)

        Args:
            rewards: Batch tensor of shape (B,).
            dones: Batch tensor of shape (B,).
            p_next: Target atom probabilities for best next action of shape (B, num_atoms).

        Returns:
            Projected target distribution m of shape (B, num_atoms).
        """
        b_size = rewards.size(0)
        gamma_term = (1.0 - dones).unsqueeze(1) * self.gamma
        t_z = rewards.unsqueeze(1) + gamma_term * self.support.unsqueeze(0)
        t_z = torch.clamp(t_z, self.v_min, self.v_max)

        # Coordinate projection: b_j in [0, N-1]
        b_j = (t_z - self.v_min) / self.delta_z
        l = b_j.floor().long().clamp(0, self.num_atoms - 1)
        u = b_j.ceil().long().clamp(0, self.num_atoms - 1)

        # Interpolation weights
        m_l = p_next * (u.float() - b_j)
        m_u = p_next * (b_j - l.float())

        # Exact atom landing case (l == u)
        eq_mask = (l == u)
        m_l = torch.where(eq_mask, p_next, m_l)
        m_u = torch.where(eq_mask, torch.zeros_like(m_u), m_u)

        # Accumulate projected target distribution m
        target_m = torch.zeros(b_size, self.num_atoms, device=self.device)
        target_m.scatter_add_(1, l, m_l)
        target_m.scatter_add_(1, u, m_u)

        return target_m

    def compute_target_distribution(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        next_states: torch.Tensor,
        next_contexts: Optional[torch.Tensor],
        next_action_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the projected target distribution for a batch of transitions."""
        b_size = rewards.size(0)
        with torch.no_grad():
            next_q = self.online_net.get_q_values(next_states, next_contexts)
            masked_next_q = mask_q_values(next_q, next_action_masks)
            best_next_actions = torch.argmax(masked_next_q, dim=-1)  # (B,)

            # Target network atom probabilities for best next actions
            next_probs, _ = self.target_net(next_states, next_contexts)  # (B, A, N)
            best_actions_exp = best_next_actions.view(b_size, 1, 1).expand(b_size, 1, self.num_atoms)
            p_next = next_probs.gather(1, best_actions_exp).squeeze(1)  # (B, N)

            target_m = self.project_distribution(rewards, dones, p_next)
        return target_m

    def train_step(self, batch_size: int = 64) -> Optional[Dict[str, float]]:
        """Execute a single Categorical DQN distributional Bellman projection update."""
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
        next_action_masks = batch["next_action_masks"]

        b_size = states.size(0)

        # -------------------------------------------------------------
        # 1. Target Distribution via Categorical Bellman Projection
        # -------------------------------------------------------------
        target_m = self.compute_target_distribution(
            rewards=rewards,
            dones=dones,
            next_states=next_states,
            next_contexts=next_contexts,
            next_action_masks=next_action_masks,
        )

        # -------------------------------------------------------------
        # 3. Online Network Evaluation and Cross-Entropy Loss
        # -------------------------------------------------------------
        probs, q_values = self.online_net(states, contexts)  # (B, A, N)
        actions_exp = actions.view(b_size, 1, 1).expand(b_size, 1, self.num_atoms)
        p_eval = probs.gather(1, actions_exp).squeeze(1)  # (B, N)

        # Cross-entropy loss: - sum_i target_m_i * log(p_eval_i)
        log_p = torch.log(torch.clamp(p_eval, min=1e-8))
        loss = -(target_m * log_p).sum(dim=-1).mean()

        # Gradient step
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # -------------------------------------------------------------
        # 4. Soft Target Update & Epsilon Annealing
        # -------------------------------------------------------------
        for target_p, online_p in zip(self.target_net.parameters(), self.online_net.parameters()):
            target_p.data.copy_(self.tau * online_p.data + (1.0 - self.tau) * target_p.data)

        self.total_steps += 1
        self.update_count += 1
        self._decay_epsilon()

        return {
            "loss": float(loss.item()),
            "mean_q": float(q_values.mean().item()),
            "epsilon": float(self.epsilon),
            "updates": self.update_count,
        }

    def _decay_epsilon(self) -> None:
        """Linearly decay exploration rate from epsilon_start to epsilon_end."""
        fraction = min(1.0, float(self.total_steps) / float(self.epsilon_decay_steps))
        self.epsilon = self.epsilon_start - fraction * (self.epsilon_start - self.epsilon_end)

    def save_checkpoint(self, path: Path) -> None:
        """Save model parameters and optimizer state to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "online_net": self.online_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "update_count": self.update_count,
            "epsilon": self.epsilon,
        }, path)

    def load_checkpoint(self, path: Path) -> None:
        """Load model parameters and optimizer state from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(checkpoint["online_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.total_steps = checkpoint.get("total_steps", 0)
        self.update_count = checkpoint.get("update_count", 0)
        self.epsilon = checkpoint.get("epsilon", self.epsilon_end)
