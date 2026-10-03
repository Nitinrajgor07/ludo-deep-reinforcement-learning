"""Double Dueling DQN Agent with Prioritized Experience Replay and Action Masking."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from phd_ludo_drl.config.config_manager import DQNConfig
from phd_ludo_drl.agents.base_agent import BaseAgent
from phd_ludo_drl.agents.dqn.dqn_network import DuelingQNetwork
from phd_ludo_drl.agents.dqn.replay_buffer import PrioritizedReplayBuffer
from phd_ludo_drl.environment.action_mask import mask_q_values
from phd_ludo_drl.encoding.structured_representation import build_structured_representation


class DQNAgent(BaseAgent):
    """Authoritative Double Dueling DQN Agent matching published IEEE research."""

    def __init__(
        self,
        config: Optional[DQNConfig] = None,
        latent_dim: int = 128,
        device: Optional[torch.device] = None,
        name: str = "DoubleDuelingDQN",
    ):
        super().__init__(name=name, latent_dim=latent_dim, device=device)
        self.config = config or DQNConfig()

        # Online network and Target network
        self.online_net = DuelingQNetwork(
            latent_dim=latent_dim,
            hidden_units=self.config.hidden_units,
        ).to(self.device)

        self.target_net = DuelingQNetwork(
            latent_dim=latent_dim,
            hidden_units=self.config.hidden_units,
        ).to(self.device)

        # Initialize target network with identical weights
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        # Optimizer: Adam with published lr=1e-4
        self.optimizer = optim.Adam(
            self.online_net.parameters(),
            lr=self.config.learning_rate,
        )

        # Prioritized Replay Buffer
        self.buffer = PrioritizedReplayBuffer(
            capacity=self.config.buffer_capacity,
            alpha=self.config.per_alpha,
            beta_start=self.config.per_beta_start,
            beta_end=self.config.per_beta_end,
        )

        # Hyperparameters
        self.gamma = self.config.gamma
        self.tau = self.config.tau
        self.epsilon = self.config.epsilon_start
        self.epsilon_start = self.config.epsilon_start
        self.epsilon_end = self.config.epsilon_end
        self.epsilon_decay_steps = self.config.epsilon_decay_steps
        self.total_steps = 0

    def select_action(
        self,
        state: Union[Dict[str, Any], np.ndarray, torch.Tensor],
        action_mask: Union[np.ndarray, torch.Tensor],
        deterministic: bool = False,
    ) -> int:
        """Select action via masked epsilon-greedy policy.

        Args:
            state: Environment state dict or spatial tensor.
            action_mask: Binary mask of shape (4,).
            deterministic: If True, executes pure greedy action without exploration.

        Returns:
            Selected legal action index [0..3].
        """
        if isinstance(action_mask, torch.Tensor):
            mask_arr = action_mask.detach().cpu().numpy()
            mask_t = action_mask.to(self.device).float()
        else:
            mask_arr = np.asarray(action_mask, dtype=np.float32)
            mask_t = torch.from_numpy(mask_arr).to(self.device).float()

        legal_actions = [i for i, m in enumerate(mask_arr) if m == 1.0]
        if not legal_actions:
            raise ValueError("DQNAgent cannot select action when legal_actions is empty.")

        # Epsilon-greedy exploration (only from legal actions)
        if not deterministic and np.random.rand() < self.epsilon:
            return int(np.random.choice(legal_actions))

        # Extract spatial tensor and context if state is a state_dict
        if isinstance(state, dict):
            struct_state = build_structured_representation(state)
            spatial_t = torch.from_numpy(struct_state.spatial_tensor).unsqueeze(0).to(self.device)
            context_t = torch.from_numpy(struct_state.context_vector).unsqueeze(0).to(self.device)
        else:
            if isinstance(state, np.ndarray):
                spatial_t = torch.from_numpy(state).float().to(self.device)
            else:
                spatial_t = state.to(self.device)
            if spatial_t.dim() == 2:
                spatial_t = spatial_t.unsqueeze(0)
            context_t = None

        with torch.no_grad():
            q_values = self.online_net(spatial_t, context_t).squeeze(0)  # (4,)
            masked_q = mask_q_values(q_values, mask_t)
            best_action = int(torch.argmax(masked_q).item())

        return best_action

    def train_step(self, batch_size: int = 64) -> Optional[Dict[str, float]]:
        """Perform a single Double DQN optimization step using Prioritized Replay."""
        if len(self.buffer) < batch_size:
            return None

        # Sample prioritized batch with current beta
        batch, indices, weights = self.buffer.sample(batch_size)

        states = batch["states"].to(self.device)
        contexts = batch["contexts"].to(self.device)
        actions = batch["actions"].to(self.device).unsqueeze(1)  # (B, 1)
        rewards = batch["rewards"].to(self.device).unsqueeze(1)  # (B, 1)
        next_states = batch["next_states"].to(self.device)
        next_contexts = batch["next_contexts"].to(self.device)
        dones = batch["dones"].to(self.device).unsqueeze(1)      # (B, 1)
        next_action_masks = batch["next_action_masks"].to(self.device)
        weights = weights.to(self.device).unsqueeze(1)           # (B, 1)

        # 1. Current Q-values: Q_online(s, a)
        q_eval = self.online_net(states, contexts).gather(1, actions)  # (B, 1)

        # 2. Double DQN Target Calculation:
        # a* = argmax_a' Q_online(s', a'; theta) with next action mask
        with torch.no_grad():
            next_q_online = self.online_net(next_states, next_contexts)
            masked_next_q_online = mask_q_values(next_q_online, next_action_masks)
            best_next_actions = torch.argmax(masked_next_q_online, dim=1, keepdim=True)  # (B, 1)

            # Evaluate Q_target(s', a*; theta^-)
            next_q_target = self.target_net(next_states, next_contexts)
            q_target_val = next_q_target.gather(1, best_next_actions)

            # Bellman equation
            q_target = rewards + (1.0 - dones) * self.gamma * q_target_val

        # 3. TD Error and Prioritized Loss
        td_errors = (q_eval - q_target).squeeze(1)  # (B,)
        loss_fn = nn.SmoothL1Loss(reduction="none")
        elementwise_loss = loss_fn(q_eval, q_target)
        weighted_loss = (weights * elementwise_loss).mean()

        # 4. Gradient Step
        self.optimizer.zero_grad()
        weighted_loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # 5. Update PER priorities
        self.buffer.update_priorities(indices, td_errors.detach().cpu().numpy())

        # 6. Soft target network update: theta_target = tau * theta + (1 - tau) * theta_target
        self._soft_update(self.tau)

        self.total_steps += 1
        self._decay_epsilon()
        self.buffer.anneal_beta(self.total_steps, self.epsilon_decay_steps)

        return {
            "loss": float(weighted_loss.item()),
            "mean_q": float(q_eval.mean().item()),
            "mean_td_error": float(td_errors.abs().mean().item()),
            "epsilon": float(self.epsilon),
            "beta": float(self.buffer.beta),
        }

    def _soft_update(self, tau: float) -> None:
        """Polyak soft update of target network parameters."""
        for target_param, online_param in zip(self.target_net.parameters(), self.online_net.parameters()):
            target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)

    def _decay_epsilon(self) -> None:
        """Linearly decay exploration rate from epsilon_start to epsilon_end."""
        fraction = min(1.0, float(self.total_steps) / float(self.epsilon_decay_steps))
        self.epsilon = self.epsilon_start - fraction * (self.epsilon_start - self.epsilon_end)

    def save_checkpoint(self, path: Path) -> None:
        """Save model weights and optimizer state to file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "online_net": self.online_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "epsilon": self.epsilon,
            "beta": self.buffer.beta,
        }, path)

    def load_checkpoint(self, path: Path) -> None:
        """Load model weights and optimizer state from file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(checkpoint["online_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.total_steps = checkpoint.get("total_steps", 0)
        self.epsilon = checkpoint.get("epsilon", self.epsilon_end)
        if "beta" in checkpoint:
            self.buffer.beta = checkpoint["beta"]
