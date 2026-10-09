"""Rollout buffer for PPO trajectory storage and Generalized Advantage Estimation (GAE)."""

from typing import Dict, Generator, Optional
import numpy as np
import torch


class PPORolloutBuffer:
    """Fixed-capacity rollout buffer for storing on-policy transitions and computing GAE.

    Matches published IEEE PPO specification:
    - Stores structured spatial board tensors (4, 52) and context vectors (12).
    - Preserves action masks (4,) per decision point.
    - Implements Generalized Advantage Estimation (GAE) with discount gamma and trace lambda.
    - Generates randomized mini-batches for multiple PPO optimization epochs.
    """

    def __init__(
        self,
        capacity: int,
        spatial_shape: tuple = (4, 52),
        context_dim: int = 12,
        action_dim: int = 4,
    ):
        self.capacity = capacity
        self.spatial_shape = spatial_shape
        self.context_dim = context_dim
        self.action_dim = action_dim

        self.states = np.zeros((capacity, *spatial_shape), dtype=np.float32)
        self.contexts = np.zeros((capacity, context_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.action_masks = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.log_probs = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

        self.advantages = np.zeros(capacity, dtype=np.float32)
        self.returns = np.zeros(capacity, dtype=np.float32)

        self.ptr = 0
        self.full = False

    def add(
        self,
        state: np.ndarray,
        context: Optional[np.ndarray],
        action: int,
        action_mask: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ) -> None:
        """Add a transition to the rollout buffer.

        Args:
            state: Spatial board tensor of shape (4, 52).
            context: Context vector of shape (12,) or None (will default to zeros).
            action: Selected action integer.
            action_mask: Valid action mask (4,).
            reward: Step reward.
            value: Value estimate V(s).
            log_prob: Log probability of the selected action under the behavior policy.
            done: Termination / truncation flag.
        """
        if self.ptr >= self.capacity:
            raise IndexError(f"Rollout buffer overflow: capacity {self.capacity} exceeded.")

        self.states[self.ptr] = np.asarray(state, dtype=np.float32)
        if context is not None:
            self.contexts[self.ptr] = np.asarray(context, dtype=np.float32)
        else:
            self.contexts[self.ptr] = np.zeros(self.context_dim, dtype=np.float32)

        self.actions[self.ptr] = int(action)
        self.action_masks[self.ptr] = np.asarray(action_mask, dtype=np.float32)
        self.rewards[self.ptr] = float(reward)
        self.values[self.ptr] = float(value)
        self.log_probs[self.ptr] = float(log_prob)
        self.dones[self.ptr] = 1.0 if done else 0.0

        self.ptr += 1
        if self.ptr == self.capacity:
            self.full = True

    def compute_gae(
        self,
        last_value: float = 0.0,
        last_done: bool = False,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        """Compute Generalized Advantage Estimation (GAE) and returns.

        Formula:
            delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
            A_t = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}
            R_t = A_t + V(s_t)

        Args:
            last_value: Estimated value of the state after the final step V(s_T).
            last_done: Whether the final step was a terminal state.
            gamma: Discount factor (default: 0.99).
            gae_lambda: GAE trace factor (default: 0.95).
        """
        last_gae = 0.0
        n_steps = self.ptr

        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_non_terminal = 1.0 - float(self.dones[t] if self.dones[t] else last_done)
                next_value = float(last_value)
            else:
                next_non_terminal = 1.0 - self.dones[t]
                next_value = self.values[t + 1]

            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        self.returns[:n_steps] = self.advantages[:n_steps] + self.values[:n_steps]

    def get_mini_batches(
        self,
        mini_batch_size: int,
        device: torch.device,
        normalize_advantages: bool = True,
    ) -> Generator[Dict[str, torch.Tensor], None, None]:
        """Yield randomized mini-batches for PPO update epochs.

        Args:
            mini_batch_size: Size of each mini-batch.
            device: PyTorch device to place mini-batch tensors on.
            normalize_advantages: Whether to normalize advantages over the whole rollout.

        Yields:
            Dict containing batch tensors on target device.
        """
        n_steps = self.ptr
        if n_steps == 0:
            return

        advantages = self.advantages[:n_steps].copy()
        if normalize_advantages and n_steps > 1:
            adv_mean = np.mean(advantages)
            adv_std = np.std(advantages) + 1e-8
            advantages = (advantages - adv_mean) / adv_std

        indices = np.random.permutation(n_steps)
        for start_idx in range(0, n_steps, mini_batch_size):
            batch_idx = indices[start_idx : start_idx + mini_batch_size]

            yield {
                "states": torch.from_numpy(self.states[batch_idx]).to(device),
                "contexts": torch.from_numpy(self.contexts[batch_idx]).to(device),
                "actions": torch.from_numpy(self.actions[batch_idx]).to(device),
                "action_masks": torch.from_numpy(self.action_masks[batch_idx]).to(device),
                "old_log_probs": torch.from_numpy(self.log_probs[batch_idx]).to(device),
                "old_values": torch.from_numpy(self.values[batch_idx]).to(device),
                "returns": torch.from_numpy(self.returns[batch_idx]).to(device),
                "advantages": torch.from_numpy(advantages[batch_idx]).to(device),
            }

    def reset(self) -> None:
        """Reset the buffer pointer."""
        self.ptr = 0
        self.full = False

    def __len__(self) -> int:
        return self.ptr
