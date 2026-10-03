"""Prioritized Experience Replay (PER) buffer for Double Dueling DQN."""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch


class PrioritizedReplayBuffer:
    """Proportional Prioritized Experience Replay buffer (Schaul et al., 2015).

    Stores transitions with priority p_i = (|delta_i| + epsilon)^alpha.
    Computes importance sampling weights w_i = (N * P(i))^(-beta) / max_w.
    """

    def __init__(
        self,
        capacity: int = 50000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        epsilon: float = 1e-5,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.epsilon = epsilon

        self.pos = 0
        self.size = 0

        # Preallocated buffers for spatial states (4, 52), contexts (12), and scalars
        self.states = np.zeros((capacity, 4, 52), dtype=np.float32)
        self.contexts = np.zeros((capacity, 12), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, 4, 52), dtype=np.float32)
        self.next_contexts = np.zeros((capacity, 12), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.action_masks = np.zeros((capacity, 4), dtype=np.float32)
        self.next_action_masks = np.zeros((capacity, 4), dtype=np.float32)

        # Proportional priorities array
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0

    def __len__(self) -> int:
        return self.size

    def push(
        self,
        state: np.ndarray,
        context: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        next_context: np.ndarray,
        done: bool,
        action_mask: np.ndarray,
        next_action_mask: np.ndarray,
    ) -> None:
        """Insert a transition with maximum current priority."""
        idx = self.pos

        self.states[idx] = state
        self.contexts[idx] = context
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.next_contexts[idx] = next_context
        self.dones[idx] = float(done)
        self.action_masks[idx] = action_mask
        self.next_action_masks[idx] = next_action_mask

        # Assign max priority so new transitions are sampled at least once
        self.priorities[idx] = self.max_priority

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        beta: Optional[float] = None,
    ) -> Tuple[Dict[str, torch.Tensor], np.ndarray, torch.Tensor]:
        """Sample a batch of transitions proportional to priorities.

        Args:
            batch_size: Number of transitions to sample.
            beta: Current importance sampling exponent (defaults to self.beta).

        Returns:
            - batch dict containing PyTorch tensors
            - indices array of sampled transitions
            - weights tensor of normalized importance sampling weights
        """
        if self.size < batch_size:
            raise ValueError(f"Buffer has {self.size} samples, less than batch size {batch_size}")

        b = beta if beta is not None else self.beta

        # Compute sampling probabilities
        prios = self.priorities[: self.size]
        probs = prios ** self.alpha
        prob_sum = probs.sum()
        if prob_sum == 0.0 or not np.isfinite(prob_sum):
            probs = np.ones(self.size, dtype=np.float32) / self.size
        else:
            probs = probs / prob_sum

        indices = np.random.choice(self.size, batch_size, p=probs, replace=False)

        # Compute Importance Sampling weights
        weights = (self.size * probs[indices]) ** (-b)
        weights = weights / weights.max()  # normalize so max weight is 1.0

        batch = {
            "states": torch.from_numpy(self.states[indices]).float(),
            "contexts": torch.from_numpy(self.contexts[indices]).float(),
            "actions": torch.from_numpy(self.actions[indices]).long(),
            "rewards": torch.from_numpy(self.rewards[indices]).float(),
            "next_states": torch.from_numpy(self.next_states[indices]).float(),
            "next_contexts": torch.from_numpy(self.next_contexts[indices]).float(),
            "dones": torch.from_numpy(self.dones[indices]).float(),
            "action_masks": torch.from_numpy(self.action_masks[indices]).float(),
            "next_action_masks": torch.from_numpy(self.next_action_masks[indices]).float(),
        }

        weights_tensor = torch.from_numpy(weights.astype(np.float32)).float()
        return batch, indices, weights_tensor

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """Update priorities of sampled transitions based on absolute TD error."""
        abs_errors = np.abs(td_errors) + self.epsilon
        for idx, err in zip(indices, abs_errors):
            self.priorities[idx] = float(err)
            self.max_priority = max(self.max_priority, float(err))

    def anneal_beta(self, step: int, total_steps: int) -> float:
        """Linearly anneal beta from beta_start to beta_end."""
        fraction = min(1.0, float(step) / float(total_steps))
        self.beta = self.beta_start + fraction * (self.beta_end - self.beta_start)
        return self.beta
