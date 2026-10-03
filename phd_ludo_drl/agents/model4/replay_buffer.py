"""Replay buffer for Categorical DQN (C51 / Model 4)."""

from typing import Dict, Optional
import numpy as np
import torch


class C51ReplayBuffer:
    """Experience replay buffer for storing off-policy transitions with companion action masks.

    Stores:
    - Spatial states: (capacity, 4, 52)
    - Context vectors: (capacity, 12)
    - Actions: (capacity,)
    - Rewards: (capacity,)
    - Next spatial states: (capacity, 4, 52)
    - Next context vectors: (capacity, 12)
    - Dones: (capacity,)
    - Action masks: (capacity, 4)
    - Next action masks: (capacity, 4)
    """

    def __init__(
        self,
        capacity: int = 50000,
        spatial_shape: tuple = (4, 52),
        context_dim: int = 12,
        action_dim: int = 4,
    ):
        self.capacity = capacity
        self.spatial_shape = spatial_shape
        self.context_dim = context_dim
        self.action_dim = action_dim

        self.pos = 0
        self.size = 0

        self.states = np.zeros((capacity, *spatial_shape), dtype=np.float32)
        self.contexts = np.zeros((capacity, context_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, *spatial_shape), dtype=np.float32)
        self.next_contexts = np.zeros((capacity, context_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.action_masks = np.zeros((capacity, action_dim), dtype=np.float32)
        self.next_action_masks = np.zeros((capacity, action_dim), dtype=np.float32)

    def __len__(self) -> int:
        return self.size

    def push(
        self,
        state: np.ndarray,
        context: Optional[np.ndarray],
        action: int,
        reward: float,
        next_state: np.ndarray,
        next_context: Optional[np.ndarray],
        done: bool,
        action_mask: np.ndarray,
        next_action_mask: np.ndarray,
    ) -> None:
        """Store a transition into the circular buffer."""
        idx = self.pos

        self.states[idx] = np.asarray(state, dtype=np.float32)
        self.contexts[idx] = np.asarray(context, dtype=np.float32) if context is not None else np.zeros(self.context_dim, dtype=np.float32)
        self.actions[idx] = int(action)
        self.rewards[idx] = float(reward)
        self.next_states[idx] = np.asarray(next_state, dtype=np.float32)
        self.next_contexts[idx] = np.asarray(next_context, dtype=np.float32) if next_context is not None else np.zeros(self.context_dim, dtype=np.float32)
        self.dones[idx] = 1.0 if done else 0.0
        self.action_masks[idx] = np.asarray(action_mask, dtype=np.float32)
        self.next_action_masks[idx] = np.asarray(next_action_mask, dtype=np.float32)

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """Sample a uniform random batch of transitions."""
        if self.size < batch_size:
            raise ValueError(f"Cannot sample {batch_size} from buffer of size {self.size}")

        indices = np.random.choice(self.size, size=batch_size, replace=False)

        return {
            "states": torch.from_numpy(self.states[indices]).to(device),
            "contexts": torch.from_numpy(self.contexts[indices]).to(device),
            "actions": torch.from_numpy(self.actions[indices]).to(device),
            "rewards": torch.from_numpy(self.rewards[indices]).to(device),
            "next_states": torch.from_numpy(self.next_states[indices]).to(device),
            "next_contexts": torch.from_numpy(self.next_contexts[indices]).to(device),
            "dones": torch.from_numpy(self.dones[indices]).to(device),
            "action_masks": torch.from_numpy(self.action_masks[indices]).to(device),
            "next_action_masks": torch.from_numpy(self.next_action_masks[indices]).to(device),
        }

    def clear(self) -> None:
        """Reset the buffer."""
        self.pos = 0
        self.size = 0
