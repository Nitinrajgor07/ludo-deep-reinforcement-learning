"""Abstract base agent interface ensuring uniform evaluation and training protocol."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np
import torch


class BaseAgent(ABC):
    """Abstract Base Class for all Ludo RL agents (DQN, PPO, Model 3, Model 4)."""

    def __init__(self, name: str, latent_dim: int = 128, device: Optional[torch.device] = None):
        self.name = name
        self.latent_dim = latent_dim
        self.device = device or torch.device("cpu")

    @abstractmethod
    def select_action(
        self,
        state: Union[np.ndarray, torch.Tensor],
        action_mask: Union[np.ndarray, torch.Tensor],
        deterministic: bool = False,
    ) -> int:
        """Select a legal action (0..3) given state observation and binary action mask.

        Args:
            state: Encoded latent state or observation tensor.
            action_mask: Binary mask of shape (4,) where 1 denotes legal, 0 illegal.
            deterministic: True for greedy/mode action, False for exploration/sampling.

        Returns:
            Selected token action index (0, 1, 2, or 3).
        """
        pass

    @abstractmethod
    def save_checkpoint(self, path: Path) -> None:
        """Save model weights and training state to file."""
        pass

    @abstractmethod
    def load_checkpoint(self, path: Path) -> None:
        """Load model weights and training state from file."""
        pass
