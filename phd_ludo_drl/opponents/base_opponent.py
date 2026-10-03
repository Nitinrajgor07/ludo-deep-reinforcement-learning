"""Abstract base class for benchmark evaluation opponents."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Sequence, Union
import numpy as np


class BaseOpponent(ABC):
    """Abstract Base Class for evaluation opponents (Random, Expert)."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def select_action(
        self,
        game_state: Dict[str, Any],
        legal_actions: Sequence[int],
    ) -> int:
        """Select an action from available legal actions.

        Args:
            game_state: Authoritative board and game state dict.
            legal_actions: Non-empty sequence of legal token indices [0..3].

        Returns:
            Chosen action index.
        """
        pass
