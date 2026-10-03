"""Random baseline opponent selecting uniformly among legal actions."""

from typing import Any, Dict, Optional, Sequence
import numpy as np

from phd_ludo_drl.opponents.base_opponent import BaseOpponent


class RandomOpponent(BaseOpponent):
    """Opponent that selects uniformly at random among legal actions."""

    def __init__(self, name: str = "RandomOpponent", seed: Optional[int] = None):
        super().__init__(name=name)
        self.rng = np.random.default_rng(seed)

    def seed(self, seed: Optional[int] = None) -> None:
        """Seed the random number generator."""
        self.rng = np.random.default_rng(seed)

    def select_action(
        self,
        game_state: Dict[str, Any],
        legal_actions: Sequence[int],
    ) -> int:
        """Select uniformly at random among legal actions.

        Args:
            game_state: State dictionary (unused by random agent).
            legal_actions: Sequence of legal action indices.

        Returns:
            Selected legal action index.

        Raises:
            ValueError: If legal_actions is empty.
        """
        if not legal_actions:
            raise ValueError("RandomOpponent cannot select an action from an empty legal_actions list.")

        return int(self.rng.choice(legal_actions))
