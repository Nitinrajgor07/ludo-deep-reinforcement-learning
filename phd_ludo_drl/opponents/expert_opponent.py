"""Heuristic Expert Opponent implementing Alhajry et al. (2012) priority strategy.

Priority Ordering (IEEE Paper Section III-B):
1. Prefer Release: If dice == 6 and any token is in yard, release it.
2. Defensive Move: Move a vulnerable piece to a safe square (or out of threat range).
3. Aggressive (Capture): Execute a move that captures an opponent token.
4. Fast (Progress): Advance the most-progressed token closest to the goal.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from phd_ludo_drl.environment.constants import (
    TRACK_LENGTH,
    SAFE_SQUARES,
    PLAYER_START_SQUARES,
    TokenState,
)
from phd_ludo_drl.environment.token import Token
from phd_ludo_drl.environment.board import LudoBoard
from phd_ludo_drl.opponents.base_opponent import BaseOpponent


class ExpertOpponent(BaseOpponent):
    """Authoritative Heuristic Expert Opponent reproducing published baseline strategy."""

    def __init__(self, name: str = "ExpertOpponent", seed: Optional[int] = None):
        super().__init__(name=name)
        self.board = LudoBoard()
        self.rng = np.random.default_rng(seed)

    def seed(self, seed: Optional[int] = None) -> None:
        """Seed the random number generator for tie-breaking."""
        self.rng = np.random.default_rng(seed)

    def is_vulnerable(self, token: Token, all_tokens: List[Token]) -> bool:
        """Check if friendly token on track is vulnerable to capture by an opponent."""
        if not token.is_on_track or token.track_square is None:
            return False

        if self.board.is_safe_square(token.track_square):
            return False

        sq = token.track_square
        opponent_id = 1 - token.player_id

        # Check if any opponent token is on track within 1..6 squares behind
        for other in all_tokens:
            if other.player_id == opponent_id and other.is_on_track and other.track_square is not None:
                dist = (sq - other.track_square) % TRACK_LENGTH
                if 1 <= dist <= 6:
                    return True
        return False

    def does_move_capture(self, token: Token, dice: int, all_tokens: List[Token]) -> bool:
        """Check if moving this token lands on and captures an opponent token."""
        if token.is_in_yard:
            # Releasing to start square might capture
            start_sq = PLAYER_START_SQUARES[token.player_id]
            if self.board.is_safe_square(start_sq):
                return False
            dest_sq = start_sq
        elif token.is_on_track:
            new_step = token.step_count + dice
            dest_sq = self.board.get_track_square_for_step(token.player_id, new_step)
            if dest_sq is None or self.board.is_safe_square(dest_sq):
                return False
        else:
            return False

        opponent_id = 1 - token.player_id
        for other in all_tokens:
            if other.player_id == opponent_id and other.is_on_track and other.track_square == dest_sq:
                return True
        return False

    def does_move_defend(self, token: Token, dice: int, all_tokens: List[Token]) -> bool:
        """Check if move transitions a currently vulnerable token into safety."""
        if not self.is_vulnerable(token, all_tokens):
            return False

        new_step = token.step_count + dice
        dest_sq = self.board.get_track_square_for_step(token.player_id, new_step)

        # Entering safe square or entering home column guarantees safety
        if dest_sq is None and new_step >= 51:
            return True
        if dest_sq is not None and self.board.is_safe_square(dest_sq):
            return True

        return False

    def select_action(
        self,
        game_state: Dict[str, Any],
        legal_actions: Sequence[int],
    ) -> int:
        """Select action following strict 4-level research priority order.

        Args:
            game_state: Game state dictionary.
            legal_actions: Sequence of legal action indices.

        Returns:
            Selected action index.

        Raises:
            ValueError: If legal_actions is empty.
        """
        if not legal_actions:
            raise ValueError("ExpertOpponent cannot select an action from an empty legal_actions list.")

        if len(legal_actions) == 1:
            return legal_actions[0]

        dice = game_state["dice_value"]
        current_player = game_state["current_player"]
        tokens: List[Token] = game_state["tokens"]
        player_tokens = {t.token_id: t for t in tokens if t.player_id == current_player}

        # 1. Prefer Release: If dice == 6 and any yard token can be released
        if dice == 6:
            release_actions = [
                act for act in legal_actions
                if player_tokens[act].is_in_yard
            ]
            if release_actions:
                return release_actions[0]

        # 2. Defensive Move: Move vulnerable token to safety
        defensive_actions = [
            act for act in legal_actions
            if self.does_move_defend(player_tokens[act], dice, tokens)
        ]
        if defensive_actions:
            # Pick the most progressed among defensive moves
            return max(defensive_actions, key=lambda act: player_tokens[act].step_count)

        # 3. Aggressive (Capture): Execute capture if available
        capture_actions = [
            act for act in legal_actions
            if self.does_move_capture(player_tokens[act], dice, tokens)
        ]
        if capture_actions:
            return max(capture_actions, key=lambda act: player_tokens[act].step_count)

        # 4. Fast (Progress): Advance the most-progressed token
        return max(legal_actions, key=lambda act: player_tokens[act].step_count)
