"""Board geometry, coordinate mapping, and spatial query engine for 2-player Ludo."""

from typing import Dict, List, Optional, Tuple
from phd_ludo_drl.environment.constants import (
    TRACK_LENGTH,
    HOME_COLUMN_LENGTH,
    NUM_PLAYERS,
    TOKENS_PER_PLAYER,
    SAFE_SQUARES,
    PLAYER_START_SQUARES,
    TRACK_STEPS,
    GOAL_STEP,
    TokenState,
)
from phd_ludo_drl.environment.token import Token


class LudoBoard:
    """Manages board state queries, blockade detection, and spatial mappings."""

    def __init__(self):
        self.safe_squares = set(SAFE_SQUARES)

    def is_safe_square(self, track_square: int) -> bool:
        """Check if a global track square is safe."""
        return track_square in self.safe_squares

    @staticmethod
    def get_track_square_for_step(player_id: int, step_count: int) -> Optional[int]:
        """Convert a player's local step count to global track square.

        Returns None if the step is outside the common track (in yard, home, or finished).
        """
        if 0 <= step_count < TRACK_STEPS:
            start = PLAYER_START_SQUARES[player_id]
            return (start + step_count) % TRACK_LENGTH
        return None

    @staticmethod
    def get_tokens_at_track_square(tokens: List[Token], track_square: int) -> List[Token]:
        """Return all tokens currently located on a given global track square."""
        return [t for t in tokens if t.is_on_track and t.track_square == track_square]

    @classmethod
    def has_blockade(cls, tokens: List[Token], track_square: int, player_id: int) -> bool:
        """Check if player_id has formed a blockade (2+ tokens) at track_square."""
        count = sum(
            1 for t in tokens
            if t.player_id == player_id and t.is_on_track and t.track_square == track_square
        )
        return count >= 2

    @classmethod
    def is_path_blocked_by_opponent(
        cls,
        tokens: List[Token],
        moving_token: Token,
        dice_value: int,
    ) -> bool:
        """Check if moving the token by dice_value is obstructed by an opponent blockade.

        In Ludo, a blockade (2+ opponent tokens on a single square) acts as an impassable wall.
        No opponent token may land on or jump over a blockade.
        """
        current_step = moving_token.step_count
        opponent_id = 1 - moving_token.player_id

        # If releasing from yard (dice == 6)
        if moving_token.is_in_yard:
            start_sq = PLAYER_START_SQUARES[moving_token.player_id]
            return cls.has_blockade(tokens, start_sq, opponent_id)

        # Check every step the token traverses along the track
        for step in range(current_step + 1, current_step + dice_value + 1):
            if step < TRACK_STEPS:
                sq = cls.get_track_square_for_step(moving_token.player_id, step)
                if sq is not None and cls.has_blockade(tokens, sq, opponent_id):
                    return True
            else:
                # Entered home column; opponent cannot enter or blockade friendly home column
                break

        return False
