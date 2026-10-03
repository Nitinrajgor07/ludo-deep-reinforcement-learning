"""Token representation and state tracking for Ludo."""

from dataclasses import dataclass
from typing import Optional
from phd_ludo_drl.environment.constants import (
    TokenState,
    PLAYER_START_SQUARES,
    TRACK_LENGTH,
    TRACK_STEPS,
    GOAL_STEP,
)


@dataclass
class Token:
    """Represents a single player token in Ludo."""
    player_id: int
    token_id: int
    state: TokenState = TokenState.YARD
    step_count: int = -1  # -1 in yard, 0..50 on track, 51..55 in home column, 56 finished

    def reset(self) -> None:
        """Reset token to starting yard state."""
        self.state = TokenState.YARD
        self.step_count = -1

    @property
    def is_in_yard(self) -> bool:
        return self.state == TokenState.YARD

    @property
    def is_on_track(self) -> bool:
        return self.state == TokenState.TRACK

    @property
    def is_in_home(self) -> bool:
        return self.state == TokenState.HOME

    @property
    def is_finished(self) -> bool:
        return self.state == TokenState.FINISHED

    @property
    def track_square(self) -> Optional[int]:
        """Global track square index (0..51) if currently on track, else None."""
        if self.state != TokenState.TRACK:
            return None
        start = PLAYER_START_SQUARES[self.player_id]
        return (start + self.step_count) % TRACK_LENGTH

    @property
    def home_index(self) -> Optional[int]:
        """Home column index (0..4) if in home path, else None."""
        if self.state != TokenState.HOME:
            return None
        return self.step_count - TRACK_STEPS

    def copy(self) -> "Token":
        """Return a detached clone of this token."""
        return Token(
            player_id=self.player_id,
            token_id=self.token_id,
            state=self.state,
            step_count=self.step_count,
        )
