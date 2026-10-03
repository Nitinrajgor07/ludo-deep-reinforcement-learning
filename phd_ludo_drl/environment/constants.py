"""Constants and Board Specifications for 2-Player Ludo Environment.

Aligned with Alhajry et al. (2012) and IEEE published research foundation.
"""

from enum import IntEnum, auto
from typing import Final, List, Tuple


class TokenState(IntEnum):
    """Lifecycle state of an individual token."""
    YARD = 0       # Token is in the starting yard
    TRACK = 1      # Token is on the common 52-square circular track
    HOME = 2       # Token is in the player's 5-square home column
    FINISHED = 3   # Token has reached the goal


# Board Dimensions
TRACK_LENGTH: Final[int] = 52
HOME_COLUMN_LENGTH: Final[int] = 5
TOTAL_PATH_LENGTH: Final[int] = 56  # 0..50 on track (51 steps), 51..55 in home (5 steps), 56 is Goal
NUM_PLAYERS: Final[int] = 2
TOKENS_PER_PLAYER: Final[int] = 4

# Player Starting Track Squares
# Player 0 enters track at 0; Player 1 enters track at 26 (opposite side of 52-square track)
PLAYER_START_SQUARES: Final[Tuple[int, int]] = (0, 26)

# Standard Safe Squares on 52-square track (cannot capture or be captured)
# Indices: 0, 8, 13, 21, 26, 34, 39, 47
SAFE_SQUARES: Final[Tuple[int, ...]] = (0, 8, 13, 21, 26, 34, 39, 47)

# Step Counts along local progression (0 to 56):
# Step 0: Token lands on start square upon release
# Steps 0..50: 51 track squares traversed
# Steps 51..55: 5 home column squares (index 0..4)
# Step 56: Goal square (FINISHED)
TRACK_STEPS: Final[int] = 51  # Number of steps spent on common track (0 to 50)
GOAL_STEP: Final[int] = 56

# Action Space
ACTION_SPACE_SIZE: Final[int] = TOKENS_PER_PLAYER  # 4 discrete actions: 0, 1, 2, 3
