"""Transforms raw Ludo state into the frozen authoritative structured representation.

Authoritative Specification: FINAL_EXPERIMENT_PROTOCOL.md Section 3 (Version 1.0.0 FROZEN)

Spatial representation:
- Shape: (4, 52)
- Channel 0: Active-player token occupancy on track squares (0..51).
- Channel 1: Opponent token occupancy on track squares (0..51).
- Channel 2: Safe-square indicators (1.0 at safe tiles, 0.0 elsewhere).
- Channel 3: Blockade indicators (1.0 at tiles occupied by >= 2 tokens of the same color, 0.0 elsewhere).

Context vector:
- Exactly 12 dimensions containing protocol-defined context features in documented order:
  Index 0: Active player tokens in yard [0..4] normalized by 4.
  Index 1: Opponent tokens in yard [0..4] normalized by 4.
  Index 2: Active player tokens in home column [0..4] normalized by 4.
  Index 3: Opponent tokens in home column [0..4] normalized by 4.
  Index 4: Active player tokens in goal [0..4] normalized by 4.
  Index 5: Opponent tokens in goal [0..4] normalized by 4.
  Index 6: Current dice roll [1..6] normalized by 6.
  Index 7: Active player ID one-hot index 0 (1.0 for P0, 0.0 for P1).
  Index 8: Active player ID one-hot index 1 (0.0 for P0, 1.0 for P1).
  Index 9: Turn fraction (turn / 1000.0).
  Index 10: Bonus roll flag (1.0 if rolling after 6, else 0.0).
  Index 11: Capture bonus flag (1.0 if rolling after capture, else 0.0).

Companion Action Mask:
- Binary vector of shape (4,): 1.0 if token action is legal, 0.0 otherwise.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from phd_ludo_drl.environment.constants import (
    TRACK_LENGTH,
    SAFE_SQUARES,
    NUM_PLAYERS,
    TOKENS_PER_PLAYER,
    ACTION_SPACE_SIZE,
    TokenState,
)
from phd_ludo_drl.environment.token import Token

# Invariant Constants
SPATIAL_SHAPE: Tuple[int, int] = (4, TRACK_LENGTH)  # (4, 52)
CONTEXT_DIMENSION: int = 12
ACTION_MASK_DIMENSION: int = ACTION_SPACE_SIZE      # 4

# Spatial Channels
CHANNEL_ACTIVE_OCCUPANCY: int = 0
CHANNEL_OPPONENT_OCCUPANCY: int = 1
CHANNEL_SAFE_SQUARES: int = 2
CHANNEL_BLOCKADES: int = 3

CHANNEL_NAMES: Tuple[str, ...] = (
    "active_token_occupancy",
    "opponent_token_occupancy",
    "safe_square_indicators",
    "blockade_indicators",
)

# Context Vector Index Ordering
CONTEXT_IDX_ACTIVE_YARD: int = 0
CONTEXT_IDX_OPPONENT_YARD: int = 1
CONTEXT_IDX_ACTIVE_HOME: int = 2
CONTEXT_IDX_OPPONENT_HOME: int = 3
CONTEXT_IDX_ACTIVE_GOAL: int = 4
CONTEXT_IDX_OPPONENT_GOAL: int = 5
CONTEXT_IDX_DICE: int = 6
CONTEXT_IDX_PLAYER_0: int = 7
CONTEXT_IDX_PLAYER_1: int = 8
CONTEXT_IDX_TURN_FRACTION: int = 9
CONTEXT_IDX_BONUS_SIX: int = 10
CONTEXT_IDX_BONUS_CAPTURE: int = 11

CONTEXT_FEATURE_NAMES: Tuple[str, ...] = (
    "active_yard_normalized",       # Index 0
    "opponent_yard_normalized",     # Index 1
    "active_home_normalized",       # Index 2
    "opponent_home_normalized",     # Index 3
    "active_goal_normalized",       # Index 4
    "opponent_goal_normalized",     # Index 5
    "dice_roll_normalized",         # Index 6
    "active_player_p0_onehot",      # Index 7
    "active_player_p1_onehot",      # Index 8
    "turn_fraction",                # Index 9
    "bonus_roll_six_flag",          # Index 10
    "bonus_roll_capture_flag",      # Index 11
)


@dataclass(frozen=True)
class StructuredLudoState:
    """Frozen structured spatial state representation.

    Attributes:
        spatial_tensor: np.ndarray of shape (4, 52) and float32 dtype.
        context_vector: np.ndarray of shape (12,) and float32 dtype.
        action_mask: np.ndarray of shape (4,) and float32 dtype.
    """
    spatial_tensor: np.ndarray  # shape (4, 52)
    context_vector: np.ndarray  # shape (12,)
    action_mask: np.ndarray     # shape (4,) companion tensor


def _is_yard_token(t: Any) -> bool:
    if hasattr(t, "is_in_yard"):
        return bool(t.is_in_yard)
    if hasattr(t, "state"):
        return t.state == TokenState.YARD
    if isinstance(t, dict):
        return t.get("state") in (TokenState.YARD, "YARD", 0) or bool(t.get("is_in_yard", False))
    return False


def _is_home_token(t: Any) -> bool:
    if hasattr(t, "is_in_home"):
        return bool(t.is_in_home)
    if hasattr(t, "state"):
        return t.state == TokenState.HOME
    if isinstance(t, dict):
        return t.get("state") in (TokenState.HOME, "HOME", 2) or bool(t.get("is_in_home", False))
    return False


def _is_goal_token(t: Any) -> bool:
    if hasattr(t, "is_finished"):
        return bool(t.is_finished)
    if hasattr(t, "state"):
        return t.state == TokenState.FINISHED
    if isinstance(t, dict):
        return t.get("state") in (TokenState.FINISHED, "FINISHED", 3) or bool(t.get("is_finished", False))
    return False


def _is_track_token(t: Any) -> bool:
    if hasattr(t, "is_on_track"):
        return bool(t.is_on_track)
    if hasattr(t, "state"):
        return t.state == TokenState.TRACK
    if isinstance(t, dict):
        return t.get("state") in (TokenState.TRACK, "TRACK", 1) or bool(t.get("is_on_track", False))
    return False


def _get_track_square(t: Any) -> Optional[int]:
    if hasattr(t, "track_square"):
        return t.track_square
    if isinstance(t, dict):
        return t.get("track_square")
    return None


def _get_player_id(t: Any) -> int:
    if hasattr(t, "player_id"):
        return int(t.player_id)
    if isinstance(t, dict):
        return int(t["player_id"])
    raise ValueError(f"Unable to resolve player_id on token: {t}")


def build_structured_representation(state_dict: Dict[str, Any]) -> StructuredLudoState:
    """Construct the authoritative 4-channel spatial board tensor and 12-D context vector.

    Conforms strictly to FINAL_EXPERIMENT_PROTOCOL.md Section 3:
    1. Spatial Tensor (4, 52):
       - Channel 0: Active-player token occupancy on track squares (0..51).
       - Channel 1: Opponent token occupancy on track squares (0..51).
       - Channel 2: Safe square spatial indicators (1.0 at safe tiles, 0.0 elsewhere).
       - Channel 3: Blockade indicators (1.0 at tiles with >= 2 tokens of the same color).

    2. Context Vector (12,):
       - Index 0: Active player yard count [0..4] / 4.0
       - Index 1: Opponent yard count [0..4] / 4.0
       - Index 2: Active player home column count [0..4] / 4.0
       - Index 3: Opponent home column count [0..4] / 4.0
       - Index 4: Active player goal count [0..4] / 4.0
       - Index 5: Opponent goal count [0..4] / 4.0
       - Index 6: Current dice roll [1..6] / 6.0
       - Index 7: Active player ID one-hot index 0 (1.0 for P0, 0.0 for P1)
       - Index 8: Active player ID one-hot index 1 (0.0 for P0, 1.0 for P1)
       - Index 9: Turn fraction (turn / 1000.0)
       - Index 10: Bonus roll flag (1.0 if rolling after 6, else 0.0)
       - Index 11: Capture bonus flag (1.0 if rolling after capture, else 0.0)

    3. Action Mask (4,):
       - Binary legal actions vector.
    """
    current_player: int = int(state_dict["current_player"])
    opponent_player: int = 1 - current_player
    dice_value: int = int(state_dict.get("dice_value", 0))
    raw_tokens = state_dict.get("tokens", [])
    raw_mask = state_dict.get("action_mask", np.zeros(ACTION_MASK_DIMENSION, dtype=np.float32))
    action_mask = np.asarray(raw_mask, dtype=np.float32)

    # -------------------------------------------------------------
    # 1. Spatial Board Tensor S in R^{4 x 52}
    # -------------------------------------------------------------
    spatial = np.zeros(SPATIAL_SHAPE, dtype=np.float32)

    # Channel 2: Safe square indicators (1.0 at safe tiles, 0.0 elsewhere)
    for sq in SAFE_SQUARES:
        spatial[CHANNEL_SAFE_SQUARES, sq] = 1.0

    # Separate tokens by player
    active_tokens = [t for t in raw_tokens if _get_player_id(t) == current_player]
    opponent_tokens = [t for t in raw_tokens if _get_player_id(t) == opponent_player]

    # Channel 0: Active player token count per track tile (0..51)
    for t in active_tokens:
        if _is_track_token(t):
            sq = _get_track_square(t)
            if sq is not None and 0 <= sq < TRACK_LENGTH:
                spatial[CHANNEL_ACTIVE_OCCUPANCY, sq] += 1.0

    # Channel 1: Opponent token count per track tile (0..51)
    for t in opponent_tokens:
        if _is_track_token(t):
            sq = _get_track_square(t)
            if sq is not None and 0 <= sq < TRACK_LENGTH:
                spatial[CHANNEL_OPPONENT_OCCUPANCY, sq] += 1.0

    # Channel 3: Blockade indicators (1.0 at tiles occupied by >= 2 tokens of the same color)
    for sq in range(TRACK_LENGTH):
        if (spatial[CHANNEL_ACTIVE_OCCUPANCY, sq] >= 2.0 or
                spatial[CHANNEL_OPPONENT_OCCUPANCY, sq] >= 2.0):
            spatial[CHANNEL_BLOCKADES, sq] = 1.0

    # -------------------------------------------------------------
    # 2. Context Vector C in R^{12}
    # -------------------------------------------------------------
    context = np.zeros(CONTEXT_DIMENSION, dtype=np.float32)

    # Indices 0 & 1: Tokens in Yard [0..4] normalized by 4
    context[CONTEXT_IDX_ACTIVE_YARD] = sum(1 for t in active_tokens if _is_yard_token(t)) / 4.0
    context[CONTEXT_IDX_OPPONENT_YARD] = sum(1 for t in opponent_tokens if _is_yard_token(t)) / 4.0

    # Indices 2 & 3: Tokens in Home Column [0..4] normalized by 4
    context[CONTEXT_IDX_ACTIVE_HOME] = sum(1 for t in active_tokens if _is_home_token(t)) / 4.0
    context[CONTEXT_IDX_OPPONENT_HOME] = sum(1 for t in opponent_tokens if _is_home_token(t)) / 4.0

    # Indices 4 & 5: Tokens in Goal [0..4] normalized by 4
    context[CONTEXT_IDX_ACTIVE_GOAL] = sum(1 for t in active_tokens if _is_goal_token(t)) / 4.0
    context[CONTEXT_IDX_OPPONENT_GOAL] = sum(1 for t in opponent_tokens if _is_goal_token(t)) / 4.0

    # Index 6: Current dice roll [1..6] normalized by 6
    if 1 <= dice_value <= 6:
        context[CONTEXT_IDX_DICE] = float(dice_value) / 6.0
    else:
        context[CONTEXT_IDX_DICE] = 0.0

    # Indices 7 & 8: Active player ID one-hot ([1, 0] for P0, [0, 1] for P1)
    if current_player == 0:
        context[CONTEXT_IDX_PLAYER_0] = 1.0
        context[CONTEXT_IDX_PLAYER_1] = 0.0
    elif current_player == 1:
        context[CONTEXT_IDX_PLAYER_0] = 0.0
        context[CONTEXT_IDX_PLAYER_1] = 1.0
    else:
        raise ValueError(f"Invalid current_player {current_player}. Must be 0 or 1.")

    # Index 9: Turn fraction (turn / 1000.0)
    turn_val = state_dict.get("turn_count", state_dict.get("turn", 0))
    context[CONTEXT_IDX_TURN_FRACTION] = float(np.clip(float(turn_val) / 1000.0, 0.0, 1.0))

    # Index 10: Bonus roll flag (1.0 if rolling after 6, else 0.0)
    bonus_six_raw = state_dict.get("bonus_six", False) or state_dict.get("is_bonus_six", False)
    if not bonus_six_raw:
        # Fallback if consecutive_sixes is tracked
        bonus_six_raw = state_dict.get("consecutive_sixes", 0) > 0 and state_dict.get("bonus_roll", False)
    context[CONTEXT_IDX_BONUS_SIX] = 1.0 if bool(bonus_six_raw) else 0.0

    # Index 11: Capture bonus flag (1.0 if rolling after capture, else 0.0)
    bonus_capture_raw = state_dict.get("bonus_capture", False) or state_dict.get("is_bonus_capture", False)
    context[CONTEXT_IDX_BONUS_CAPTURE] = 1.0 if bool(bonus_capture_raw) else 0.0

    return StructuredLudoState(
        spatial_tensor=spatial,
        context_vector=context,
        action_mask=action_mask,
    )
