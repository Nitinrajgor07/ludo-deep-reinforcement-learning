"""Transforms raw Ludo state into structured 4-channel spatial tensor."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch

from phd_ludo_drl.environment.constants import (
    TRACK_LENGTH,
    SAFE_SQUARES,
    PLAYER_START_SQUARES,
    TokenState,
)
from phd_ludo_drl.environment.token import Token


@dataclass
class StructuredLudoState:
    """Structured spatial state representation containing 4 channels and scalar context."""
    spatial_tensor: np.ndarray  # shape (4, 52)
    context_vector: np.ndarray  # shape (12,)
    action_mask: np.ndarray     # shape (4,) companion tensor


def build_structured_representation(state_dict: Dict[str, Any]) -> StructuredLudoState:
    """Construct the 4-channel spatial board tensor and scalar context.

    Channels (shape 4 x 52):
    - Channel 0: Active player token occupancy on track squares (0..51).
    - Channel 1: Opponent token occupancy on track squares (0..51).
    - Channel 2: Strategic board features:
        - Safe square indicator: +0.5
        - Active player blockade (2+ tokens): +1.0
        - Opponent blockade (2+ tokens): -1.0
    - Channel 3: Path progression & home column activity:
        - Active player tokens in home column mapped to home entry track square.
        - Opponent tokens in home column mapped to opponent home entry track square.

    Context Vector (shape 12):
    - Current player one-hot (2)
    - Dice roll one-hot (6)
    - Token counts in Yard/Goal: [Active Yard, Opponent Yard, Active Finished, Opponent Finished] / 4.0 (4)

    Action Mask (shape 4):
    - Binary legal actions companion tensor.
    """
    current_player = state_dict["current_player"]
    opponent_player = 1 - current_player
    dice_value = state_dict["dice_value"]
    tokens: List[Token] = state_dict["tokens"]
    action_mask = np.asarray(state_dict["action_mask"], dtype=np.float32)

    spatial = np.zeros((4, TRACK_LENGTH), dtype=np.float32)

    # Channel 2 baseline: safe squares
    for sq in SAFE_SQUARES:
        spatial[2, sq] = 0.5

    active_tokens = [t for t in tokens if t.player_id == current_player]
    opponent_tokens = [t for t in tokens if t.player_id == opponent_player]

    # Populate Channel 0 & Channel 1 track occupancy
    for t in active_tokens:
        if t.is_on_track and t.track_square is not None:
            spatial[0, t.track_square] += 1.0

    for t in opponent_tokens:
        if t.is_on_track and t.track_square is not None:
            spatial[1, t.track_square] += 1.0

    # Populate Channel 2 blockades
    for sq in range(TRACK_LENGTH):
        if spatial[0, sq] >= 2.0:
            spatial[2, sq] = 1.0  # Friendly blockade
        elif spatial[1, sq] >= 2.0:
            spatial[2, sq] = -1.0  # Opponent blockade

    # Populate Channel 3 home path indicators
    # Active player entry is start square of opponent (or start square + 50)
    p0_home_entry = (PLAYER_START_SQUARES[0] + 50) % TRACK_LENGTH
    p1_home_entry = (PLAYER_START_SQUARES[1] + 50) % TRACK_LENGTH

    active_home_entry = p0_home_entry if current_player == 0 else p1_home_entry
    opponent_home_entry = p1_home_entry if current_player == 0 else p0_home_entry

    for t in active_tokens:
        if t.is_in_home and t.home_index is not None:
            spatial[3, active_home_entry] += (t.home_index + 1) / 5.0

    for t in opponent_tokens:
        if t.is_in_home and t.home_index is not None:
            spatial[3, opponent_home_entry] += (t.home_index + 1) / 5.0

    # Build Context Vector (12-dim)
    context = np.zeros(12, dtype=np.float32)

    # 1. Current player (2)
    context[current_player] = 1.0

    # 2. Dice roll (6)
    if 1 <= dice_value <= 6:
        context[2 + (dice_value - 1)] = 1.0

    # 3. Yard and finished counts (4)
    active_yard = sum(1 for t in active_tokens if t.is_in_yard) / 4.0
    opp_yard = sum(1 for t in opponent_tokens if t.is_in_yard) / 4.0
    active_fin = sum(1 for t in active_tokens if t.is_finished) / 4.0
    opp_fin = sum(1 for t in opponent_tokens if t.is_finished) / 4.0

    context[8] = active_yard
    context[9] = opp_yard
    context[10] = active_fin
    context[11] = opp_fin

    return StructuredLudoState(
        spatial_tensor=spatial,
        context_vector=context,
        action_mask=action_mask,
    )
