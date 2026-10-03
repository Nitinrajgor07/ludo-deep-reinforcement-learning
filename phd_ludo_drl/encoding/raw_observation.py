"""Raw observation extractor and 60-dimensional flat feature vector representation."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch

from phd_ludo_drl.environment.constants import (
    NUM_PLAYERS,
    TOKENS_PER_PLAYER,
    TRACK_STEPS,
    HOME_COLUMN_LENGTH,
    TokenState,
)
from phd_ludo_drl.environment.token import Token

# Authoritative Feature Dimensions (Phase 4 Task 1 Contract)
CURRENT_PLAYER_DIM = 2
DICE_DIM = 6
TOKEN_POS_DIM = 2
TOKEN_STATUS_DIM = 4
TOKEN_FEATURE_DIM = TOKEN_POS_DIM + TOKEN_STATUS_DIM  # 6
ALL_TOKENS_DIM = NUM_PLAYERS * TOKENS_PER_PLAYER * TOKEN_FEATURE_DIM  # 8 * 6 = 48
WINNER_DIM = 3
TERMINAL_DIM = 1

RAW_STATE_DIMENSION = (
    CURRENT_PLAYER_DIM + DICE_DIM + ALL_TOKENS_DIM + WINNER_DIM + TERMINAL_DIM
)  # 2 + 6 + 48 + 3 + 1 = 60

MAX_TRACK_POS = 51.0
MAX_HOME_POS = 4.0


@dataclass
class RawLudoObservation:
    """Disentangled components of a raw Ludo observation."""
    current_player_vec: np.ndarray  # shape (2,)
    dice_vec: np.ndarray            # shape (6,)
    tokens_vec: np.ndarray          # shape (48,)
    winner_vec: np.ndarray          # shape (3,)
    terminal_vec: np.ndarray        # shape (1,)
    action_mask: np.ndarray         # shape (4,) companion tensor

    def to_flat_vector(self) -> np.ndarray:
        """Concatenate sub-vectors into canonical 60-dimensional vector."""
        return np.concatenate([
            self.current_player_vec,
            self.dice_vec,
            self.tokens_vec,
            self.winner_vec,
            self.terminal_vec,
        ], axis=0).astype(np.float32)


def extract_raw_observation(state_dict: Dict[str, Any]) -> RawLudoObservation:
    """Extract authoritative 60-D raw observation from environment state dictionary.

    Args:
        state_dict: Dictionary returned by LudoEnvironment.get_state_dict().

    Returns:
        RawLudoObservation with normalized features and companion action mask.
    """
    current_player = state_dict["current_player"]
    dice_value = state_dict["dice_value"]
    tokens: List[Token] = state_dict["tokens"]
    is_terminal = state_dict["is_terminal"]
    winner = state_dict["winner"]
    action_mask = np.asarray(state_dict["action_mask"], dtype=np.float32)

    # 1. Current player (2-dim one-hot)
    player_vec = np.zeros(CURRENT_PLAYER_DIM, dtype=np.float32)
    player_vec[current_player] = 1.0

    # 2. Dice roll (6-dim one-hot, 1..6 mapped to index 0..5)
    dice_vec = np.zeros(DICE_DIM, dtype=np.float32)
    if 1 <= dice_value <= 6:
        dice_vec[dice_value - 1] = 1.0

    # 3. Token descriptors (8 tokens * 6 features = 48-dim)
    # Sorted deterministically: Player 0 tokens (0..3), Player 1 tokens (0..3)
    sorted_tokens = sorted(tokens, key=lambda t: (t.player_id, t.token_id))
    token_features: List[float] = []

    for t in sorted_tokens:
        # Normalized position features (2)
        norm_track = 0.0
        norm_home = 0.0
        if t.is_on_track and t.track_square is not None:
            norm_track = float(t.track_square) / MAX_TRACK_POS
        elif t.is_in_home and t.home_index is not None:
            norm_home = float(t.home_index) / MAX_HOME_POS

        # Status one-hot (4): [YARD, TRACK, HOME, FINISHED]
        status_vec = [0.0, 0.0, 0.0, 0.0]
        status_vec[int(t.state)] = 1.0

        token_features.extend([norm_track, norm_home] + status_vec)

    tokens_vec = np.array(token_features, dtype=np.float32)

    # 4. Winner state (3-dim one-hot: [None, P0, P1])
    winner_vec = np.zeros(WINNER_DIM, dtype=np.float32)
    if winner is None:
        winner_vec[0] = 1.0
    elif winner == 0:
        winner_vec[1] = 1.0
    elif winner == 1:
        winner_vec[2] = 1.0

    # 5. Terminal state (1-dim binary)
    terminal_vec = np.array([1.0 if is_terminal else 0.0], dtype=np.float32)

    return RawLudoObservation(
        current_player_vec=player_vec,
        dice_vec=dice_vec,
        tokens_vec=tokens_vec,
        winner_vec=winner_vec,
        terminal_vec=terminal_vec,
        action_mask=action_mask,
    )
