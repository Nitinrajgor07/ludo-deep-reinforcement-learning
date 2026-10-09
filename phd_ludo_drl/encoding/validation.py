"""Observation and state dimension validation with strict protocol constraint checks.

Enforces FINAL_EXPERIMENT_PROTOCOL.md Section 3:
- Spatial shape (4, 52) and channel semantics
- Context dimension exactly 12 and ordered features
- Normalization invariants and token conservation
"""

from typing import Any, Dict, Sequence, Union
import numpy as np
import torch

from phd_ludo_drl.encoding.raw_observation import (
    RAW_STATE_DIMENSION,
    RawLudoObservation,
)
from phd_ludo_drl.environment.constants import (
    TRACK_LENGTH,
    SAFE_SQUARES,
    NUM_PLAYERS,
    TOKENS_PER_PLAYER,
    ACTION_SPACE_SIZE,
    TokenState,
)


class StateValidationError(ValueError):
    """Raised when an observation fails integrity, type, or range checks."""
    pass


class StateDimensionError(ValueError):
    """Raised when an observation tensor does not match expected dimensional contracts."""
    pass


class ProtocolDeviationError(ValueError):
    """Raised when a state representation deviates from the frozen experiment protocol."""
    pass


def validate_raw_observation(raw_obs: RawLudoObservation) -> bool:
    """Validate numerical properties, one-hot sums, and shapes of RawLudoObservation."""
    # Validate current player one-hot
    if raw_obs.current_player_vec.shape != (2,):
        raise StateDimensionError(f"Expected current_player_vec shape (2,), got {raw_obs.current_player_vec.shape}")
    if not np.isclose(raw_obs.current_player_vec.sum(), 1.0):
        raise StateValidationError(f"Current player must be a valid one-hot vector, sum={raw_obs.current_player_vec.sum()}")

    # Validate dice one-hot (can sum to 0.0 before first roll or 1.0)
    if raw_obs.dice_vec.shape != (6,):
        raise StateDimensionError(f"Expected dice_vec shape (6,), got {raw_obs.dice_vec.shape}")
    d_sum = raw_obs.dice_vec.sum()
    if not (np.isclose(d_sum, 0.0) or np.isclose(d_sum, 1.0)):
        raise StateValidationError(f"Dice vector must sum to 0 or 1, got {d_sum}")

    # Validate tokens features
    if raw_obs.tokens_vec.shape != (48,):
        raise StateDimensionError(f"Expected tokens_vec shape (48,), got {raw_obs.tokens_vec.shape}")
    if np.any(raw_obs.tokens_vec < 0.0) or np.any(raw_obs.tokens_vec > 1.0 + 1e-5):
        raise StateValidationError("Token features must be bounded in [0.0, 1.0]")

    # Check each token's status one-hot sum
    for i in range(8):
        status_slice = raw_obs.tokens_vec[i * 6 + 2 : i * 6 + 6]
        if not np.isclose(status_slice.sum(), 1.0):
            raise StateValidationError(f"Token {i} status one-hot sum is {status_slice.sum()}, expected 1.0")

    # Validate winner one-hot
    if raw_obs.winner_vec.shape != (3,):
        raise StateDimensionError(f"Expected winner_vec shape (3,), got {raw_obs.winner_vec.shape}")
    if not np.isclose(raw_obs.winner_vec.sum(), 1.0):
        raise StateValidationError(f"Winner vector must sum to 1.0, got {raw_obs.winner_vec.sum()}")

    # Validate terminal flag
    if raw_obs.terminal_vec.shape != (1,):
        raise StateDimensionError(f"Expected terminal_vec shape (1,), got {raw_obs.terminal_vec.shape}")
    term_val = raw_obs.terminal_vec[0]
    if term_val not in (0.0, 1.0):
        raise StateValidationError(f"Terminal flag must be 0.0 or 1.0, got {term_val}")

    # Validate action mask
    if raw_obs.action_mask.shape != (4,):
        raise StateDimensionError(f"Expected action_mask shape (4,), got {raw_obs.action_mask.shape}")
    if not np.all(np.isin(raw_obs.action_mask, [0.0, 1.0])):
        raise StateValidationError("Action mask must be binary {0.0, 1.0}")

    return True


def validate_flat_vector(vec: Union[np.ndarray, torch.Tensor], expected_dim: int = RAW_STATE_DIMENSION) -> bool:
    """Validate flat state vector dimensionality and bounds."""
    if isinstance(vec, torch.Tensor):
        shape = tuple(vec.shape)
    elif isinstance(vec, np.ndarray):
        shape = vec.shape
    else:
        raise TypeError(f"Expected numpy.ndarray or torch.Tensor, got {type(vec).__name__}")

    if len(shape) == 1:
        dim = shape[0]
    elif len(shape) == 2:
        dim = shape[1]
    else:
        raise StateDimensionError(f"Invalid tensor rank {len(shape)}. Expected 1D or 2D tensor.")

    if dim != expected_dim:
        raise StateDimensionError(f"Expected flat vector dimension {expected_dim}, got {dim}")

    return True


def validate_structured_tensor(tensor: Union[np.ndarray, torch.Tensor]) -> bool:
    """Validate structured spatial representation tensor shape: (4, 52) or (batch, 4, 52)."""
    if isinstance(tensor, torch.Tensor):
        shape = tuple(tensor.shape)
    elif isinstance(tensor, np.ndarray):
        shape = tensor.shape
    else:
        raise TypeError(f"Expected numpy.ndarray or torch.Tensor, got {type(tensor).__name__}")

    if len(shape) == 2:
        channels, length = shape
    elif len(shape) == 3:
        _, channels, length = shape
    else:
        raise StateDimensionError(f"Structured tensor must be rank 2 (4, 52) or rank 3 (B, 4, 52), got {shape}")

    if channels != 4:
        raise StateDimensionError(f"Expected 4 spatial channels, got {channels}")
    if length != TRACK_LENGTH:
        raise StateDimensionError(f"Expected spatial length {TRACK_LENGTH}, got {length}")

    return True


def validate_spatial_tensor(tensor: Union[np.ndarray, torch.Tensor]) -> bool:
    """Validate spatial tensor shape, channel values, and protocol invariants.

    Checks:
    - Shape: (4, 52) or (B, 4, 52)
    - Channel 0: Active token occupancy >= 0, sum <= 4
    - Channel 1: Opponent token occupancy >= 0, sum <= 4
    - Channel 2: Safe square indicators == 1.0 at SAFE_SQUARES, 0.0 elsewhere
    - Channel 3: Blockade indicators == 1.0 where Ch0 >= 2 or Ch1 >= 2, 0.0 elsewhere
    """
    validate_structured_tensor(tensor)

    arr = tensor.cpu().numpy() if isinstance(tensor, torch.Tensor) else np.asarray(tensor)

    if arr.ndim == 2:
        arr_batch = np.expand_dims(arr, 0)
    else:
        arr_batch = arr

    for b in range(arr_batch.shape[0]):
        sp = arr_batch[b]
        # Channel 0 & 1 checks
        if np.any(sp[0] < 0) or np.any(sp[1] < 0):
            raise ProtocolDeviationError("Channel 0/1 token occupancy contains negative values")
        if sp[0].sum() > TOKENS_PER_PLAYER + 1e-4:
            raise ProtocolDeviationError(f"Channel 0 active tokens sum {sp[0].sum()} exceeds total {TOKENS_PER_PLAYER}")
        if sp[1].sum() > TOKENS_PER_PLAYER + 1e-4:
            raise ProtocolDeviationError(f"Channel 1 opponent tokens sum {sp[1].sum()} exceeds total {TOKENS_PER_PLAYER}")

        # Channel 2 safe squares check
        for sq in range(TRACK_LENGTH):
            expected_safe = 1.0 if sq in SAFE_SQUARES else 0.0
            if not np.isclose(sp[2, sq], expected_safe, atol=1e-5):
                raise ProtocolDeviationError(
                    f"Channel 2 safe square mismatch at square {sq}: expected {expected_safe}, got {sp[2, sq]}"
                )

        # Channel 3 blockades check
        for sq in range(TRACK_LENGTH):
            expected_blockade = 1.0 if (sp[0, sq] >= 2.0 or sp[1, sq] >= 2.0) else 0.0
            if not np.isclose(sp[3, sq], expected_blockade, atol=1e-5):
                raise ProtocolDeviationError(
                    f"Channel 3 blockade mismatch at square {sq}: expected {expected_blockade}, got {sp[3, sq]}"
                )

    return True


def validate_context_vector(context: Union[np.ndarray, torch.Tensor]) -> bool:
    """Validate context vector dimensionality (strictly 12) and value invariants.

    Context features (ordered strictly per protocol):
    0: Active yard tokens [0..4] / 4.0
    1: Opponent yard tokens [0..4] / 4.0
    2: Active home column tokens [0..4] / 4.0
    3: Opponent home column tokens [0..4] / 4.0
    4: Active goal tokens [0..4] / 4.0
    5: Opponent goal tokens [0..4] / 4.0
    6: Current dice roll [1..6] / 6.0
    7: Active player ID one-hot index 0 (1.0 for P0, 0.0 for P1)
    8: Active player ID one-hot index 1 (0.0 for P0, 1.0 for P1)
    9: Turn fraction (turn / 1000.0) in [0.0, 1.0]
    10: Bonus roll flag (1.0 if rolling after 6, else 0.0)
    11: Capture bonus flag (1.0 if rolling after capture, else 0.0)
    """
    if isinstance(context, torch.Tensor):
        shape = tuple(context.shape)
        arr = context.cpu().numpy()
    elif isinstance(context, np.ndarray):
        shape = context.shape
        arr = context
    else:
        raise TypeError(f"Expected numpy.ndarray or torch.Tensor, got {type(context).__name__}")

    if len(shape) == 1:
        if shape[0] != 12:
            raise StateDimensionError(f"Context vector must have exactly 12 dimensions, got {shape[0]}")
        arr_batch = np.expand_dims(arr, 0)
    elif len(shape) == 2:
        if shape[1] != 12:
            raise StateDimensionError(f"Batched context vector must have feature dimension 12, got {shape[1]}")
        arr_batch = arr
    else:
        raise StateDimensionError(f"Context tensor rank must be 1 or 2, got rank {len(shape)}")

    for b in range(arr_batch.shape[0]):
        c = arr_batch[b]

        # All features must lie within [0.0, 1.0]
        if np.any(c < -1e-5) or np.any(c > 1.0 + 1e-5):
            raise ProtocolDeviationError(f"Context features out of [0.0, 1.0] bounds: {c}")

        # Yard, Home, Goal token counts in steps of 0.25 (0, 0.25, 0.5, 0.75, 1.0)
        valid_fractions = [0.0, 0.25, 0.5, 0.75, 1.0]
        for idx in range(6):
            if not any(np.isclose(c[idx], vf, atol=1e-5) for vf in valid_fractions):
                raise ProtocolDeviationError(
                    f"Context index {idx} value {c[idx]} is not a valid 4-token normalized fraction {valid_fractions}"
                )

        # Token conservation invariants
        active_sum = c[0] + c[2] + c[4]  # yard + home + goal
        if active_sum > 1.0 + 1e-5:
            raise ProtocolDeviationError(f"Active player off-track tokens exceed 1.0 (got {active_sum})")

        opp_sum = c[1] + c[3] + c[5]      # opp yard + home + goal
        if opp_sum > 1.0 + 1e-5:
            raise ProtocolDeviationError(f"Opponent player off-track tokens exceed 1.0 (got {opp_sum})")

        # Dice roll normalized: 0.0 or [1..6] / 6.0
        dice_val = c[6] * 6.0
        if not np.isclose(dice_val, np.round(dice_val), atol=1e-4):
            raise ProtocolDeviationError(f"Context index 6 dice value is not normalized by 6.0: {c[6]}")

        # Active player one-hot sum: indices 7 and 8
        if not (np.isclose(c[7] + c[8], 1.0, atol=1e-5) and
                c[7] in (0.0, 1.0) and c[8] in (0.0, 1.0)):
            raise ProtocolDeviationError(f"Active player one-hot (indices 7, 8) must sum to 1.0, got [{c[7]}, {c[8]}]")

        # Turn fraction: index 9 in [0.0, 1.0]
        if c[9] < 0.0 or c[9] > 1.0 + 1e-5:
            raise ProtocolDeviationError(f"Turn fraction out of [0.0, 1.0]: {c[9]}")

        # Bonus flags: indices 10 and 11 must be binary {0.0, 1.0}
        if c[10] not in (0.0, 1.0):
            raise ProtocolDeviationError(f"Bonus roll flag (index 10) must be 0.0 or 1.0, got {c[10]}")
        if c[11] not in (0.0, 1.0):
            raise ProtocolDeviationError(f"Capture bonus flag (index 11) must be 0.0 or 1.0, got {c[11]}")

    return True


def validate_frozen_state(state: Any) -> bool:
    """Validate a StructuredLudoState against all frozen protocol rules."""
    if not hasattr(state, "spatial_tensor") or not hasattr(state, "context_vector") or not hasattr(state, "action_mask"):
        raise ProtocolDeviationError("Structured state missing required attributes")

    validate_spatial_tensor(state.spatial_tensor)
    validate_context_vector(state.context_vector)

    mask = np.asarray(state.action_mask)
    if mask.shape != (ACTION_SPACE_SIZE,):
        raise StateDimensionError(f"Action mask shape must be ({ACTION_SPACE_SIZE},), got {mask.shape}")
    if not np.all(np.isin(mask, [0.0, 1.0])):
        raise ProtocolDeviationError("Action mask must be binary {0.0, 1.0}")

    return True


def verify_representation_against_protocol(state: Any, state_dict: Dict[str, Any]) -> bool:
    """Verify state representation against ground-truth state_dict.

    Fails with ProtocolDeviationError if ANY feature diverges from the frozen protocol.
    """
    validate_frozen_state(state)

    curr_p = int(state_dict["current_player"])
    opp_p = 1 - curr_p
    tokens = state_dict.get("tokens", [])
    dice = int(state_dict.get("dice_value", 0))

    active_toks = [t for t in tokens if getattr(t, "player_id", None) == curr_p or (isinstance(t, dict) and t.get("player_id") == curr_p)]
    opp_toks = [t for t in tokens if getattr(t, "player_id", None) == opp_p or (isinstance(t, dict) and t.get("player_id") == opp_p)]

    # 1. Spatial channel 0 ground truth
    ch0_gt = np.zeros(TRACK_LENGTH, dtype=np.float32)
    for t in active_toks:
        if getattr(t, "is_on_track", False) or (isinstance(t, dict) and t.get("state") in (TokenState.TRACK, 1)):
            sq = getattr(t, "track_square", None) if hasattr(t, "track_square") else t.get("track_square")
            if sq is not None:
                ch0_gt[sq] += 1.0

    if not np.allclose(state.spatial_tensor[0], ch0_gt):
        raise ProtocolDeviationError("Spatial Channel 0 does not match active player track occupancy")

    # 2. Spatial channel 1 ground truth
    ch1_gt = np.zeros(TRACK_LENGTH, dtype=np.float32)
    for t in opp_toks:
        if getattr(t, "is_on_track", False) or (isinstance(t, dict) and t.get("state") in (TokenState.TRACK, 1)):
            sq = getattr(t, "track_square", None) if hasattr(t, "track_square") else t.get("track_square")
            if sq is not None:
                ch1_gt[sq] += 1.0

    if not np.allclose(state.spatial_tensor[1], ch1_gt):
        raise ProtocolDeviationError("Spatial Channel 1 does not match opponent track occupancy")

    # 3. Spatial channel 2 safe squares
    ch2_gt = np.zeros(TRACK_LENGTH, dtype=np.float32)
    for sq in SAFE_SQUARES:
        ch2_gt[sq] = 1.0
    if not np.allclose(state.spatial_tensor[2], ch2_gt):
        raise ProtocolDeviationError("Spatial Channel 2 does not match safe square indicators")

    # 4. Spatial channel 3 blockades
    ch3_gt = np.zeros(TRACK_LENGTH, dtype=np.float32)
    for sq in range(TRACK_LENGTH):
        if ch0_gt[sq] >= 2.0 or ch1_gt[sq] >= 2.0:
            ch3_gt[sq] = 1.0
    if not np.allclose(state.spatial_tensor[3], ch3_gt):
        raise ProtocolDeviationError("Spatial Channel 3 does not match blockade indicators")

    # 5. Context features ground truth
    c = state.context_vector

    # Index 0 & 1: Yard
    act_yard = sum(1 for t in active_toks if getattr(t, "is_in_yard", False) or (isinstance(t, dict) and t.get("state") in (TokenState.YARD, 0))) / 4.0
    opp_yard = sum(1 for t in opp_toks if getattr(t, "is_in_yard", False) or (isinstance(t, dict) and t.get("state") in (TokenState.YARD, 0))) / 4.0
    if not np.isclose(c[0], act_yard):
        raise ProtocolDeviationError(f"Context index 0 (active yard) expected {act_yard}, got {c[0]}")
    if not np.isclose(c[1], opp_yard):
        raise ProtocolDeviationError(f"Context index 1 (opp yard) expected {opp_yard}, got {c[1]}")

    # Index 2 & 3: Home column
    act_home = sum(1 for t in active_toks if getattr(t, "is_in_home", False) or (isinstance(t, dict) and t.get("state") in (TokenState.HOME, 2))) / 4.0
    opp_home = sum(1 for t in opp_toks if getattr(t, "is_in_home", False) or (isinstance(t, dict) and t.get("state") in (TokenState.HOME, 2))) / 4.0
    if not np.isclose(c[2], act_home):
        raise ProtocolDeviationError(f"Context index 2 (active home) expected {act_home}, got {c[2]}")
    if not np.isclose(c[3], opp_home):
        raise ProtocolDeviationError(f"Context index 3 (opp home) expected {opp_home}, got {c[3]}")

    # Index 4 & 5: Goal
    act_goal = sum(1 for t in active_toks if getattr(t, "is_finished", False) or (isinstance(t, dict) and t.get("state") in (TokenState.FINISHED, 3))) / 4.0
    opp_goal = sum(1 for t in opp_toks if getattr(t, "is_finished", False) or (isinstance(t, dict) and t.get("state") in (TokenState.FINISHED, 3))) / 4.0
    if not np.isclose(c[4], act_goal):
        raise ProtocolDeviationError(f"Context index 4 (active goal) expected {act_goal}, got {c[4]}")
    if not np.isclose(c[5], opp_goal):
        raise ProtocolDeviationError(f"Context index 5 (opp goal) expected {opp_goal}, got {c[5]}")

    # Index 6: Dice
    dice_exp = (dice / 6.0) if 1 <= dice <= 6 else 0.0
    if not np.isclose(c[6], dice_exp):
        raise ProtocolDeviationError(f"Context index 6 (dice) expected {dice_exp}, got {c[6]}")

    # Index 7 & 8: Player
    p0_exp = 1.0 if curr_p == 0 else 0.0
    p1_exp = 1.0 if curr_p == 1 else 0.0
    if not np.isclose(c[7], p0_exp) or not np.isclose(c[8], p1_exp):
        raise ProtocolDeviationError(f"Context index 7/8 (player one-hot) expected [{p0_exp}, {p1_exp}], got [{c[7]}, {c[8]}]")

    # Index 9: Turn fraction
    turn_val = state_dict.get("turn_count", state_dict.get("turn", 0))
    turn_exp = float(np.clip(float(turn_val) / 1000.0, 0.0, 1.0))
    if not np.isclose(c[9], turn_exp):
        raise ProtocolDeviationError(f"Context index 9 (turn fraction) expected {turn_exp}, got {c[9]}")

    # Index 10: Bonus six
    bonus_six_raw = state_dict.get("bonus_six", False) or state_dict.get("is_bonus_six", False)
    if not bonus_six_raw:
        bonus_six_raw = state_dict.get("consecutive_sixes", 0) > 0 and state_dict.get("bonus_roll", False)
    bonus_six_exp = 1.0 if bool(bonus_six_raw) else 0.0
    if not np.isclose(c[10], bonus_six_exp):
        raise ProtocolDeviationError(f"Context index 10 (bonus roll six) expected {bonus_six_exp}, got {c[10]}")

    # Index 11: Capture bonus
    bonus_cap_raw = state_dict.get("bonus_capture", False) or state_dict.get("is_bonus_capture", False)
    bonus_cap_exp = 1.0 if bool(bonus_cap_raw) else 0.0
    if not np.isclose(c[11], bonus_cap_exp):
        raise ProtocolDeviationError(f"Context index 11 (bonus capture) expected {bonus_cap_exp}, got {c[11]}")

    return True
