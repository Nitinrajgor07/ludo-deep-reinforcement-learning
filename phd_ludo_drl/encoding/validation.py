"""Observation and state dimension validation with strict constraint checks."""

from typing import Any, Dict, Sequence, Union
import numpy as np
import torch

from phd_ludo_drl.encoding.raw_observation import (
    RAW_STATE_DIMENSION,
    RawLudoObservation,
)


class StateValidationError(ValueError):
    """Raised when an observation fails integrity, type, or range checks."""
    pass


class StateDimensionError(ValueError):
    """Raised when an observation tensor does not match expected dimensional contracts."""
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
    if length != 52:
        raise StateDimensionError(f"Expected spatial length 52, got {length}")

    return True
