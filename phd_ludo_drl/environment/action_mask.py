"""Action Masking Engine and Policy/Q-Value Masking Utilities for RL Agents.

Guarantees that an RL agent cannot select an illegal action during decision making:
State -> Legal Actions -> Action Mask -> RL Agent (DQN Q-mask / PPO Logit-mask)
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch

from phd_ludo_drl.environment.constants import (
    ACTION_SPACE_SIZE,
    TOKENS_PER_PLAYER,
)
from phd_ludo_drl.environment.token import Token
from phd_ludo_drl.environment.rules import LudoRules


class ActionMaskValidationError(ValueError):
    """Raised when an action mask violates dimensional or binary value constraints."""
    pass


@dataclass(frozen=True)
class ActionMaskInfo:
    """Action mask metadata and legal action summary."""
    mask: np.ndarray            # shape (4,) binary array
    legal_actions: Tuple[int, ...]
    num_legal: int
    has_legal_actions: bool
    is_forced_move: bool        # Exactly one legal action


def validate_action_mask(mask: Union[np.ndarray, torch.Tensor, Sequence[int]]) -> bool:
    """Validate that an action mask is strictly 4-dimensional and binary {0, 1}.

    Args:
        mask: Array, Tensor, or Sequence representing action mask.

    Returns:
        True if valid.

    Raises:
        ActionMaskValidationError: If shape != (4,) or contains non-binary values.
    """
    if isinstance(mask, torch.Tensor):
        shape = tuple(mask.shape)
        arr = mask.detach().cpu().numpy()
    elif isinstance(mask, np.ndarray):
        shape = mask.shape
        arr = mask
    elif isinstance(mask, (list, tuple)):
        shape = (len(mask),)
        arr = np.array(mask)
    else:
        raise ActionMaskValidationError(f"Expected array or tensor, got {type(mask).__name__}")

    # Check shape: (4,) for single, or (B, 4) for batched
    if len(shape) == 1:
        if shape[0] != ACTION_SPACE_SIZE:
            raise ActionMaskValidationError(f"Expected 1D mask of size {ACTION_SPACE_SIZE}, got {shape[0]}")
    elif len(shape) == 2:
        if shape[1] != ACTION_SPACE_SIZE:
            raise ActionMaskValidationError(f"Expected 2D batched mask with shape (B, {ACTION_SPACE_SIZE}), got {shape}")
    else:
        raise ActionMaskValidationError(f"Invalid mask rank {len(shape)}. Expected 1D or 2D tensor.")

    # Check binary values
    unique_vals = np.unique(arr)
    for val in unique_vals:
        if not (np.isclose(val, 0.0) or np.isclose(val, 1.0)):
            raise ActionMaskValidationError(f"Action mask must contain only 0 and 1, found {val}")

    return True


def generate_action_mask(
    tokens: List[Token],
    player_id: int,
    dice: int,
    rules: Optional[LudoRules] = None,
    is_terminal: bool = False,
) -> ActionMaskInfo:
    """Generate the authoritative action mask given current board state and dice roll.

    Args:
        tokens: Current token objects.
        player_id: Active player ID (0 or 1).
        dice: Current dice roll (1..6).
        rules: Optional LudoRules instance.
        is_terminal: If True, mask is guaranteed to be all zeros [0, 0, 0, 0].

    Returns:
        ActionMaskInfo with binary mask and legal actions summary.
    """
    if is_terminal:
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.int8)
        legal_tuple: Tuple[int, ...] = ()
        return ActionMaskInfo(
            mask=mask,
            legal_actions=legal_tuple,
            num_legal=0,
            has_legal_actions=False,
            is_forced_move=False,
        )

    r = rules or LudoRules()
    legal_actions = r.get_legal_actions(tokens, player_id, dice)
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.int8)
    for act in legal_actions:
        mask[act] = 1

    legal_tuple = tuple(sorted(legal_actions))
    num_legal = len(legal_tuple)

    return ActionMaskInfo(
        mask=mask,
        legal_actions=legal_tuple,
        num_legal=num_legal,
        has_legal_actions=(num_legal > 0),
        is_forced_move=(num_legal == 1),
    )


def mask_q_values(
    q_values: torch.Tensor,
    action_mask: torch.Tensor,
    mask_value: float = -1e9,
) -> torch.Tensor:
    """Mask Q-values of illegal actions to large negative value for DQN agents.

    When selecting greedy action via argmax(q_masked), illegal actions with mask=0
    receive mask_value (-1e9) and will never be selected over any legal action.

    Args:
        q_values: Q-value tensor of shape (4,) or (batch_size, 4).
        action_mask: Binary mask tensor of shape (4,) or (batch_size, 4).
        mask_value: Value assigned to illegal actions (default: -1e9).

    Returns:
        Masked Q-value tensor of identical shape.
    """
    if q_values.shape != action_mask.shape:
        raise ValueError(f"Shape mismatch: q_values {q_values.shape} vs action_mask {action_mask.shape}")

    # Set illegal positions (where mask == 0) to mask_value
    illegal_indices = (action_mask == 0)
    masked_q = q_values.clone()
    masked_q[illegal_indices] = mask_value
    return masked_q


def mask_action_logits(
    logits: torch.Tensor,
    action_mask: torch.Tensor,
    mask_value: float = -1e9,
) -> torch.Tensor:
    """Mask action logits prior to softmax for PPO / Policy Gradient agents.

    When softmax is applied to masked logits:
        pi(a_illegal | s) = exp(-1e9) / sum -> 0.0
    guaranteeing that illegal actions have strictly zero probability.

    Args:
        logits: Unnormalized action logits of shape (4,) or (batch_size, 4).
        action_mask: Binary mask tensor of shape (4,) or (batch_size, 4).
        mask_value: Value assigned to illegal logits (default: -1e9).

    Returns:
        Masked logits tensor of identical shape.
    """
    if logits.shape != action_mask.shape:
        raise ValueError(f"Shape mismatch: logits {logits.shape} vs action_mask {action_mask.shape}")

    illegal_indices = (action_mask == 0)
    masked_logits = logits.clone()
    masked_logits[illegal_indices] = mask_value
    return masked_logits


def sample_masked_action(
    logits_or_q: torch.Tensor,
    action_mask: torch.Tensor,
    deterministic: bool = False,
    is_q_values: bool = False,
) -> int:
    """Select a guaranteed legal action given policy logits or Q-values and action mask.

    Args:
        logits_or_q: Tensor of shape (4,).
        action_mask: Binary mask tensor of shape (4,).
        deterministic: If True, picks argmax. If False, samples from masked softmax.
        is_q_values: If True, treats input as Q-values (always argmax over masked Q).

    Returns:
        Selected legal action index in [0..3].

    Raises:
        ValueError: If no legal actions exist (mask is all zeros).
    """
    if action_mask.sum().item() == 0:
        raise ValueError("Cannot select action when action mask contains zero legal actions.")

    if is_q_values or deterministic:
        masked = mask_q_values(logits_or_q, action_mask)
        return int(torch.argmax(masked).item())

    # Stochastic policy sampling with masked softmax
    masked_logits = mask_action_logits(logits_or_q, action_mask)
    probs = torch.softmax(masked_logits, dim=-1)
    dist = torch.distributions.Categorical(probs=probs)
    action = int(dist.sample().item())

    # Assert legality
    assert action_mask[action].item() == 1.0, f"Sampled illegal action {action}!"
    return action
