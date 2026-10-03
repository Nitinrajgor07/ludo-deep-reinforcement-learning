"""Unit Tests for Action Masking Engine (Phase 4).

Covers:
- Action mask generation
- Action mask validation (shape, binary constraints)
- Zero legal actions state ([0, 0, 0, 0])
- One legal action state (forced move)
- Multiple legal actions state (2, 3, 4 actions)
- Terminal state mask ([0, 0, 0, 0])
- Q-value masking for DQN (argmax never selects illegal action)
- Policy logit masking for PPO (softmax probability of illegal action is strictly 0.0)
- Action sampling legality enforcement
"""

import numpy as np
import pytest
import torch

from phd_ludo_drl.environment.constants import TokenState
from phd_ludo_drl.environment.token import Token
from phd_ludo_drl.environment.rules import LudoRules
from phd_ludo_drl.environment.action_mask import (
    ActionMaskInfo,
    ActionMaskValidationError,
    validate_action_mask,
    generate_action_mask,
    mask_q_values,
    mask_action_logits,
    sample_masked_action,
)


@pytest.fixture
def clean_tokens():
    return [
        Token(player_id=p, token_id=t)
        for p in range(2)
        for t in range(4)
    ]


# =====================================================================
# 1. ACTION MASK GENERATION & VARIATION TESTS
# =====================================================================

def test_zero_legal_actions_state(clean_tokens):
    """Verify state where all tokens are in yard and roll is non-six has zero legal actions."""
    info = generate_action_mask(clean_tokens, player_id=0, dice=3)
    assert info.num_legal == 0
    assert info.has_legal_actions is False
    assert info.is_forced_move is False
    assert np.array_equal(info.mask, [0, 0, 0, 0])
    assert info.legal_actions == ()


def test_multiple_legal_actions_state(clean_tokens):
    """Verify rolling a six in initial state gives 4 legal release options."""
    info = generate_action_mask(clean_tokens, player_id=0, dice=6)
    assert info.num_legal == 4
    assert info.has_legal_actions is True
    assert info.is_forced_move is False
    assert np.array_equal(info.mask, [1, 1, 1, 1])
    assert info.legal_actions == (0, 1, 2, 3)


def test_one_legal_action_forced_move(clean_tokens):
    """Verify state with exactly 1 active track token on non-six is a forced move."""
    # Place only token 0 on track, others remain in yard
    clean_tokens[0].state = TokenState.TRACK
    clean_tokens[0].step_count = 10

    info = generate_action_mask(clean_tokens, player_id=0, dice=4)
    assert info.num_legal == 1
    assert info.has_legal_actions is True
    assert info.is_forced_move is True
    assert np.array_equal(info.mask, [1, 0, 0, 0])
    assert info.legal_actions == (0,)


def test_two_legal_actions_subset(clean_tokens):
    """Verify subset of active tokens where only 2 tokens can legally move."""
    # Token 0 is on track at step 10
    clean_tokens[0].state = TokenState.TRACK
    clean_tokens[0].step_count = 10
    # Token 1 is on track at step 54 (needs 2 to finish)
    clean_tokens[1].state = TokenState.HOME
    clean_tokens[1].step_count = 54
    # Tokens 2 & 3 in yard

    # Roll 3: Token 0 can move (10 + 3 = 13), Token 1 cannot move (54 + 3 = 57 > 56 overshoot)
    info = generate_action_mask(clean_tokens, player_id=0, dice=3)
    assert np.array_equal(info.mask, [1, 0, 0, 0])

    # Roll 2: Token 0 can move, Token 1 can finish (54 + 2 = 56)
    info2 = generate_action_mask(clean_tokens, player_id=0, dice=2)
    assert np.array_equal(info2.mask, [1, 1, 0, 0])
    assert info2.num_legal == 2


def test_terminal_state_mask(clean_tokens):
    """Verify terminal state produces all-zero action mask [0, 0, 0, 0]."""
    info = generate_action_mask(clean_tokens, player_id=0, dice=6, is_terminal=True)
    assert info.num_legal == 0
    assert info.has_legal_actions is False
    assert np.array_equal(info.mask, [0, 0, 0, 0])


# =====================================================================
# 2. ACTION MASK VALIDATION TESTS
# =====================================================================

def test_validate_action_mask_valid():
    """Verify clean 4-element binary mask passes validation."""
    assert validate_action_mask(np.array([1, 0, 1, 0])) is True
    assert validate_action_mask(torch.tensor([0, 0, 1, 1])) is True
    assert validate_action_mask([0, 0, 0, 0]) is True


def test_validate_action_mask_invalid_size():
    """Verify non-4-element mask raises ActionMaskValidationError."""
    with pytest.raises(ActionMaskValidationError):
        validate_action_mask(np.array([1, 0, 1]))  # 3 elements


def test_validate_action_mask_invalid_values():
    """Verify non-binary values raise ActionMaskValidationError."""
    with pytest.raises(ActionMaskValidationError):
        validate_action_mask(np.array([1.0, 0.5, 0.0, 0.0]))


# =====================================================================
# 3. Q-VALUE MASKING (DQN AGENT INTEGRATION)
# =====================================================================

def test_mask_q_values_prevents_illegal_argmax():
    """Verify Q-masking forces argmax to select legal action even if illegal Q-value is higher."""
    # Assume unmasked Q-values strongly prefer action 1 (which is illegal)
    q_values = torch.tensor([2.0, 100.0, -1.0, 50.0])
    mask = torch.tensor([1.0, 0.0, 1.0, 0.0])  # Only 0 and 2 are legal

    masked_q = mask_q_values(q_values, mask, mask_value=-1e9)

    # Illegal actions 1 and 3 must be set to -1e9
    assert masked_q[1].item() == -1e9
    assert masked_q[3].item() == -1e9

    # Greedy choice must pick action 0 (highest legal Q)
    best_action = int(torch.argmax(masked_q).item())
    assert best_action == 0
    assert mask[best_action].item() == 1.0


# =====================================================================
# 4. LOGIT MASKING (PPO POLICY INTEGRATION)
# =====================================================================

def test_mask_action_logits_zeroes_illegal_probabilities():
    """Verify logit-masking assigns strictly 0.0 softmax probability to illegal actions."""
    logits = torch.tensor([1.0, 10.0, -2.0, 5.0])
    mask = torch.tensor([1.0, 0.0, 0.0, 1.0])  # Actions 0 and 3 are legal

    masked_logits = mask_action_logits(logits, mask, mask_value=-1e9)
    probs = torch.softmax(masked_logits, dim=-1)

    # Illegal actions must have 0.0 probability
    assert probs[1].item() == pytest.approx(0.0, abs=1e-7)
    assert probs[2].item() == pytest.approx(0.0, abs=1e-7)

    # Legal actions must sum to 1.0
    assert (probs[0] + probs[3]).item() == pytest.approx(1.0, abs=1e-6)


# =====================================================================
# 5. SAMPLE MASKED ACTION TESTS
# =====================================================================

def test_sample_masked_action_always_legal():
    """Verify repeated sampling from stochastic masked policy never picks illegal actions."""
    logits = torch.tensor([5.0, 100.0, 3.0, 50.0])
    mask = torch.tensor([1.0, 0.0, 1.0, 0.0])  # Only 0 and 2 legal

    for _ in range(50):
        action = sample_masked_action(logits, mask, deterministic=False)
        assert action in (0, 2)


def test_sample_masked_action_zero_legal_raises():
    """Verify sampling from an all-zero mask raises ValueError."""
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0])
    mask = torch.tensor([0.0, 0.0, 0.0, 0.0])

    with pytest.raises(ValueError):
        sample_masked_action(logits, mask)
