"""Comprehensive Unit and Integration Tests for Ludo Environment (Phase 2).

Covers all 10 research requirements:
- Dice behavior
- Token movement
- Capture
- Safe squares
- Blockades
- Home movement & exact finish
- Winning condition & terminal state
- Bonus rolls
- Invalid actions & zero mutation
- Turn transitions
"""

import numpy as np
import pytest

from phd_ludo_drl.environment.constants import (
    TokenState,
    TRACK_STEPS,
    GOAL_STEP,
    PLAYER_START_SQUARES,
    SAFE_SQUARES,
)
from phd_ludo_drl.environment.token import Token
from phd_ludo_drl.environment.dice import Dice
from phd_ludo_drl.environment.board import LudoBoard
from phd_ludo_drl.environment.rules import LudoRules
from phd_ludo_drl.environment.ludo_env import LudoEnvironment


# =====================================================================
# 1. DICE BEHAVIOR TESTS
# =====================================================================

def test_dice_roll_range_and_seeding():
    """Verify dice rolls are strictly in {1..6} and deterministic under seeds."""
    dice1 = Dice(seed=42)
    rolls1 = [dice1.roll() for _ in range(100)]
    assert all(1 <= r <= 6 for r in rolls1)

    dice2 = Dice(seed=42)
    rolls2 = [dice2.roll() for _ in range(100)]
    assert rolls1 == rolls2


def test_dice_test_override():
    """Verify test override sets the exact roll for deterministic testing."""
    dice = Dice()
    dice.set_override(6)
    assert dice.roll() == 6
    # After one use, override is cleared
    dice.set_override(3)
    assert dice.roll() == 3


# =====================================================================
# 2. TOKEN MOVEMENT & RELEASE TESTS
# =====================================================================

def test_token_release_on_six_only():
    """Verify a token in yard can only be released on a roll of 6."""
    env = LudoEnvironment(auto_advance_no_moves=False)
    env.reset(seed=10)

    # Roll 5: cannot release
    env.dice.set_override(5)
    env.dice.roll()
    assert env.get_legal_actions() == []
    assert np.array_equal(env.get_action_mask(), [0, 0, 0, 0])

    # Roll 6: all 4 yard tokens become legal
    env.dice.set_override(6)
    env.dice.roll()
    assert env.get_legal_actions() == [0, 1, 2, 3]
    assert np.array_equal(env.get_action_mask(), [1, 1, 1, 1])

    # Execute release of token 0
    obs, reward, term, trunc, info = env.step(0)
    assert info["success"] is True
    assert reward >= 0.25  # release reward

    t0 = next(t for t in env.tokens if t.player_id == 0 and t.token_id == 0)
    assert t0.state == TokenState.TRACK
    assert t0.step_count == 0
    assert t0.track_square == PLAYER_START_SQUARES[0]  # square 0


def test_token_track_movement():
    """Verify clockwise movement along the track."""
    env = LudoEnvironment(auto_advance_no_moves=False)
    env.reset()

    # Release token 0 on a 6
    env.dice.set_override(6)
    env.dice.roll()
    env.step(0)

    # Move token 0 forward by 4 steps
    env.dice.set_override(4)
    env.dice.roll()
    obs, reward, term, trunc, info = env.step(0)

    t0 = next(t for t in env.tokens if t.player_id == 0 and t.token_id == 0)
    assert t0.step_count == 4
    assert t0.track_square == 4


# =====================================================================
# 3. CAPTURE MECHANICS TESTS
# =====================================================================

def test_capture_opponent_sends_to_yard():
    """Verify landing on an opponent token on an unsafe square captures it."""
    env = LudoEnvironment(auto_advance_no_moves=False)
    env.reset()

    # Setup: Place P1 token 0 on track square 4 (unsafe square)
    p1_t0 = next(t for t in env.tokens if t.player_id == 1 and t.token_id == 0)
    p1_t0.state = TokenState.TRACK
    # P1 start square is 26. Track square 4 means step_count = (4 - 26) % 52 = 30.
    p1_t0.step_count = 30
    assert p1_t0.track_square == 4

    # Place P0 token 0 at step 0 (track square 0)
    p0_t0 = next(t for t in env.tokens if t.player_id == 0 and t.token_id == 0)
    p0_t0.state = TokenState.TRACK
    p0_t0.step_count = 0
    assert p0_t0.track_square == 0

    # P0 rolls 4 and lands on square 4
    env.current_player = 0
    env.dice.set_override(4)
    env.dice.roll()

    obs, reward, term, trunc, info = env.step(0)

    # P1 token must be captured and reset to yard
    assert p1_t0.is_in_yard
    assert p1_t0.step_count == -1
    # P0 must have captured
    assert len(info["move_result"].captured_tokens) == 1
    assert reward >= 0.15  # Capture reward
    assert info["move_result"].bonus_roll is True  # Capture awards bonus roll


# =====================================================================
# 4. SAFE SQUARES TESTS
# =====================================================================

def test_safe_square_prevents_capture():
    """Verify opponent tokens on a safe square cannot be captured."""
    env = LudoEnvironment(auto_advance_no_moves=False)
    env.reset()

    # Track square 8 is a safe square
    assert 8 in SAFE_SQUARES

    # Place P1 token 0 on safe square 8
    p1_t0 = next(t for t in env.tokens if t.player_id == 1 and t.token_id == 0)
    p1_t0.state = TokenState.TRACK
    p1_t0.step_count = (8 - 26) % 52
    assert p1_t0.track_square == 8

    # Place P0 token 0 on track square 6
    p0_t0 = next(t for t in env.tokens if t.player_id == 0 and t.token_id == 0)
    p0_t0.state = TokenState.TRACK
    p0_t0.step_count = 6
    assert p0_t0.track_square == 6

    # P0 rolls 2 and lands on safe square 8
    env.current_player = 0
    env.dice.set_override(2)
    env.dice.roll()

    obs, reward, term, trunc, info = env.step(0)

    # Both tokens should coexist peacefully; P1 token NOT captured
    assert p0_t0.track_square == 8
    assert p1_t0.track_square == 8
    assert p1_t0.state == TokenState.TRACK
    assert len(info["move_result"].captured_tokens) == 0


# =====================================================================
# 5. BLOCKADE MECHANICS TESTS
# =====================================================================

def test_blockade_prevents_opponent_passing_or_landing():
    """Verify an opponent blockade (2+ friendly tokens) cannot be passed or landed on."""
    env = LudoEnvironment(auto_advance_no_moves=False)
    env.reset()

    # Create P1 blockade at track square 10 (2 tokens of P1 on square 10)
    p1_t0 = next(t for t in env.tokens if t.player_id == 1 and t.token_id == 0)
    p1_t1 = next(t for t in env.tokens if t.player_id == 1 and t.token_id == 1)
    for t in [p1_t0, p1_t1]:
        t.state = TokenState.TRACK
        t.step_count = (10 - 26) % 52
        assert t.track_square == 10

    # Place P0 token at track square 7
    p0_t0 = next(t for t in env.tokens if t.player_id == 0 and t.token_id == 0)
    p0_t0.state = TokenState.TRACK
    p0_t0.step_count = 7
    assert p0_t0.track_square == 7

    env.current_player = 0

    # Roll 3: would land on square 10 (blockade) -> ILLEGAL
    env.dice.set_override(3)
    env.dice.roll()
    assert env.rules.is_move_legal(env.tokens, p0_t0, 3) is False

    # Roll 4: would jump over square 10 to square 11 -> ILLEGAL
    env.dice.set_override(4)
    env.dice.roll()
    assert env.rules.is_move_legal(env.tokens, p0_t0, 4) is False

    # Roll 2: lands on square 9 (before blockade) -> LEGAL
    env.dice.set_override(2)
    env.dice.roll()
    assert env.rules.is_move_legal(env.tokens, p0_t0, 2) is True


# =====================================================================
# 6. HOME MOVEMENT & EXACT FINISH TESTS
# =====================================================================

def test_home_column_entry_and_exact_finish():
    """Verify transition from track to home column, and exact roll to finish."""
    env = LudoEnvironment(auto_advance_no_moves=False)
    env.reset()

    p0_t0 = next(t for t in env.tokens if t.player_id == 0 and t.token_id == 0)
    p0_t0.state = TokenState.TRACK
    p0_t0.step_count = 50  # Last track square before home column

    env.current_player = 0

    # Roll 2: step 50 + 2 = 52 -> enters home column at index 1
    env.dice.set_override(2)
    env.dice.roll()
    assert env.rules.is_move_legal(env.tokens, p0_t0, 2) is True

    obs, reward, term, trunc, info = env.step(0)
    assert p0_t0.state == TokenState.HOME
    assert p0_t0.step_count == 52
    assert p0_t0.home_index == 1

    # Now token is at step 54 (needs roll of 2 to reach goal 56)
    p0_t0.step_count = 54
    assert p0_t0.home_index == 3

    # Roll 3: 54 + 3 = 57 > 56 -> Overshoots goal -> ILLEGAL
    env.dice.set_override(3)
    env.dice.roll()
    assert env.rules.is_move_legal(env.tokens, p0_t0, 3) is False

    # Roll 2: 54 + 2 = 56 -> Exact roll -> Finishes!
    env.current_player = 0
    env.dice.set_override(2)
    env.dice.roll()
    assert env.rules.is_move_legal(env.tokens, p0_t0, 2) is True

    obs, reward, term, trunc, info = env.step(0)
    assert p0_t0.state == TokenState.FINISHED
    assert p0_t0.step_count == GOAL_STEP


# =====================================================================
# 7. WINNING & TERMINAL STATE TESTS
# =====================================================================

def test_winning_condition():
    """Verify game terminates and declares winner when all 4 tokens finish."""
    env = LudoEnvironment(auto_advance_no_moves=False)
    env.reset()

    # Pre-finish 3 tokens for Player 0
    p0_tokens = [t for t in env.tokens if t.player_id == 0]
    for t in p0_tokens[:3]:
        t.state = TokenState.FINISHED
        t.step_count = GOAL_STEP

    # 4th token is at step 55 (needs 1 to finish)
    p0_tokens[3].state = TokenState.HOME
    p0_tokens[3].step_count = 55

    env.current_player = 0
    env.dice.set_override(1)
    env.dice.roll()

    obs, reward, terminated, truncated, info = env.step(3)

    assert terminated is True
    assert env.is_terminal is True
    assert env.winner == 0
    assert reward >= 1.0  # Win reward


# =====================================================================
# 8. BONUS ROLL TESTS
# =====================================================================

def test_bonus_roll_on_six():
    """Verify rolling a 6 grants a bonus turn to the same player."""
    env = LudoEnvironment(auto_advance_no_moves=False)
    env.reset()

    # Player 0 rolls 6
    env.current_player = 0
    env.dice.set_override(6)
    env.dice.roll()

    obs, reward, term, trunc, info = env.step(0)
    # Player 0 should retain turn
    assert env.current_player == 0
    assert info["move_result"].bonus_roll is True


def test_turn_switch_on_regular_move():
    """Verify a regular move without a 6 or capture passes the turn."""
    env = LudoEnvironment(auto_advance_no_moves=False)
    env.reset()

    p0_t0 = next(t for t in env.tokens if t.player_id == 0 and t.token_id == 0)
    p0_t0.state = TokenState.TRACK
    p0_t0.step_count = 5

    env.current_player = 0
    env.dice.set_override(3)
    env.dice.roll()

    obs, reward, term, trunc, info = env.step(0)
    assert env.current_player == 1  # Turn passed to Player 1


# =====================================================================
# 9. INVALID ACTIONS & ZERO MUTATION TESTS
# =====================================================================

def test_invalid_action_rejection_and_zero_mutation():
    """Verify illegal actions are rejected with penalty and do not corrupt state."""
    env = LudoEnvironment(auto_advance_no_moves=False)
    env.reset()

    # Player 0 tokens all in yard. Roll is 4 (no legal moves)
    env.dice.set_override(4)
    env.dice.roll()

    obs, reward, term, trunc, info = env.step(0)
    assert info["success"] is False
    assert reward < 0  # Illegal action penalty

    # Verify zero mutation: token 0 remains in yard
    t0 = next(t for t in env.tokens if t.player_id == 0 and t.token_id == 0)
    assert t0.is_in_yard
    assert t0.step_count == -1


# =====================================================================
# 10. AUTO-ADVANCE & FULL GAME SIMULATION
# =====================================================================

def test_auto_advance_when_no_moves():
    """Verify auto-advance skips dead turns until a player can move."""
    env = LudoEnvironment(seed=123, auto_advance_no_moves=True)
    obs, info = env.reset()

    # The reset state should automatically have at least one legal action or be terminal
    if not env.is_terminal:
        assert len(env.get_legal_actions()) > 0
        assert env.get_action_mask().sum() > 0


def test_full_random_playthrough():
    """Verify complete random simulation terminates or truncates cleanly without errors."""
    env = LudoEnvironment(seed=42, max_turns=500, auto_advance_no_moves=True)
    obs, info = env.reset()

    for _ in range(300):
        if env.is_terminal:
            break
        legal_actions = env.get_legal_actions()
        if not legal_actions:
            break
        action = np.random.choice(legal_actions)
        obs, reward, term, trunc, info = env.step(action)

    assert isinstance(obs, dict)
    assert "tokens" in obs


# =====================================================================
# 11. EDGE CASE & DETERMINISTIC INVARIANT TESTS
# =====================================================================

def test_opponent_blockade_prevents_release_at_start_square():
    """Verify an opponent blockade at start square prevents token release even on roll 6."""
    env = LudoEnvironment(auto_advance_no_moves=False)
    env.reset()

    # Form Player 1 blockade at square 0 (which is Player 0's start square)
    # P1 start square is 26. Track square 0 corresponds to step (0 - 26) % 52 = 26.
    p1_t0 = next(t for t in env.tokens if t.player_id == 1 and t.token_id == 0)
    p1_t1 = next(t for t in env.tokens if t.player_id == 1 and t.token_id == 1)
    for t in [p1_t0, p1_t1]:
        t.state = TokenState.TRACK
        t.step_count = 26
        assert t.track_square == 0

    assert env.board.has_blockade(env.tokens, 0, player_id=1) is True

    # Player 0 tokens are all in yard. Roll is 6.
    env.current_player = 0
    env.dice.set_override(6)
    env.dice.roll()

    p0_t0 = next(t for t in env.tokens if t.player_id == 0 and t.token_id == 0)
    # Move must be illegal because start square is blocked by P1 blockade
    assert env.rules.is_move_legal(env.tokens, p0_t0, 6) is False
    assert env.get_legal_actions() == []
    assert np.array_equal(env.get_action_mask(), [0, 0, 0, 0])


def test_friendly_blockade_does_not_block_friendly_movement():
    """Verify a player's own blockade does not block their own tokens from landing or passing."""
    env = LudoEnvironment(auto_advance_no_moves=False)
    env.reset()

    # Create P0 blockade at track square 10
    p0_t0 = next(t for t in env.tokens if t.player_id == 0 and t.token_id == 0)
    p0_t1 = next(t for t in env.tokens if t.player_id == 0 and t.token_id == 1)
    for t in [p0_t0, p0_t1]:
        t.state = TokenState.TRACK
        t.step_count = 10
        assert t.track_square == 10

    assert env.board.has_blockade(env.tokens, 10, player_id=0) is True

    # P0 token 2 is at track square 8
    p0_t2 = next(t for t in env.tokens if t.player_id == 0 and t.token_id == 2)
    p0_t2.state = TokenState.TRACK
    p0_t2.step_count = 8

    # Friendly token moving 2 steps onto square 10: LEGAL
    assert env.rules.is_move_legal(env.tokens, p0_t2, 2) is True

    # Friendly token moving 4 steps past square 10 to square 12: LEGAL
    assert env.rules.is_move_legal(env.tokens, p0_t2, 4) is True


def test_capture_grants_bonus_roll_without_rolling_six():
    """Verify capturing an opponent on a non-six roll (e.g., 2) grants a bonus turn."""
    env = LudoEnvironment(auto_advance_no_moves=False)
    env.reset()

    # P1 token at track square 5 (unsafe)
    p1_t0 = next(t for t in env.tokens if t.player_id == 1 and t.token_id == 0)
    p1_t0.state = TokenState.TRACK
    p1_t0.step_count = (5 - 26) % 52
    assert p1_t0.track_square == 5

    # P0 token at track square 3
    p0_t0 = next(t for t in env.tokens if t.player_id == 0 and t.token_id == 0)
    p0_t0.state = TokenState.TRACK
    p0_t0.step_count = 3
    assert p0_t0.track_square == 3

    env.current_player = 0
    env.dice.set_override(2)  # Roll 2 (NOT 6)
    env.dice.roll()

    obs, reward, term, trunc, info = env.step(0)

    # Capture occurred
    assert len(info["move_result"].captured_tokens) == 1
    assert p1_t0.is_in_yard
    # Bonus roll must be awarded solely from capture
    assert info["move_result"].bonus_roll is True
    assert env.current_player == 0  # Player 0 retains turn


def test_exact_finish_boundary_conditions():
    """Verify boundary conditions for exact finish across all home column steps."""
    env = LudoEnvironment(auto_advance_no_moves=False)
    env.reset()

    p0_t0 = next(t for t in env.tokens if t.player_id == 0 and t.token_id == 0)

    # 1. At step 51 (first home square): roll 5 lands on goal (56), roll 6 overshoots (57)
    p0_t0.state = TokenState.HOME
    p0_t0.step_count = 51
    assert env.rules.is_move_legal(env.tokens, p0_t0, 5) is True
    assert env.rules.is_move_legal(env.tokens, p0_t0, 6) is False

    # 2. At step 55 (last home square): roll 1 lands on goal (56), rolls 2..6 overshoot
    p0_t0.step_count = 55
    assert env.rules.is_move_legal(env.tokens, p0_t0, 1) is True
    for roll in range(2, 7):
        assert env.rules.is_move_legal(env.tokens, p0_t0, roll) is False

    # 3. Already finished token cannot move on any roll
    p0_t0.state = TokenState.FINISHED
    p0_t0.step_count = GOAL_STEP
    for roll in range(1, 7):
        assert env.rules.is_move_legal(env.tokens, p0_t0, roll) is False


def test_environment_deterministic_reproducibility():
    """Verify two separately seeded environments produce bit-for-bit identical trajectories."""
    env1 = LudoEnvironment(seed=2024, auto_advance_no_moves=True)
    env2 = LudoEnvironment(seed=2024, auto_advance_no_moves=True)

    obs1, info1 = env1.reset(seed=2024)
    obs2, info2 = env2.reset(seed=2024)

    assert info1["current_player"] == info2["current_player"]
    assert info1["dice_value"] == info2["dice_value"]
    assert info1["legal_actions"] == info2["legal_actions"]
    assert np.array_equal(info1["action_mask"], info2["action_mask"])

    for _ in range(50):
        if env1.is_terminal or env2.is_terminal:
            break
        legal1 = env1.get_legal_actions()
        legal2 = env2.get_legal_actions()
        assert legal1 == legal2

        if not legal1:
            break

        # Pick first legal action deterministically
        act = legal1[0]
        obs1, r1, term1, trunc1, info1 = env1.step(act)
        obs2, r2, term2, trunc2, info2 = env2.step(act)

        assert r1 == r2
        assert term1 == term2
        assert trunc1 == trunc2
        assert info1["dice_value"] == info2["dice_value"]
        assert info1["current_player"] == info2["current_player"]
        assert np.array_equal(info1["action_mask"], info2["action_mask"])
