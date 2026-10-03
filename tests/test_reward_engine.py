"""Comprehensive Unit Tests for Reward Engine (Phase 4).

Verifies all 8 baseline reward events:
1. Token release (+0.25)
2. Defensive move (+0.20)
3. Capture (+0.15 for capturer)
4. Progress / Advance (+0.10)
5. Blockade (+0.05)
6. Win (+1.00)
7. Token captured (-0.25 for victim)
8. Loss (-1.00)
9. Out-of-distribution illegal penalty (-0.50)

Also verifies:
- Compound reward events
- Simultaneous two-player symmetric rewards
- Ablation modes (Shaped vs Sparse vs Dense)
- Algorithm independence
"""

import pytest

from phd_ludo_drl.config.config_manager import RewardConfig
from phd_ludo_drl.environment.token import Token
from phd_ludo_drl.environment.rules import MoveResult
from phd_ludo_drl.environment.reward_engine import RewardEngine, RewardBreakdown


@pytest.fixture
def engine():
    return RewardEngine(RewardConfig())


# =====================================================================
# 1. INDIVIDUAL REWARD EVENT TESTS
# =====================================================================

def test_reward_token_release(engine):
    """Event 1: Token released from yard receives exactly +0.25."""
    move = MoveResult(
        token_id=0,
        player_id=0,
        previous_step=-1,
        new_step=0,
        released=True,
    )
    p0_r, p1_r = engine.compute_player_rewards(move, is_terminal=False, winner=None)
    assert p0_r == pytest.approx(0.25)
    assert p1_r == pytest.approx(0.0)


def test_reward_defensive_move(engine):
    """Event 2: Moving a token to a safe square receives exactly +0.20 (+ 0.10 progress)."""
    move = MoveResult(
        token_id=0,
        player_id=0,
        previous_step=6,
        new_step=8,
        moved_to_safe=True,
    )
    p0_bd, _ = engine.compute_player_rewards_breakdown(move, is_terminal=False, winner=None)
    assert p0_bd.defend == pytest.approx(0.20)
    assert p0_bd.advance == pytest.approx(0.10)
    assert p0_bd.total == pytest.approx(0.30)


def test_reward_capture_and_victim_penalty(engine):
    """Event 3 & 7: Capturing opponent token gives +0.15 to capturer, -0.25 to victim."""
    victim_token = Token(player_id=1, token_id=2)
    move = MoveResult(
        token_id=0,
        player_id=0,
        previous_step=10,
        new_step=14,
        captured_tokens=[victim_token],
    )
    p0_bd, p1_bd = engine.compute_player_rewards_breakdown(move, is_terminal=False, winner=None)

    # Player 0 (capturer): +0.15 capture + 0.10 advance = +0.25
    assert p0_bd.capture == pytest.approx(0.15)
    assert p0_bd.advance == pytest.approx(0.10)
    assert p0_bd.total == pytest.approx(0.25)

    # Player 1 (victim): -0.25 knocked
    assert p1_bd.knocked == pytest.approx(-0.25)
    assert p1_bd.total == pytest.approx(-0.25)


def test_reward_progress(engine):
    """Event 4: Regular forward movement receives exactly +0.10 progress reward."""
    move = MoveResult(
        token_id=1,
        player_id=0,
        previous_step=20,
        new_step=24,
    )
    p0_r, p1_r = engine.compute_player_rewards(move, is_terminal=False, winner=None)
    assert p0_r == pytest.approx(0.10)
    assert p1_r == pytest.approx(0.0)


def test_reward_blockade_formation(engine):
    """Event 5: Forming a blockade receives +0.05 (+ 0.10 advance)."""
    move = MoveResult(
        token_id=2,
        player_id=0,
        previous_step=12,
        new_step=16,
        formed_blockade=True,
    )
    p0_bd, _ = engine.compute_player_rewards_breakdown(move, is_terminal=False, winner=None)
    assert p0_bd.blockade == pytest.approx(0.05)
    assert p0_bd.advance == pytest.approx(0.10)
    assert p0_bd.total == pytest.approx(0.15)


def test_reward_win_and_loss(engine):
    """Event 6 & 8: Winning terminal move gives +1.00 to winner, -1.00 to loser."""
    move = MoveResult(
        token_id=3,
        player_id=0,
        previous_step=54,
        new_step=56,
        finished=True,
    )
    p0_bd, p1_bd = engine.compute_player_rewards_breakdown(move, is_terminal=True, winner=0)

    # Player 0 won
    assert p0_bd.win == pytest.approx(1.00)
    assert p0_bd.advance == pytest.approx(0.10)
    assert p0_bd.total == pytest.approx(1.10)

    # Player 1 lost
    assert p1_bd.loss == pytest.approx(-1.00)
    assert p1_bd.total == pytest.approx(-1.00)


def test_reward_illegal_action_penalty(engine):
    """Event 9: Illegal action penalty is strictly -0.50."""
    penalty = engine.get_illegal_action_penalty()
    assert penalty == pytest.approx(-0.50)


# =====================================================================
# 2. ABLATION MODES (SPARSE VS SHAPED VS DENSE)
# =====================================================================

def test_sparse_reward_mode():
    """Verify sparse ablation mode gives 0 for intermediate moves and +/-1.0 on terminal."""
    sparse_engine = RewardEngine(RewardConfig(), mode="sparse")

    # Intermediate release and capture move gives 0 in sparse mode
    move_inter = MoveResult(
        token_id=0,
        player_id=0,
        previous_step=-1,
        new_step=0,
        released=True,
    )
    p0_r, p1_r = sparse_engine.compute_player_rewards(move_inter, is_terminal=False, winner=None)
    assert p0_r == 0.0
    assert p1_r == 0.0

    # Terminal win move gives exactly +1.0 to winner and -1.0 to loser
    move_term = MoveResult(
        token_id=0,
        player_id=0,
        previous_step=55,
        new_step=56,
        finished=True,
    )
    p0_term, p1_term = sparse_engine.compute_player_rewards(move_term, is_terminal=True, winner=0)
    assert p0_term == pytest.approx(1.0)
    assert p1_term == pytest.approx(-1.0)


def test_dense_reward_mode():
    """Verify dense ablation mode includes distance-based scaling."""
    dense_engine = RewardEngine(RewardConfig(), mode="dense")
    # Move forward 4 steps: base 0.10 + 4 * 0.02 = 0.18
    move = MoveResult(
        token_id=0,
        player_id=0,
        previous_step=10,
        new_step=14,
    )
    p0_r, _ = dense_engine.compute_player_rewards(move, is_terminal=False, winner=None)
    assert p0_r == pytest.approx(0.18)


# =====================================================================
# 3. ALGORITHM INDEPENDENCE
# =====================================================================

def test_reward_algorithm_independence(engine):
    """Verify reward calculation is completely decoupled from RL algorithms."""
    move = MoveResult(
        token_id=1,
        player_id=0,
        previous_step=0,
        new_step=6,
        moved_to_safe=False,
    )
    r1 = engine.compute_move_reward(move, is_terminal=False, winner=None)
    r2 = engine.compute_move_reward(move, is_terminal=False, winner=None)
    assert r1 == r2 == pytest.approx(0.10)
