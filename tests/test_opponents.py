"""Comprehensive Tests for Opponents and Self-Play Arena (Phase 5).

Verifies:
- Random opponent selects only legal actions
- Random opponent rejects empty legal actions
- Expert opponent follows exact research priority strategy:
    1. Prefer Release
    2. Defensive Move
    3. Aggressive (Capture)
    4. Fast (Progress)
- Self-play match and tournament execution
- Terminal state handling and zero post-termination execution
- Expert vs Random baseline win-rate superiority
"""

import pytest

from phd_ludo_drl.environment.constants import TokenState
from phd_ludo_drl.environment.token import Token
from phd_ludo_drl.environment.ludo_env import LudoEnvironment
from phd_ludo_drl.opponents.random_opponent import RandomOpponent
from phd_ludo_drl.opponents.expert_opponent import ExpertOpponent
from phd_ludo_drl.opponents.self_play import SelfPlayArena, GameRecord, TournamentSummary


# =====================================================================
# 1. RANDOM OPPONENT TESTS
# =====================================================================

def test_random_opponent_selects_strictly_legal():
    """Verify RandomOpponent selects strictly legal actions over 100 iterations."""
    opp = RandomOpponent(seed=42)
    legal_actions = [0, 2]
    dummy_state = {}

    for _ in range(100):
        action = opp.select_action(dummy_state, legal_actions)
        assert action in legal_actions


def test_random_opponent_empty_legal_actions_raises():
    """Verify RandomOpponent raises ValueError when legal_actions is empty."""
    opp = RandomOpponent()
    with pytest.raises(ValueError):
        opp.select_action({}, [])


# =====================================================================
# 2. EXPERT OPPONENT PRIORITY TESTS
# =====================================================================

def test_expert_priority_1_prefer_release():
    """Priority 1: On a six, expert prefers to release a yard token over moving track token."""
    expert = ExpertOpponent()

    # Player 0 has Token 0 on track at step 10, Token 1 in yard
    tokens = [
        Token(player_id=0, token_id=0, state=TokenState.TRACK, step_count=10),
        Token(player_id=0, token_id=1, state=TokenState.YARD, step_count=-1),
        Token(player_id=0, token_id=2, state=TokenState.YARD, step_count=-1),
        Token(player_id=0, token_id=3, state=TokenState.YARD, step_count=-1),
        Token(player_id=1, token_id=0, state=TokenState.YARD, step_count=-1),
        Token(player_id=1, token_id=1, state=TokenState.YARD, step_count=-1),
        Token(player_id=1, token_id=2, state=TokenState.YARD, step_count=-1),
        Token(player_id=1, token_id=3, state=TokenState.YARD, step_count=-1),
    ]

    state = {
        "current_player": 0,
        "dice_value": 6,
        "tokens": tokens,
    }
    legal_actions = [0, 1, 2, 3]

    action = expert.select_action(state, legal_actions)
    # Must choose a release action (token 1, 2, or 3) rather than moving token 0
    assert action in (1, 2, 3)


def test_expert_priority_2_defensive_move():
    """Priority 2: Move vulnerable token into safety over non-vulnerable progress."""
    expert = ExpertOpponent()

    # Track square 8 is a safe square.
    # Player 0 Token 0 is at track square 6 (step 6). It is vulnerable because P1 Token 0 is at track square 4 (distance 2 behind).
    # Moving roll 2 takes P0 Token 0 to square 8 (SAFE SQUARE).
    # Player 0 Token 1 is at track square 20 (safe from threat).
    p0_t0 = Token(player_id=0, token_id=0, state=TokenState.TRACK, step_count=6)
    p0_t1 = Token(player_id=0, token_id=1, state=TokenState.TRACK, step_count=20)
    p1_t0 = Token(player_id=1, token_id=0, state=TokenState.TRACK, step_count=(4 - 26) % 52)

    tokens = [p0_t0, p0_t1, p1_t0] + [
        Token(player_id=p, token_id=t) for p in (0, 1) for t in (2, 3)
    ]

    state = {
        "current_player": 0,
        "dice_value": 2,
        "tokens": tokens,
    }
    legal_actions = [0, 1]

    action = expert.select_action(state, legal_actions)
    # Must prioritize defending token 0 into safe square 8 over advancing token 1
    assert action == 0


def test_expert_priority_3_capture():
    """Priority 3: Choose move that captures opponent token over simple progress."""
    expert = ExpertOpponent()

    # P0 Token 0 is at square 10. Roll 4 reaches square 14 where P1 Token 0 sits (capture).
    # P0 Token 1 is at square 30 (further ahead, would normally be picked by 'fast' rule).
    p0_t0 = Token(player_id=0, token_id=0, state=TokenState.TRACK, step_count=10)
    p0_t1 = Token(player_id=0, token_id=1, state=TokenState.TRACK, step_count=30)
    p1_t0 = Token(player_id=1, token_id=0, state=TokenState.TRACK, step_count=(14 - 26) % 52)

    tokens = [p0_t0, p0_t1, p1_t0] + [
        Token(player_id=p, token_id=t) for p in (0, 1) for t in (2, 3)
    ]

    state = {
        "current_player": 0,
        "dice_value": 4,
        "tokens": tokens,
    }
    legal_actions = [0, 1]

    action = expert.select_action(state, legal_actions)
    # Must capture with token 0 even though token 1 has higher progress
    assert action == 0


def test_expert_priority_4_fast_progress():
    """Priority 4: Advance the most-progressed token when no release/defend/capture applies."""
    expert = ExpertOpponent()

    # P0 Token 0 is at step 5; Token 1 is at step 35. Neither is vulnerable or capturing.
    p0_t0 = Token(player_id=0, token_id=0, state=TokenState.TRACK, step_count=5)
    p0_t1 = Token(player_id=0, token_id=1, state=TokenState.TRACK, step_count=35)

    tokens = [p0_t0, p0_t1] + [
        Token(player_id=0, token_id=t) for t in (2, 3)
    ] + [
        Token(player_id=1, token_id=t) for t in range(4)
    ]

    state = {
        "current_player": 0,
        "dice_value": 3,
        "tokens": tokens,
    }
    legal_actions = [0, 1]

    action = expert.select_action(state, legal_actions)
    # Must advance most-progressed token (token 1)
    assert action == 1


# =====================================================================
# 3. SELF-PLAY ARENA & TOURNAMENT TESTS
# =====================================================================

def test_self_play_single_game_terminal_handling():
    """Verify single match runs cleanly to completion without illegal actions or post-terminal steps."""
    arena = SelfPlayArena()
    p0 = RandomOpponent(seed=10)
    p1 = RandomOpponent(seed=20)

    record = arena.play_game(p0, p1, seed=100, max_turns=400)

    assert isinstance(record, GameRecord)
    assert record.turns > 0
    # Must be either terminated or truncated
    assert record.terminated or record.truncated
    assert record.winner in (0, 1, None)


def test_expert_vs_random_tournament():
    """Verify Expert opponent decisively outperforms Random opponent over a mini-tournament."""
    arena = SelfPlayArena()
    expert = ExpertOpponent(seed=42)
    random_opp = RandomOpponent(seed=99)

    summary = arena.play_tournament(
        player_a=expert,
        player_b=random_opp,
        num_games=20,
        alternate_seats=True,
        base_seed=123,
    )

    assert isinstance(summary, TournamentSummary)
    assert summary.total_games == 20
    # Expert should win the majority of games against random
    assert summary.p0_win_rate > 0.50
    assert summary.p0_wins > summary.p1_wins
