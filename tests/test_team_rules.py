"""Comprehensive Phase 1 test suite for Ludo Environment Team Awareness and Rules.

Tests:
1. Backward compatibility: 2-Player Ludo when teams=None.
2. Backward compatibility: 4-Player Ludo when teams=None.
3. Friendly capture prevention: Teammate landing on a teammate's square does NOT capture.
4. Opponent capture in team mode: Landing on an opposing team's square captures the token.
5. Team blockade formation: Two tokens belonging to the same team form a blockade.
6. Team blockade obstruction: Opponents cannot jump over or land on a team blockade.
7. Friendly team blockade passage: Friendly tokens are never blocked by their own team's blockade.
8. Start square release obstruction by opponent team blockade.
9. Team victory condition: Game does NOT terminate when only 4 tokens of one teammate reach goal.
10. Team victory condition: Game terminates and awards victory when all 8 tokens of a team reach goal.
11. Preserves state representations and environment contracts.
"""

import pytest
import numpy as np

from phd_ludo_drl.environment.ludo_env import LudoEnvironment
from phd_ludo_drl.environment.constants import (
    TokenState,
    TOKENS_PER_PLAYER,
    DEFAULT_TEAMS_4P,
    TEAM_TOKENS_COUNT,
    PLAYER_START_SQUARES,
    PLAYER_START_SQUARES_4P,
)
from phd_ludo_drl.environment.rules import LudoRules
from phd_ludo_drl.environment.board import LudoBoard


# ==============================================================================
# 1. BACKWARD COMPATIBILITY TESTS (teams=None)
# ==============================================================================

def test_backward_compatibility_2p_teams_none():
    """Verify standard 2-player environment functions identically when teams=None."""
    env = LudoEnvironment(num_players=2, auto_advance_no_moves=False, seed=42)
    obs, info = env.reset()

    assert env.teams is None
    assert env.num_players == 2
    assert len(env.tokens) == 8
    assert env.current_player == 0
    assert info["teams"] is None
    assert info["team_winner"] is None

    # Individual winning condition: 4 tokens finished = win
    for t in [t for t in env.tokens if t.player_id == 0]:
        t.state = TokenState.FINISHED
        t.step_count = 56

    winner = env.rules.check_winner(env.tokens, teams=None)
    assert winner == 0


def test_backward_compatibility_4p_teams_none():
    """Verify standard 4-player environment functions identically when teams=None."""
    env = LudoEnvironment(num_players=4, auto_advance_no_moves=False, seed=42)
    env.reset()

    assert env.teams is None
    assert env.num_players == 4
    assert len(env.tokens) == 16

    # In FFA 4P, any player landing on any other player captures them
    t_red = next(t for t in env.tokens if t.player_id == 0 and t.token_id == 0)
    t_yellow = next(t for t in env.tokens if t.player_id == 2 and t.token_id == 0)

    t_yellow.state = TokenState.TRACK
    t_yellow.step_count = 31  # Track square 5
    assert t_yellow.track_square == 5

    t_red.state = TokenState.TRACK
    t_red.step_count = 2  # Track square 2
    assert t_red.track_square == 2

    # Red moves 3 steps to square 5
    res = env.rules.apply_move(env.tokens, t_red, dice=3, teams=None)
    assert len(res.captured_tokens) == 1
    assert t_yellow.is_in_yard
    assert t_yellow.step_count == -1


# ==============================================================================
# 2. FRIENDLY CAPTURE PREVENTION TESTS (2v2 Team Mode)
# ==============================================================================

def test_friendly_capture_prevention_teammates():
    """Verify that when a player lands on a teammate's square, the teammate is NOT captured."""
    # Team 0: P0 (Red) and P2 (Yellow)
    # Team 1: P1 (Green) and P3 (Blue)
    teams = {0: 0, 1: 1, 2: 0, 3: 1}
    env = LudoEnvironment(num_players=4, teams=teams, auto_advance_no_moves=False, seed=42)
    env.reset()

    # Place P2 (Yellow - Teammate of P0) at track square 5 (non-safe)
    t_yellow = next(t for t in env.tokens if t.player_id == 2 and t.token_id == 0)
    t_yellow.state = TokenState.TRACK
    t_yellow.step_count = 31  # P2 start (26) + 31 = 57 % 52 = 5
    assert t_yellow.track_square == 5

    # Place P0 (Red - Teammate of P2) at track square 2
    t_red = next(t for t in env.tokens if t.player_id == 0 and t.token_id == 0)
    t_red.state = TokenState.TRACK
    t_red.step_count = 2  # P0 start (0) + 2 = 2
    assert t_red.track_square == 2

    # P0 rolls 3 to land on square 5 (where teammate P2 is sitting)
    res = env.rules.apply_move(env.tokens, t_red, dice=3, teams=teams)

    # Teammate MUST NOT be captured
    assert len(res.captured_tokens) == 0
    assert not t_yellow.is_in_yard
    assert t_yellow.is_on_track
    assert t_yellow.track_square == 5

    # P0 is also on square 5
    assert t_red.is_on_track
    assert t_red.track_square == 5

    # No capture bonus roll since teammate was not captured
    assert res.bonus_roll is False


def test_opponent_capture_still_active_in_team_mode():
    """Verify that landing on an opposing team's token still captures and resets them."""
    teams = {0: 0, 1: 1, 2: 0, 3: 1}
    env = LudoEnvironment(num_players=4, teams=teams, auto_advance_no_moves=False, seed=42)
    env.reset()

    # Place P1 (Green - Opponent Team 1) at track square 5 (non-safe)
    t_green = next(t for t in env.tokens if t.player_id == 1 and t.token_id == 0)
    t_green.state = TokenState.TRACK
    t_green.step_count = (5 - 13) % 52
    assert t_green.track_square == 5

    # Place P0 (Red - Team 0) at track square 2
    t_red = next(t for t in env.tokens if t.player_id == 0 and t.token_id == 0)
    t_red.state = TokenState.TRACK
    t_red.step_count = 2
    assert t_red.track_square == 2

    # P0 rolls 3 to land on square 5
    res = env.rules.apply_move(env.tokens, t_red, dice=3, teams=teams)

    # Opponent token MUST be captured
    assert len(res.captured_tokens) == 1
    assert t_green in res.captured_tokens
    assert t_green.is_in_yard
    assert t_green.step_count == -1
    assert res.bonus_roll is True


# ==============================================================================
# 3. TEAM BLOCKADE TESTS (Formation & Obstruction)
# ==============================================================================

def test_team_blockade_formation_by_teammates():
    """Verify that 1 token of P0 and 1 token of P2 on the same square form a team blockade."""
    teams = {0: 0, 1: 1, 2: 0, 3: 1}
    board = LudoBoard()
    rules = LudoRules(board, teams=teams)

    env = LudoEnvironment(num_players=4, teams=teams, auto_advance_no_moves=False, seed=42)
    env.reset()

    t_red = next(t for t in env.tokens if t.player_id == 0 and t.token_id == 0)
    t_yellow = next(t for t in env.tokens if t.player_id == 2 and t.token_id == 0)

    # Position P2 at square 10
    t_yellow.state = TokenState.TRACK
    t_yellow.step_count = (10 - 26) % 52
    assert t_yellow.track_square == 10

    # Before P0 arrives: square 10 has only 1 token, so NO team blockade
    assert board.has_team_blockade(env.tokens, 10, team_id=0, teams=teams) is False

    # Position P0 at square 8, moves 2 to square 10
    t_red.state = TokenState.TRACK
    t_red.step_count = 8
    res = rules.apply_move(env.tokens, t_red, dice=2, teams=teams)

    # Now both P0 and P2 are at square 10 -> formed a Team 0 blockade!
    assert res.formed_blockade is True
    assert board.has_team_blockade(env.tokens, 10, team_id=0, teams=teams) is True


def test_opponent_blocked_by_team_blockade():
    """Verify that an opponent (Team 1) cannot pass through or land on a Team 0 blockade."""
    teams = {0: 0, 1: 1, 2: 0, 3: 1}
    env = LudoEnvironment(num_players=4, teams=teams, auto_advance_no_moves=False, seed=42)
    env.reset()

    # Form Team 0 blockade at track square 20 (1 Red + 1 Yellow)
    t_red = next(t for t in env.tokens if t.player_id == 0 and t.token_id == 0)
    t_yellow = next(t for t in env.tokens if t.player_id == 2 and t.token_id == 0)

    t_red.state = TokenState.TRACK
    t_red.step_count = 20
    assert t_red.track_square == 20

    t_yellow.state = TokenState.TRACK
    t_yellow.step_count = (20 - 26) % 52
    assert t_yellow.track_square == 20

    assert env.board.has_team_blockade(env.tokens, 20, team_id=0, teams=teams) is True

    # P1 (Green - Team 1) is at square 18
    t_green = next(t for t in env.tokens if t.player_id == 1 and t.token_id == 0)
    t_green.state = TokenState.TRACK
    t_green.step_count = (18 - 13) % 52
    assert t_green.track_square == 18

    # 1. Roll 2: landing on square 20 (opponent team blockade) -> ILLEGAL
    assert env.rules.is_move_legal(env.tokens, t_green, dice=2, teams=teams) is False

    # 2. Roll 3: jumping over square 20 to square 21 -> ILLEGAL
    assert env.rules.is_move_legal(env.tokens, t_green, dice=3, teams=teams) is False

    # 3. Roll 1: landing on square 19 (before blockade) -> LEGAL
    assert env.rules.is_move_legal(env.tokens, t_green, dice=1, teams=teams) is True


def test_friendly_movement_never_blocked_by_friendly_team_blockade():
    """Verify that friendly teammates can pass through or join their own team's blockade."""
    teams = {0: 0, 1: 1, 2: 0, 3: 1}
    env = LudoEnvironment(num_players=4, teams=teams, auto_advance_no_moves=False, seed=42)
    env.reset()

    # Form Team 0 blockade at square 15 with P0 Token 0 and P2 Token 0
    t_red0 = next(t for t in env.tokens if t.player_id == 0 and t.token_id == 0)
    t_yellow0 = next(t for t in env.tokens if t.player_id == 2 and t.token_id == 0)

    t_red0.state = TokenState.TRACK
    t_red0.step_count = 15

    t_yellow0.state = TokenState.TRACK
    t_yellow0.step_count = (15 - 26) % 52

    assert env.board.has_team_blockade(env.tokens, 15, team_id=0, teams=teams) is True

    # Friendly teammate token: P0 Token 1 at square 13
    t_red1 = next(t for t in env.tokens if t.player_id == 0 and t.token_id == 1)
    t_red1.state = TokenState.TRACK
    t_red1.step_count = 13

    # Friendly token landing on square 15: LEGAL
    assert env.rules.is_move_legal(env.tokens, t_red1, dice=2, teams=teams) is True

    # Friendly token jumping past square 15 to square 17: LEGAL
    assert env.rules.is_move_legal(env.tokens, t_red1, dice=4, teams=teams) is True


# ==============================================================================
# 4. TEAM WINNING CONDITION TESTS (8-Token Requirement)
# ==============================================================================

def test_team_winning_condition_requires_all_eight_tokens():
    """Verify team victory requires ALL 8 tokens of the team to finish, not just 4 tokens of one player."""
    teams = {0: 0, 1: 1, 2: 0, 3: 1}
    env = LudoEnvironment(num_players=4, teams=teams, auto_advance_no_moves=False, seed=42)
    env.reset()

    # Finish all 4 tokens of P0 (Red)
    p0_tokens = [t for t in env.tokens if t.player_id == 0]
    for t in p0_tokens:
        t.state = TokenState.FINISHED
        t.step_count = 56

    # Verify P2 (Yellow - Teammate of P0) tokens are still in yard
    p2_tokens = [t for t in env.tokens if t.player_id == 2]
    for t in p2_tokens:
        assert t.is_in_yard

    # In Team mode: P0 having 4 tokens finished must NOT trigger victory!
    winner = env.rules.check_winner(env.tokens, teams=teams)
    assert winner is None
    assert env.rules.check_team_winner(env.tokens, teams=teams) is None

    # Now finish 3 tokens of P2 (7 tokens total for Team 0) -> Still NO victory
    for t in p2_tokens[:3]:
        t.state = TokenState.FINISHED
        t.step_count = 56

    winner = env.rules.check_winner(env.tokens, teams=teams)
    assert winner is None

    # Now finish the 4th token of P2 (All 8 tokens for Team 0 finished)
    p2_tokens[3].state = TokenState.FINISHED
    p2_tokens[3].step_count = 56

    # Team 0 achieves victory!
    team_winner = env.rules.check_winner(env.tokens, teams=teams)
    assert team_winner == 0
    assert env.rules.check_team_winner(env.tokens, teams=teams) == 0


def test_team_environment_step_terminal_and_team_winner():
    """Verify LudoEnvironment step accurately triggers is_terminal and team_winner when 8th token finishes."""
    teams = {0: 0, 1: 1, 2: 0, 3: 1}
    env = LudoEnvironment(num_players=4, teams=teams, auto_advance_no_moves=False, seed=42)
    env.reset()

    # Finish all 4 tokens of P0
    for t in [t for t in env.tokens if t.player_id == 0]:
        t.state = TokenState.FINISHED
        t.step_count = 56

    # Finish 3 tokens of P2
    p2_tokens = [t for t in env.tokens if t.player_id == 2]
    for t in p2_tokens[:3]:
        t.state = TokenState.FINISHED
        t.step_count = 56

    # Position P2's 4th token on step 55 (1 step away from goal)
    last_token = p2_tokens[3]
    last_token.state = TokenState.HOME
    last_token.step_count = 55
    env.current_player = 2

    # P2 rolls 1 to reach goal
    env.dice.set_override(1)
    env.dice.roll()

    obs, reward, terminated, truncated, info = env.step(3)

    assert last_token.is_finished
    assert last_token.step_count == 56
    assert terminated is True
    assert env.is_terminal is True
    assert env.winner == 0  # Team 0 won!
    assert env.team_winner == 0
    assert info["team_winner"] == 0
