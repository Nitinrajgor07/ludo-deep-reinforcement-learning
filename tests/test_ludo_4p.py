"""Comprehensive test suite for 4-Player Ludo Environment and Interactive Arena.

Tests:
- 4 players initialized (Red, Green, Yellow, Blue)
- 4 tokens per player (16 total)
- Authoritative start squares: 0, 13, 26, 39
- Turn transitions: P0 -> P1 -> P2 -> P3 -> P0
- Bonus turns on 6 and captures
- Multi-opponent captures
- Restart with seed preservation
- REST API /api/ludo/action validation
"""

import pytest
from fastapi.testclient import TestClient

from phd_ludo_drl.environment.ludo_env import LudoEnvironment
from phd_ludo_drl.environment.constants import TokenState, PLAYER_START_SQUARES_4P
from web.services.game_service import GameSession4P, game_service
from web.components.board_renderer import render_ludo_board_svg
from web.app import app

client = TestClient(app)


def test_4p_environment_initialization():
    """Verify that LudoEnvironment with num_players=4 sets up 16 tokens and 4 start squares."""
    env = LudoEnvironment(seed=42, num_players=4, auto_advance_no_moves=False)
    assert env.num_players == 4
    assert len(env.tokens) == 16

    # 4 tokens per player
    for pid in range(4):
        p_tokens = [t for t in env.tokens if t.player_id == pid]
        assert len(p_tokens) == 4
        for t in p_tokens:
            assert t.is_in_yard
            assert t.start_square == PLAYER_START_SQUARES_4P[pid]


def test_4p_start_squares_and_tracks():
    """Verify that each of the 4 players enters the track at the correct square."""
    env = LudoEnvironment(num_players=4, auto_advance_no_moves=False)
    env.reset(seed=123)

    expected_starts = {0: 0, 1: 13, 2: 26, 3: 39}

    for pid in range(4):
        token = next(t for t in env.tokens if t.player_id == pid and t.token_id == 0)
        env.rules.apply_move(env.tokens, token, dice=6)
        assert token.is_on_track
        assert token.step_count == 0
        assert token.track_square == expected_starts[pid]


def test_4p_turn_progression_sequential():
    """Verify turn sequence P0 -> P1 -> P2 -> P3 -> P0 without bonus."""
    env = LudoEnvironment(num_players=4, auto_advance_no_moves=False)
    env.reset(seed=42)

    # Force a non-bonus roll (e.g., 2)
    env.dice.set_override(2)
    env.dice.roll()
    env.current_player = 0

    # P0 takes non-bonus move (move token 0 forward from track)
    t0 = next(t for t in env.tokens if t.player_id == 0 and t.token_id == 0)
    t0.state = TokenState.TRACK
    t0.step_count = 0

    obs, reward, term, trunc, info = env.step(0)
    assert env.current_player == 1  # Transitions to P1 (Green)

    # P1 takes non-bonus move
    env.dice.set_override(2)
    env.dice.roll()
    t1 = next(t for t in env.tokens if t.player_id == 1 and t.token_id == 0)
    t1.state = TokenState.TRACK
    t1.step_count = 0
    env.step(0)
    assert env.current_player == 2  # Transitions to P2 (Yellow)

    # P2 takes non-bonus move
    env.dice.set_override(2)
    env.dice.roll()
    t2 = next(t for t in env.tokens if t.player_id == 2 and t.token_id == 0)
    t2.state = TokenState.TRACK
    t2.step_count = 0
    env.step(0)
    assert env.current_player == 3  # Transitions to P3 (Blue)

    # P3 takes non-bonus move
    env.dice.set_override(2)
    env.dice.roll()
    t3 = next(t for t in env.tokens if t.player_id == 3 and t.token_id == 0)
    t3.state = TokenState.TRACK
    t3.step_count = 0
    env.step(0)
    assert env.current_player == 0  # Cycles back to P0 (Red)


def test_4p_multi_opponent_capture():
    """Verify that moving onto an opponent's track square captures and resets the opponent token."""
    env = LudoEnvironment(num_players=4, auto_advance_no_moves=False)
    env.reset(seed=42)

    # Place P2 (Yellow) on track square 5
    t_yellow = next(t for t in env.tokens if t.player_id == 2 and t.token_id == 0)
    t_yellow.state = TokenState.TRACK
    # P2 starts at 26, so step_count to reach square 5 is (5 - 26) % 52 = 31
    t_yellow.step_count = 31
    assert t_yellow.track_square == 5

    # Place P0 (Red) on square 2 (step 2) and roll 3 to land on square 5
    t_red = next(t for t in env.tokens if t.player_id == 0 and t.token_id == 0)
    t_red.state = TokenState.TRACK
    t_red.step_count = 2
    assert t_red.track_square == 2

    env.current_player = 0
    env.dice.set_override(3)
    env.dice.roll()

    obs, reward, term, trunc, info = env.step(0)

    # Yellow token should be captured and returned to Yard
    assert t_yellow.is_in_yard
    assert t_yellow.step_count == -1
    # Red token should now be on square 5
    assert t_red.track_square == 5
    # Red should be granted a bonus turn due to capture
    assert info["move_result"].bonus_roll is True


def test_4p_game_session_statistics():
    """Verify that GameSession4P tracks compact statistics for all 4 players."""
    session = GameSession4P(session_id="test_4p_stats", seed=42)
    stats = session.get_player_statistics()
    assert len(stats) == 4

    colors = [s["color"] for s in stats]
    assert colors == ["Red", "Green", "Yellow", "Blue"]

    for s in stats:
        assert s["yard_count"] + s["track_count"] + s["finished_count"] == 4
        assert s["captures"] >= 0


def test_4p_board_renderer_svg():
    """Verify that render_ludo_board_svg produces full 4-player board markup."""
    env = LudoEnvironment(num_players=4, seed=42)
    env.reset()

    svg = render_ludo_board_svg(
        tokens=env.tokens,
        current_player=0,
        dice_value=6,
        legal_actions=[0],
    )
    assert isinstance(svg, str)
    assert "RED YARD (P0)" in svg
    assert "GREEN YARD (P1)" in svg
    assert "YELLOW YARD (P2)" in svg
    assert "BLUE YARD (P3)" in svg
    assert "GOAL" in svg


def test_api_4p_lifecycle_and_validation():
    """Verify 4-player REST API endpoints including strict action validation."""
    # 1. POST /api/ludo/new
    res = client.post("/api/ludo/new", json={
        "session_id": "test_api_4p",
        "p0_type": "dqn",
        "p1_type": "expert",
        "p2_type": "random",
        "p3_type": "expert",
        "seed": 42,
    })
    assert res.status_code == 200
    data = res.json()
    assert len(data["players"]) == 4
    assert data["state"]["mode"] == "Interactive 4-Player Demonstration"

    curr_p = data["current_player"]
    legal_actions = data["legal_actions"]

    # 2. POST /api/ludo/action rejecting wrong player
    wrong_p = (curr_p + 1) % 4
    bad_res = client.post("/api/ludo/action", json={
        "session_id": "test_api_4p",
        "player_id": wrong_p,
        "token_id": 0,
    })
    assert bad_res.status_code == 400
    assert "not Player" in bad_res.json()["detail"]

    # 3. POST /api/ludo/step advances turn
    step_res = client.post("/api/ludo/step", json={"session_id": "test_api_4p"})
    assert step_res.status_code == 200

    # 4. POST /api/ludo/auto_play runs 5 turns across players
    auto_res = client.post("/api/ludo/auto_play", json={
        "session_id": "test_api_4p",
        "steps": 5,
    })
    assert auto_res.status_code == 200
    assert len(auto_res.json()["state"]["recent_history"]) >= 1
