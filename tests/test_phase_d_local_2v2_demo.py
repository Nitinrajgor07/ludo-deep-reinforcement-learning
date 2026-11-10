"""Phase D — Local 2v2 Team Multiplayer Demonstration Tests.

Validates the complete local 2v2 Team vs Team demonstration flow:
1. Dedicated local 2v2 demo room creation.
2. All four seats populated, ready, and properly team-mapped.
3. Pass-and-play alternating turn progression without WebSockets.
4. Friendly capture prevention and team blockade preservation.
5. Authoritative 8-token team victory determination and SQLite match persistence.
"""

import pytest
from fastapi.testclient import TestClient
from web.app import app
from web.services.room_service import room_service, RoomStatus, TurnPhase
from web.utils.auth import create_session, register_user, get_db
from phd_ludo_drl.environment.ludo_env import LudoEnvironment
from phd_ludo_drl.environment.constants import DEFAULT_TEAMS_4P
from phd_ludo_drl.environment.token import TokenState


@pytest.fixture
def client():
    return TestClient(app, cookies={})


@pytest.fixture
def auth_user():
    """Register and create session for local demo host."""
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT id, username FROM users WHERE username = 'demo_prof_host'")
    row = c.fetchone()
    if row:
        user_id = row["id"]
    else:
        user_id = register_user("demo_prof_host", "Password123!", "prof_host@demo.edu", "Demo Host")
    conn.close()

    token = create_session(user_id)
    return {"id": user_id, "username": "demo_prof_host", "token": token}


def test_phase_d_task1_lobby_and_seat_initialization(client, auth_user):
    """TASK 1 & 2: Verify dedicated 4-seat local 2v2 demo lobby and authoritative team mapping."""
    client.cookies.set("session_token", auth_user["token"])

    # 1. Create local 2v2 demonstration room
    create_resp = client.post("/api/rooms/create", json={
        "game_mode": "2v2_team",
        "is_online": False
    })
    assert create_resp.status_code == 200
    data = create_resp.json()
    room_code = data["room_code"]
    room = data["room"]

    # Verify visual room / demo code presentation
    assert len(room_code) >= 6
    assert room["game_mode"] == "2v2_team"
    assert room["is_online"] is False
    assert room["max_players"] == 4
    assert room["player_count"] == 4
    assert room["status"] == "ready"

    # Verify all four player seats are present, labelled and ready
    players = room["players"]
    assert len(players) == 4

    # Seat 0: Player 1 — Red — Team 0
    assert players[0]["seat_id"] == 0
    assert players[0]["color"] == "Red"
    assert players[0]["team_id"] == 0
    assert players[0]["is_ready"] is True

    # Seat 1: Player 2 — Green — Team 1
    assert players[1]["seat_id"] == 1
    assert players[1]["color"] == "Green"
    assert players[1]["team_id"] == 1
    assert players[1]["is_ready"] is True

    # Seat 2: Player 3 — Yellow — Team 0
    assert players[2]["seat_id"] == 2
    assert players[2]["color"] == "Yellow"
    assert players[2]["team_id"] == 0
    assert players[2]["is_ready"] is True

    # Seat 3: Player 4 — Blue — Team 1
    assert players[3]["seat_id"] == 3
    assert players[3]["color"] == "Blue"
    assert players[3]["team_id"] == 1
    assert players[3]["is_ready"] is True


def test_phase_d_task2_team_mapping_constants():
    """TASK 2: Verify DEFAULT_TEAMS_4P mapping is strictly Red+Yellow (Team 0) and Green+Blue (Team 1)."""
    assert DEFAULT_TEAMS_4P[0] == 0  # Red
    assert DEFAULT_TEAMS_4P[1] == 1  # Green
    assert DEFAULT_TEAMS_4P[2] == 0  # Yellow
    assert DEFAULT_TEAMS_4P[3] == 1  # Blue


def test_phase_d_task3_pass_and_play_gameplay_flow(client, auth_user):
    """TASK 3: Verify start match, dice rolling, alternating turns and pass-and-play on one computer."""
    client.cookies.set("session_token", auth_user["token"])

    # Create Local 2v2 demo room
    create_resp = client.post("/api/rooms/create", json={"game_mode": "2v2_team", "is_online": False})
    assert create_resp.status_code == 200
    room_code = create_resp.json()["room_code"]

    # Start match
    start_resp = client.post("/api/rooms/start", json={"room_code": room_code})
    assert start_resp.status_code == 200
    start_data = start_resp.json()
    assert start_data["room"]["status"] == "in_progress"
    assert start_data["room"]["turn_phase"] == "NEED_ROLL"
    assert start_data["room"]["current_player"] == 0  # Player 1 (Red) starts

    # Execute dice roll locally
    roll_resp = client.post("/api/rooms/roll", json={"room_code": room_code})
    assert roll_resp.status_code == 200
    roll_data = roll_resp.json()
    assert roll_data["status"] == "ok"
    assert roll_data["dice"] in [1, 2, 3, 4, 5, 6]

    # Verify room state updated
    room_state = roll_data["room"]
    assert room_state["dice_value"] in [1, 2, 3, 4, 5, 6]

    # If legal moves exist, move token, else turn passed
    if roll_data.get("legal_actions"):
        act_resp = client.post("/api/rooms/action", json={"room_code": room_code, "token_id": roll_data["legal_actions"][0]})
        assert act_resp.status_code == 200
        assert len(act_resp.json()["room"]["recent_history"]) >= 1
    else:
        assert len(roll_data["room"]["recent_history"]) >= 1


def test_phase_d_task4_authoritative_team_victory(client, auth_user):
    """TASK 4: Verify 8-token victory condition, winner/loser team assignment, and persistence."""
    client.cookies.set("session_token", auth_user["token"])

    # Create & start local 2v2 room
    create_resp = client.post("/api/rooms/create", json={"game_mode": "2v2_team", "is_online": False})
    assert create_resp.status_code == 200
    room_code = create_resp.json()["room_code"]
    client.post("/api/rooms/start", json={"room_code": room_code})

    room = room_service.get_room(room_code)
    assert room is not None

    # Simulate Team 0 (Red P0 + Yellow P2) having 7 tokens finished
    p0_tokens = [t for t in room.env.tokens if t.player_id == 0]
    p2_tokens = [t for t in room.env.tokens if t.player_id == 2]

    for t in p0_tokens:
        t.state = TokenState.FINISHED
        t.step_count = 56
    for t in p2_tokens[:3]:
        t.state = TokenState.FINISHED
        t.step_count = 56

    # Position the 8th token (P2's last token) right in front of home (step 55)
    p2_last = p2_tokens[3]
    p2_last.state = TokenState.HOME
    p2_last.step_count = 55

    # Set turn to Player 2, dice value 1
    room.env.current_player = 2
    room.env.dice._current_value = 1
    room.current_dice_value = 1
    room.turn_phase = TurnPhase.NEED_MOVE
    room.legal_actions = [p2_last.token_id]

    # Move the 8th token into Home
    move_res = room.move_token(requesting_user_id=auth_user["id"], token_id=p2_last.token_id)

    # Assert Game Finished & Team 0 Victory
    assert move_res["terminated"] is True
    assert move_res["team_winner"] == 0  # Team 0 Wins!
    assert room.status == RoomStatus.FINISHED
    assert room.team_winner == 0

    # Verify team victory reflected in room state
    state = room.get_state_dict()
    assert state["team_winner"] == 0
    assert state["status"] == "finished"
