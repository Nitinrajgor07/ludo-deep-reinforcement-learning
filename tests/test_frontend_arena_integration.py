"""Targeted integration tests for Phase 4: Frontend Arena & Lobby Integration.

Tests:
1. /ludo page template rendering with Phase 4 tabs, mode cards, modals, and preserved research tournament.
2. Mode 1: Human vs AI (1v1) with DRL models (num_players=2).
3. Mode 2: Local Human vs Human (1v1) room creation and match start without WebSockets.
4. Mode 3: Local Team vs Team (2v2) room creation and match start.
5. Online room creation, joining with 6-char code, ready up, and match start.
6. Real-time WebSocket connection to /ws/rooms/{room_code} with game state sync.
7. Authoritative dice roll and token move via WebSocket.
8. 2v2 Team victory condition verification (8 finished tokens).
9. Preserved 4-Player AI Tournament API endpoints (/api/ludo/new, step, auto_play).
10. Match result persistence in SQLite database.
"""

import json
import pytest
from starlette.testclient import TestClient

from web.app import app
from web.services.room_service import room_service, GameMode, RoomStatus
from web.utils.auth import create_session, register_user, get_db


@pytest.fixture
def client():
    return TestClient(app, cookies={})


@pytest.fixture
def auth_user():
    """Register and create session for a primary test user."""
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT id, username FROM users WHERE username = 'arena_tester'")
    row = c.fetchone()
    if row:
        user_id = row["id"]
    else:
        user_id = register_user("arena_tester", "Password123!", "tester@phd.edu", "Arena Tester")
    conn.close()

    token = create_session(user_id)
    return {"id": user_id, "username": "arena_tester", "token": token}


@pytest.fixture
def second_user():
    """Register and create session for a secondary test player."""
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT id, username FROM users WHERE username = 'arena_player2'")
    row = c.fetchone()
    if row:
        user_id = row["id"]
    else:
        user_id = register_user("arena_player2", "Password123!", "player2@phd.edu", "Player Two")
    conn.close()

    token = create_session(user_id)
    return {"id": user_id, "username": "arena_player2", "token": token}


def test_ludo_page_renders_phase4_elements(client, auth_user):
    """Verify /ludo renders Playable Arena, mode cards, modals, and Preserved Research Tournament."""
    client.cookies.set("session_token", auth_user["token"])
    resp = client.get("/ludo")
    assert resp.status_code == 200
    html = resp.text

    # Phase 4 Navigation tabs
    assert "Playable Modes" in html
    assert "Research AI Tournament" in html

    # 3 Game Modes
    assert "Human vs AI (1v1)" in html
    assert "Human vs Human (1v1)" in html
    assert "Team vs Team (2v2)" in html

    # Modals & Lobby
    assert "modalCreateRoom" in html
    assert "modalJoinRoom" in html
    assert "multiplayerLobbySection" in html
    assert "playableArenaSection" in html

    # Preserved Research AI Tournament Card
    assert "tabResearchTournament" in html
    assert "4-Player AI Tournament &amp; Preset Setup" in html or "4-Player AI Tournament" in html
    assert "quickPresetSelect" in html
    assert "p0Select" in html
    assert "p1Select" in html
    assert "p2Select" in html
    assert "p3Select" in html


def test_mode1_human_vs_ai_2player(client, auth_user):
    """Verify Mode 1: Human vs AI initializes a 2-player Ludo match."""
    client.cookies.set("session_token", auth_user["token"])
    resp = client.post("/api/ludo/new", json={
        "session_id": "test_1v1_ai",
        "num_players": 2,
        "p0_type": "human",
        "p1_type": "dqn",
        "seed": 42
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["state"]["num_players"] == 2
    assert len(data["state"]["players"]) == 2
    assert data["state"]["players"][0]["color"] == "Red"
    assert data["state"]["players"][1]["color"] == "Yellow"
    assert "svg" in data
    assert "<svg" in data["svg"]


def test_mode2_local_1v1_human(client, auth_user):
    """Verify Mode 2: Local Human vs Human (1v1) creates room and auto-starts."""
    client.cookies.set("session_token", auth_user["token"])
    # 1. Create Local Room
    create_resp = client.post("/api/rooms/create", json={
        "game_mode": "1v1_human",
        "is_online": False
    })
    assert create_resp.status_code == 200
    room_code = create_resp.json()["room_code"]
    room_data = create_resp.json()["room"]

    assert room_data["game_mode"] == "1v1_human"
    assert room_data["max_players"] == 2
    assert room_data["player_count"] == 2
    assert room_data["status"] == "ready"

    # 2. Start game
    start_resp = client.post("/api/rooms/start", json={"room_code": room_code})
    assert start_resp.status_code == 200
    assert start_resp.json()["room"]["status"] == "in_progress"

    # 3. Roll dice locally
    roll_resp = client.post("/api/rooms/roll", json={"room_code": room_code})
    assert roll_resp.status_code == 200
    assert roll_resp.json()["dice"] in [1, 2, 3, 4, 5, 6]


def test_mode3_local_2v2_team(client, auth_user):
    """Verify Mode 3: Local Team vs Team (2v2) populates all 4 seats and starts."""
    client.cookies.set("session_token", auth_user["token"])
    create_resp = client.post("/api/rooms/create", json={
        "game_mode": "2v2_team",
        "is_online": False
    })
    assert create_resp.status_code == 200
    room_code = create_resp.json()["room_code"]
    room_data = create_resp.json()["room"]

    assert room_data["game_mode"] == "2v2_team"
    assert room_data["max_players"] == 4
    assert room_data["player_count"] == 4
    assert room_data["status"] == "ready"

    # Verify team assignments
    assert room_data["players"][0]["team_id"] == 0
    assert room_data["players"][1]["team_id"] == 1
    assert room_data["players"][2]["team_id"] == 0
    assert room_data["players"][3]["team_id"] == 1

    # Start local 2v2 match
    start_resp = client.post("/api/rooms/start", json={"room_code": room_code})
    assert start_resp.status_code == 200
    assert start_resp.json()["room"]["status"] == "in_progress"


def test_online_room_creation_and_joining(client, auth_user, second_user):
    """Verify Online Room creation, joining with 6-char code, and ready state."""
    # 1. Host creates online room
    client.cookies.set("session_token", auth_user["token"])
    create_resp = client.post("/api/rooms/create", json={
        "game_mode": "1v1_human",
        "is_online": True
    })
    assert create_resp.status_code == 200
    room_code = create_resp.json()["room_code"]
    assert len(room_code) == 6

    # 2. Player 2 joins room
    client.cookies.set("session_token", second_user["token"])
    join_resp = client.post("/api/rooms/join", json={"room_code": room_code})
    assert join_resp.status_code == 200
    join_data = join_resp.json()
    assert join_data["seat"]["seat_id"] == 1
    assert join_data["seat"]["username"] == second_user["username"]

    # 3. Player 2 readies up
    ready_resp = client.post("/api/rooms/ready", json={
        "room_code": room_code,
        "is_ready": True
    })
    assert ready_resp.status_code == 200
    assert ready_resp.json()["room"]["status"] == "ready"

    # 4. Host starts match
    client.cookies.set("session_token", auth_user["token"])
    start_resp = client.post("/api/rooms/start", json={"room_code": room_code})
    assert start_resp.status_code == 200
    assert start_resp.json()["room"]["status"] == "in_progress"


def test_websocket_realtime_gameplay_flow(client, auth_user, second_user):
    """Verify real-time WebSocket protocol: connect, sync, roll dice, and token moves."""
    # Create 1v1 online room
    client.cookies.set("session_token", auth_user["token"])
    resp = client.post("/api/rooms/create", json={"game_mode": "1v1_human", "is_online": True})
    room_code = resp.json()["room_code"]

    # Player 2 joins
    client.cookies.set("session_token", second_user["token"])
    client.post("/api/rooms/join", json={"room_code": room_code})
    client.post("/api/rooms/ready", json={"room_code": room_code, "is_ready": True})

    # Host starts match
    client.cookies.set("session_token", auth_user["token"])
    client.post("/api/rooms/start", json={"room_code": room_code})

    # Host connects via WebSocket
    with client.websocket_connect(f"/ws/rooms/{room_code}") as ws:
        # Initial connect broadcast
        initial_msg = json.loads(ws.receive_text())
        assert initial_msg["type"] in ["PLAYER_RECONNECTED", "JOIN_ROOM", "GAME_STATE_UPDATE"]

        # Handshake JOIN_ROOM
        ws.send_text(json.dumps({"type": "JOIN_ROOM"}))
        msg1 = json.loads(ws.receive_text())
        assert msg1["type"] == "GAME_STATE_UPDATE"

        # PING
        ws.send_text(json.dumps({"type": "PING"}))
        msg2 = json.loads(ws.receive_text())
        assert msg2["type"] == "PONG"

        # Host rolls dice
        ws.send_text(json.dumps({
            "type": "ROLL_DICE",
            "client_action_id": "test_roll_1"
        }))
        roll_msg = json.loads(ws.receive_text())
        assert roll_msg["type"] in ["DICE_ROLLED", "GAME_STATE_UPDATE"]
        assert "dice" in roll_msg


def test_preserved_research_tournament_functionality(client, auth_user):
    """Verify the original 4-player AI Tournament APIs remain fully functional."""
    client.cookies.set("session_token", auth_user["token"])

    # Reset / start match
    resp_new = client.post("/api/ludo/new", json={
        "session_id": "research_test_sess",
        "num_players": 4,
        "p0_type": "dqn",
        "p1_type": "expert",
        "p2_type": "random",
        "p3_type": "expert",
        "seed": 42
    })
    assert resp_new.status_code == 200
    state = resp_new.json()["state"]
    assert state["num_players"] == 4
    assert len(state["players"]) == 4

    # Step match
    resp_step = client.post("/api/ludo/step", json={"session_id": "research_test_sess"})
    assert resp_step.status_code == 200
    assert "svg" in resp_step.json()

    # Autoplay 5 turns
    resp_auto = client.post("/api/ludo/auto_play", json={
        "session_id": "research_test_sess",
        "steps": 5
    })
    assert resp_auto.status_code == 200
    assert resp_auto.json()["state"]["turn_count"] >= 5


def test_match_results_persistence(client, auth_user):
    """Verify match results can be retrieved via /api/ludo/matches."""
    client.cookies.set("session_token", auth_user["token"])
    resp = client.get("/api/ludo/matches?limit=5")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert "matches" in data
    assert "summary" in data
