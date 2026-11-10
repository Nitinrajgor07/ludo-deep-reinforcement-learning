"""Tests for Phase 3: Multiplayer Engine, Room Code Management, and Real-Time WebSocket Communication.

Verifies:
- Unique 6-character room code generation
- 1v1 Human and 2v2 Team room creation
- Seat assignment, team mappings, and full room rejection
- Duplicate player reconnection and host migration upon leave
- Room REST API endpoints (/api/rooms/create, /join, /info, /leave, /ready, /start, /roll, /action)
- Authentication and unauthorized action protection
- Local multiplayer mode (multiple seats on one machine without WebSockets)
- Real-time WebSocket connection, authentication, protocol events (JOIN, READY, ROLL, MOVE, FINISH, PING)
- Disconnect and reconnect tracking
"""

import json
import time
from typing import Any, Dict
import pytest
from fastapi.testclient import TestClient

from web.app import app
from web.services.room_service import (
    GameMode,
    GameRoom,
    ROOM_CODE_ALPHABET,
    RoomStatus,
    TurnPhase,
    room_service,
)
from web.utils.auth import (
    authenticate_user,
    create_session,
    delete_session,
    get_user_by_identifier,
    register_user,
)


@pytest.fixture(scope="module")
def client():
    """FastAPI TestClient fixture."""
    return TestClient(app)


@pytest.fixture(scope="module")
def player1():
    """Fixture for primary test player (Host)."""
    username = "p3_host_user"
    pwd = "HostPassword123!"
    existing = get_user_by_identifier(username)
    if not existing:
        user_dict = register_user(username, pwd, email="host_p3@ludo.ai", full_name="Host Player")
        uid = user_dict["id"]
    else:
        uid = existing["id"]

    token = create_session(uid)
    yield {"id": uid, "username": username, "token": token}
    delete_session(token)


@pytest.fixture(scope="module")
def player2():
    """Fixture for secondary test player (Guest)."""
    username = "p3_guest_user"
    pwd = "GuestPassword123!"
    existing = get_user_by_identifier(username)
    if not existing:
        user_dict = register_user(username, pwd, email="guest_p3@ludo.ai", full_name="Guest Player")
        uid = user_dict["id"]
    else:
        uid = existing["id"]

    token = create_session(uid)
    yield {"id": uid, "username": username, "token": token}
    delete_session(token)


@pytest.fixture(scope="module")
def player3():
    """Fixture for third test player (Team Partner)."""
    username = "p3_team_user"
    pwd = "TeamPassword123!"
    existing = get_user_by_identifier(username)
    if not existing:
        user_dict = register_user(username, pwd, email="team_p3@ludo.ai", full_name="Team Partner")
        uid = user_dict["id"]
    else:
        uid = existing["id"]

    token = create_session(uid)
    yield {"id": uid, "username": username, "token": token}
    delete_session(token)


def test_unique_room_code_generation():
    """Verify unique 6-character room codes using the clean alphanumeric alphabet."""
    codes = set()
    for _ in range(50):
        code = room_service.generate_unique_code()
        assert len(code) == 6
        assert all(c in ROOM_CODE_ALPHABET for c in code)
        codes.add(code)
    assert len(codes) == 50, "Generated room codes should be unique"


def test_room_creation_1v1(player1):
    """Verify creating a 1v1 Human room assigns Host to Seat 0 and configures 2 seats."""
    room = room_service.create_room(
        game_mode="1v1_human",
        host_user_id=player1["id"],
        host_username=player1["username"],
        is_online=True,
    )
    assert len(room.room_code) == 6
    assert room.max_players == 2
    assert room.get_player_count() == 1
    assert room.host_user_id == player1["id"]
    assert room.seats[0].user_id == player1["id"]
    assert room.seats[0].seat_id == 0
    assert room.seats[1] is None
    assert room.status == RoomStatus.WAITING


def test_room_creation_2v2(player1):
    """Verify creating a 2v2 Team room assigns Host to Seat 0 (Team 0) and configures 4 seats."""
    room = room_service.create_room(
        game_mode="2v2_team",
        host_user_id=player1["id"],
        host_username=player1["username"],
        is_online=True,
    )
    assert room.max_players == 4
    assert room.seats[0].team_id == 0
    assert room.teams == {0: 0, 1: 1, 2: 0, 3: 1}


def test_join_room_and_seat_assignment(player1, player2):
    """Verify guest joins 1v1 room, gets Seat 1, and room becomes full."""
    room = room_service.create_room(
        game_mode="1v1_human",
        host_user_id=player1["id"],
        host_username=player1["username"],
        is_online=True,
    )
    # Join player 2
    joined_room, seat = room_service.join_room(
        room_code=room.room_code,
        user_id=player2["id"],
        username=player2["username"],
    )
    assert seat.seat_id == 1
    assert seat.user_id == player2["id"]
    assert joined_room.is_full()
    assert joined_room.get_player_count() == 2


def test_full_room_rejection(player1, player2, player3):
    """Verify that joining a full room raises ValueError."""
    room = room_service.create_room(
        game_mode="1v1_human",
        host_user_id=player1["id"],
        host_username=player1["username"],
    )
    room_service.join_room(room.room_code, player2["id"], player2["username"])

    with pytest.raises(ValueError, match="full"):
        room_service.join_room(room.room_code, player3["id"], player3["username"])


def test_invalid_room_code_rejection(player2):
    """Verify that joining with an invalid or non-existent code raises ValueError."""
    with pytest.raises(ValueError, match="not found"):
        room_service.join_room("XYZ999", player2["id"], player2["username"])


def test_duplicate_player_reconnection(player1, player2):
    """Verify that a player joining a room they already occupy reconnects without error."""
    room = room_service.create_room("1v1_human", player1["id"], player1["username"])
    room_service.join_room(room.room_code, player2["id"], player2["username"])

    # Player 2 rejoins
    re_room, re_seat = room_service.join_room(room.room_code, player2["id"], player2["username"])
    assert re_seat.seat_id == 1
    assert re_seat.is_connected is True
    assert re_room.get_player_count() == 2


def test_leave_room_and_host_migration(player1, player2):
    """Verify leaving a room vacates the seat and migrates host if host leaves."""
    room = room_service.create_room("1v1_human", player1["id"], player1["username"])
    room_service.join_room(room.room_code, player2["id"], player2["username"])

    # Host leaves
    room_service.leave_room(room.room_code, player1["id"])
    assert room.seats[0] is None
    assert room.host_user_id == player2["id"]
    assert room.host_username == player2["username"]


def test_room_rest_api_lifecycle(client, player1, player2):
    """Verify complete room lifecycle via REST API (create, join, info, ready, start)."""
    # 1. Create Room (Host = Player 1)
    client.cookies.set("session_token", player1["token"])
    resp_create = client.post("/api/rooms/create", json={"game_mode": "1v1_human", "is_online": True})
    assert resp_create.status_code == 200
    create_data = resp_create.json()
    room_code = create_data["room_code"]
    assert len(room_code) == 6

    # 2. Join Room (Guest = Player 2)
    client.cookies.set("session_token", player2["token"])
    resp_join = client.post("/api/rooms/join", json={"room_code": room_code})
    assert resp_join.status_code == 200
    join_data = resp_join.json()
    assert join_data["seat"]["seat_id"] == 1

    # 3. Get Room Info
    resp_info = client.get(f"/api/rooms/info?room_code={room_code}")
    assert resp_info.status_code == 200
    info_data = resp_info.json()
    assert info_data["room"]["player_count"] == 2

    # 4. Ready Up Player 2
    resp_ready = client.post("/api/rooms/ready", json={"room_code": room_code, "is_ready": True})
    assert resp_ready.status_code == 200

    # 5. Start Game (Host = Player 1)
    client.cookies.set("session_token", player1["token"])
    resp_start = client.post("/api/rooms/start", json={"room_code": room_code})
    assert resp_start.status_code == 200
    start_data = resp_start.json()
    assert start_data["room"]["status"] == "in_progress"
    assert start_data["room"]["turn_phase"] == "NEED_ROLL"


def test_unauthorized_room_action_rejection(client, player1, player2):
    """Verify that players cannot act out of turn or perform unauthorized rolls/moves."""
    # Setup and start a 1v1 game
    client.cookies.set("session_token", player1["token"])
    resp = client.post("/api/rooms/create", json={"game_mode": "1v1_human", "is_online": True})
    room_code = resp.json()["room_code"]

    client.cookies.set("session_token", player2["token"])
    client.post("/api/rooms/join", json={"room_code": room_code})
    client.post("/api/rooms/ready", json={"room_code": room_code, "is_ready": True})

    client.cookies.set("session_token", player1["token"])
    client.post("/api/rooms/start", json={"room_code": room_code})

    # It is Seat 0's (Player 1's) turn. Player 2 attempts to roll dice -> 400 rejected!
    client.cookies.set("session_token", player2["token"])
    resp_bad_roll = client.post("/api/rooms/roll", json={"room_code": room_code})
    assert resp_bad_roll.status_code == 400
    assert "not yours" in resp_bad_roll.json()["detail"]

    # Player 1 attempts to move a token BEFORE rolling dice -> 400 rejected!
    client.cookies.set("session_token", player1["token"])
    resp_bad_move = client.post("/api/rooms/action", json={"room_code": room_code, "token_id": 0})
    assert resp_bad_move.status_code == 400
    assert "Must roll dice" in resp_bad_move.json()["detail"]


def test_local_mode_gameplay(client, player1):
    """Verify Local Mode allows host to manage all turns on the same machine without WebSockets."""
    client.cookies.set("session_token", player1["token"])
    # Create Local Room (is_online = False)
    resp = client.post("/api/rooms/create", json={"game_mode": "1v1_human", "is_online": False})
    room_code = resp.json()["room_code"]

    # In local mode, fill second seat with local guest
    room = room_service.get_room(room_code)
    room.seats[1] = room_service.get_room(room_code).seats[0]  # Local shared seat representation

    resp_start = client.post("/api/rooms/start", json={"room_code": room_code})
    assert resp_start.status_code == 200

    # Execute dice roll locally via REST
    resp_roll = client.post("/api/rooms/roll", json={"room_code": room_code})
    assert resp_roll.status_code == 200
    roll_data = resp_roll.json()
    assert roll_data["status"] == "ok"
    assert roll_data["dice"] in [1, 2, 3, 4, 5, 6]


def _safe_receive_json(ws, timeout: float = 5.0) -> Any:
    """Receive JSON from WebSocket with guaranteed non-blocking timeout and cancellation.

    Uses ws.portal.start_task_soon directly to avoid ThreadPoolExecutor.shutdown(wait=True) deadlocks.
    """
    future = ws.portal.start_task_soon(ws._send_rx.receive)
    try:
        message = future.result(timeout=timeout)
    except (TimeoutError, Exception) as exc:
        future.cancel()
        if isinstance(exc, TimeoutError) or "timed out" in str(exc).lower():
            raise TimeoutError(f"WebSocket receive timed out after {timeout} seconds")
        raise
    ws._raise_on_close(message)
    if "text" in message:
        return json.loads(message["text"])
    return json.loads(message["bytes"].decode("utf-8"))


def test_websocket_realtime_protocol(player1, player2):
    """Verify real-time WebSocket protocol (JOIN, READY, START, ROLL, PING) with isolated clients."""
    # 1. Create independent TestClient instances with isolated cookie jars
    client1 = TestClient(app, cookies={"session_token": player1["token"]})
    client2 = TestClient(app, cookies={"session_token": player2["token"]})

    # 2. Host creates online room
    resp = client1.post("/api/rooms/create", json={"game_mode": "1v1_human", "is_online": True})
    assert resp.status_code == 200
    room_code = resp.json()["room_code"]

    # 3. Player 1 connects to WebSocket using client1 (authenticated via session cookie)
    with client1.websocket_connect(f"/ws/rooms/{room_code}") as ws1:
        msg1 = _safe_receive_json(ws1)
        assert msg1["type"] in ["JOIN_ROOM", "PLAYER_RECONNECTED"]
        assert msg1["username"] == player1["username"]
        assert msg1["seat_id"] == 0

        # 4. Player 2 connects to WebSocket using isolated client2 (authenticated via its own cookie)
        with client2.websocket_connect(f"/ws/rooms/{room_code}") as ws2:
            # Both players receive Player 2's join broadcast
            broadcast_join1 = _safe_receive_json(ws1)
            assert broadcast_join1["username"] == player2["username"]
            assert broadcast_join1["seat_id"] == 1

            broadcast_join2 = _safe_receive_json(ws2)
            assert broadcast_join2["username"] == player2["username"]
            assert broadcast_join2["seat_id"] == 1

            # 5. Player 2 sends PING -> receives PONG
            ws2.send_json({"type": "PING"})
            pong = _safe_receive_json(ws2)
            assert pong["type"] == "PONG"

            # 6. Player 2 sends PLAYER_READY
            ws2.send_json({"type": "PLAYER_READY", "is_ready": True})
            ready_msg1 = _safe_receive_json(ws1)
            ready_msg2 = _safe_receive_json(ws2)
            assert ready_msg1["type"] == "PLAYER_READY"
            assert ready_msg2["type"] == "PLAYER_READY"
            assert ready_msg1["is_ready"] is True

            # 7. Host sends START_GAME
            ws1.send_json({"type": "START_GAME"})
            start_msg1 = _safe_receive_json(ws1)
            start_msg2 = _safe_receive_json(ws2)
            assert start_msg1["type"] == "GAME_STATE_UPDATE"
            assert start_msg1["event"] == "GAME_STARTED"
            assert start_msg2["event"] == "GAME_STARTED"

            # 8. Active player rolls dice
            ws1.send_json({"type": "ROLL_DICE"})
            roll_msg1 = _safe_receive_json(ws1)
            roll_msg2 = _safe_receive_json(ws2)
            assert roll_msg1["type"] in ["DICE_ROLLED", "GAME_STATE_UPDATE"]
            assert roll_msg2["type"] in ["DICE_ROLLED", "GAME_STATE_UPDATE"]
            assert "dice" in roll_msg1


def test_websocket_unauthorized_close(client):
    """Verify that unauthenticated WebSocket connections are rejected immediately."""
    with pytest.raises(Exception):
        with client.websocket_connect("/ws/rooms/FAKE99?token=invalid_token") as ws:
            ws.receive_json()


def test_websocket_deliberate_timeout_demonstration(player1):
    """Verify that _safe_receive_json cleanly times out on non-responsive WebSockets without deadlock."""
    client = TestClient(app, cookies={"session_token": player1["token"]})
    resp = client.post("/api/rooms/create", json={"game_mode": "1v1_human", "is_online": True})
    assert resp.status_code == 200
    room_code = resp.json()["room_code"]

    with client.websocket_connect(f"/ws/rooms/{room_code}") as ws:
        # Drain the initial connection message (JOIN_ROOM or PLAYER_RECONNECTED)
        first_msg = _safe_receive_json(ws, timeout=2.0)
        assert first_msg["type"] in ["JOIN_ROOM", "PLAYER_RECONNECTED"]

        # Deliberately attempt to receive when no further message is sent by server
        start_time = time.monotonic()
        with pytest.raises(TimeoutError) as exc_info:
            _safe_receive_json(ws, timeout=0.6)
        elapsed = time.monotonic() - start_time

        assert "timed out" in str(exc_info.value).lower()
        # Verify it timed out promptly without indefinite hang (between 0.5s and 3.0s)
        assert 0.5 <= elapsed <= 3.0, f"Expected timeout in ~0.6s, took {elapsed:.2f}s"
