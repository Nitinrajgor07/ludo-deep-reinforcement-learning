"""Phase 3.1 Security & Reliability Audit Test Suite.

Verifies:
- Invalid and expired WebSocket sessions are rejected immediately
- Replayed dice-roll requests (same turn or duplicate action_id or stale sequence) are rejected
- Replayed token-move requests (moving twice with same dice or duplicate action_id) are rejected
- Concurrent simultaneous actions in one room are serialized without corruption
- Unauthorized seat access (acting out of turn or for other players) is rejected
- Reconnection to original seats preserves seat and does not create duplicate players
- Host migration transfers host correctly when host leaves
- Finished match rejects any further gameplay actions (rolls, moves, starts, ready toggles)
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import pytest
from fastapi.testclient import TestClient

from web.app import app
from web.services.room_service import (
    GameRoom,
    RoomStatus,
    TurnPhase,
    room_service,
)
from web.utils.auth import (
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
def audit_host():
    """Fixture for audit host user."""
    username = "audit_host_user"
    pwd = "AuditHostPass123!"
    existing = get_user_by_identifier(username)
    if not existing:
        u = register_user(username, pwd, email="audit_host@ludo.ai", full_name="Audit Host")
        uid = u["id"]
    else:
        uid = existing["id"]

    token = create_session(uid)
    yield {"id": uid, "username": username, "token": token}
    delete_session(token)


@pytest.fixture(scope="module")
def audit_guest():
    """Fixture for audit guest user."""
    username = "audit_guest_user"
    pwd = "AuditGuestPass123!"
    existing = get_user_by_identifier(username)
    if not existing:
        u = register_user(username, pwd, email="audit_guest@ludo.ai", full_name="Audit Guest")
        uid = u["id"]
    else:
        uid = existing["id"]

    token = create_session(uid)
    yield {"id": uid, "username": username, "token": token}
    delete_session(token)


@pytest.fixture(scope="module")
def audit_intruder():
    """Fixture for unauthorized user trying to tamper with rooms."""
    username = "audit_intruder_user"
    pwd = "AuditIntruderPass123!"
    existing = get_user_by_identifier(username)
    if not existing:
        u = register_user(username, pwd, email="audit_intruder@ludo.ai", full_name="Audit Intruder")
        uid = u["id"]
    else:
        uid = existing["id"]

    token = create_session(uid)
    yield {"id": uid, "username": username, "token": token}
    delete_session(token)


def test_invalid_and_expired_websocket_sessions(client, audit_host):
    """Verify WebSocket connection is rejected for missing, invalid, or expired sessions."""
    room = room_service.create_room("1v1_human", audit_host["id"], audit_host["username"])

    # 1. No token at all -> rejected
    with pytest.raises(Exception):
        with client.websocket_connect(f"/ws/rooms/{room.room_code}") as ws:
            ws.receive_json()

    # 2. Completely invalid token -> rejected
    with pytest.raises(Exception):
        with client.websocket_connect(f"/ws/rooms/{room.room_code}?token=INVALID_EXPIRED_TOKEN_9999") as ws:
            ws.receive_json()

    # 3. Deleted/expired session token -> rejected
    temp_token = create_session(audit_host["id"])
    delete_session(temp_token)  # Invalidate session
    with pytest.raises(Exception):
        with client.websocket_connect(f"/ws/rooms/{room.room_code}?token={temp_token}") as ws:
            ws.receive_json()


def test_cookie_based_websocket_authentication(client, audit_host):
    """Verify WebSocket connects cleanly using secure session cookie without query token."""
    room = room_service.create_room("1v1_human", audit_host["id"], audit_host["username"])
    client.cookies.set("session_token", audit_host["token"])

    with client.websocket_connect(f"/ws/rooms/{room.room_code}") as ws:
        msg = ws.receive_json()
        assert msg["type"] in ["JOIN_ROOM", "PLAYER_RECONNECTED"]
        assert msg["username"] == audit_host["username"]


def test_unauthorized_room_joining_active_match(client, audit_host, audit_guest, audit_intruder):
    """Verify non-participant cannot connect to or join an active match."""
    # Create and start 1v1 match between host and guest
    room = room_service.create_room("1v1_human", audit_host["id"], audit_host["username"])
    room_service.join_room(room.room_code, audit_guest["id"], audit_guest["username"])
    room.set_player_ready(audit_guest["id"], True)
    room.start_game(audit_host["id"])
    assert room.status == RoomStatus.IN_PROGRESS

    # Intruder attempts to connect to the ongoing match via WebSocket -> rejected!
    client.cookies.set("session_token", audit_intruder["token"])
    with pytest.raises(Exception):
        with client.websocket_connect(f"/ws/rooms/{room.room_code}") as ws:
            ws.receive_json()


def test_replayed_dice_roll_rejection(audit_host, audit_guest):
    """Verify rolling dice twice in the same turn or reusing action IDs is rejected."""
    room = room_service.create_room("1v1_human", audit_host["id"], audit_host["username"])
    room_service.join_room(room.room_code, audit_guest["id"], audit_guest["username"])
    room.set_player_ready(audit_guest["id"], True)
    room.start_game(audit_host["id"])

    # First roll succeeds
    res1 = room.roll_dice(audit_host["id"], client_action_id="act_roll_001")
    assert res1["dice"] in [1, 2, 3, 4, 5, 6]

    # Replay 1: Attempting to roll again in same turn (when moves are available)
    if room.turn_phase == TurnPhase.NEED_MOVE:
        with pytest.raises(ValueError, match="already been rolled"):
            room.roll_dice(audit_host["id"], client_action_id="act_roll_002")

    # Replay 2: Reusing duplicate action ID 'act_roll_001'
    with pytest.raises(ValueError, match="Duplicate or replayed action"):
        room.roll_dice(audit_host["id"], client_action_id="act_roll_001")


def test_stale_sequence_rejection(audit_host, audit_guest):
    """Verify actions with stale expected_action_seq are rejected."""
    room = room_service.create_room("1v1_human", audit_host["id"], audit_host["username"])
    room_service.join_room(room.room_code, audit_guest["id"], audit_guest["username"])
    room.set_player_ready(audit_guest["id"], True)
    room.start_game(audit_host["id"])

    current_seq = room.action_seq

    # Submit action with stale sequence (current_seq + 99) -> rejected!
    with pytest.raises(ValueError, match="Stale action rejected"):
        room.roll_dice(audit_host["id"], expected_action_seq=current_seq + 99)


def test_replayed_token_move_rejection(audit_host, audit_guest):
    """Verify moving twice with same dice result or without rolling is rejected."""
    room = room_service.create_room("1v1_human", audit_host["id"], audit_host["username"])
    room_service.join_room(room.room_code, audit_guest["id"], audit_guest["username"])
    room.set_player_ready(audit_guest["id"], True)
    room.start_game(audit_host["id"])

    # Move token before rolling dice -> rejected!
    with pytest.raises(ValueError, match="Must roll dice before moving"):
        room.move_token(audit_host["id"], token_id=0)

    # Roll dice until legal actions are available
    rolled = False
    for _ in range(50):
        if room.turn_phase == TurnPhase.NEED_ROLL:
            curr_uid = room.seats[room.env.current_player].user_id
            res = room.roll_dice(curr_uid)
            if len(res.get("legal_actions", [])) > 0:
                rolled = True
                active_uid = curr_uid
                valid_token = res["legal_actions"][0]
                break

    if rolled:
        # Move token
        move_res = room.move_token(active_uid, token_id=valid_token, client_action_id="act_move_001")
        assert move_res is not None

        # Replay 1: Attempt to move AGAIN with the same dice result -> rejected!
        with pytest.raises(ValueError, match="(Must roll dice|already finished)"):
            room.move_token(active_uid, token_id=valid_token, client_action_id="act_move_002")

        # Replay 2: Reusing duplicate action ID 'act_move_001' -> rejected!
        with pytest.raises(ValueError, match="Duplicate or replayed action"):
            room.move_token(active_uid, token_id=valid_token, client_action_id="act_move_001")


def test_concurrent_actions_synchronization(audit_host, audit_guest):
    """Verify concurrent action execution does not corrupt room state."""
    room = room_service.create_room("1v1_human", audit_host["id"], audit_host["username"])
    room_service.join_room(room.room_code, audit_guest["id"], audit_guest["username"])
    room.set_player_ready(audit_guest["id"], True)
    room.start_game(audit_host["id"])

    # Launch two simultaneous roll requests from threads
    results = []
    errors = []

    def try_roll(uid, act_id):
        try:
            r = room.roll_dice(uid, client_action_id=act_id)
            results.append(r)
        except Exception as e:
            errors.append(str(e))

    with ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(try_roll, audit_host["id"], "thread_act_1")
        f2 = executor.submit(try_roll, audit_host["id"], "thread_act_2")
        f1.result()
        f2.result()

    # Exactly one must succeed, while the other is safely rejected by the state machine
    assert len(results) == 1, "Only one concurrent roll should succeed"
    assert len(errors) == 1, "The conflicting roll must be rejected safely"
    assert any("already been rolled" in err or "turn" in err for err in errors)


def test_reconnection_preserves_seat_without_duplicates(audit_host, audit_guest):
    """Verify reconnecting maintains original seat and never creates duplicate players."""
    room = room_service.create_room("1v1_human", audit_host["id"], audit_host["username"])
    _, seat1 = room_service.join_room(room.room_code, audit_guest["id"], audit_guest["username"])
    assert seat1.seat_id == 1
    assert room.get_player_count() == 2

    # Simulate disconnect
    seat1.is_connected = False

    # Guest reconnects
    _, seat1_re = room_service.join_room(room.room_code, audit_guest["id"], audit_guest["username"])
    assert seat1_re.seat_id == 1
    assert seat1_re.is_connected is True
    assert room.get_player_count() == 2, "Reconnection must not duplicate player seats"


def test_host_migration_robustness(audit_host, audit_guest):
    """Verify host ownership transfers correctly when host leaves."""
    room = room_service.create_room("1v1_human", audit_host["id"], audit_host["username"])
    room_service.join_room(room.room_code, audit_guest["id"], audit_guest["username"])
    assert room.host_user_id == audit_host["id"]

    # Host leaves room
    vacated = room.leave(audit_host["id"])
    assert vacated == 0
    assert room.seats[0] is None
    # Guest is now the host
    assert room.host_user_id == audit_guest["id"]
    assert room.host_username == audit_guest["username"]


def test_finished_match_rejects_further_actions(audit_host, audit_guest):
    """Verify that once a match is finished, all gameplay actions are rejected."""
    room = room_service.create_room("1v1_human", audit_host["id"], audit_host["username"])
    room_service.join_room(room.room_code, audit_guest["id"], audit_guest["username"])
    room.set_player_ready(audit_guest["id"], True)
    room.start_game(audit_host["id"])

    # Manually transition room to FINISHED
    room.status = RoomStatus.FINISHED
    room.turn_phase = TurnPhase.FINISHED

    # Attempt to roll -> rejected
    with pytest.raises(ValueError, match="already finished"):
        room.roll_dice(audit_host["id"])

    # Attempt to move -> rejected
    with pytest.raises(ValueError, match="already finished"):
        room.move_token(audit_host["id"], token_id=0)

    # Attempt to ready up -> rejected
    with pytest.raises(ValueError, match="already finished"):
        room.set_player_ready(audit_host["id"], True)

    # Attempt to start game again -> rejected
    with pytest.raises(ValueError, match="already finished"):
        room.start_game(audit_host["id"])
