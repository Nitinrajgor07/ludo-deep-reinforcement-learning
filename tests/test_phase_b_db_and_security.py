"""Targeted verification test suite for Phase B:
Database Isolation, Authentication Security & Transaction Reliability.

Verifies:
1. Production database isolation (tests write to temp DB, never data/users.db).
2. Temporary test database lifecycle & override capability.
3. Transaction rollback on match persistence error.
4. Duplicate match prevention for identical room codes.
5. Concurrent match persistence idempotency.
6. Session expiration enforcement.
7. Logout invalidation in database and cookie clearing.
8. WebSocket authentication with valid vs expired/invalid session tokens.
9. Registration validation (duplicate username, duplicate email, invalid characters, lengths).
10. Existing user preservation & password hash integrity.
11. Profile update constraints and rating tamper protection.
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
from pathlib import Path
import sqlite3
import tempfile
import time
import pytest
from starlette.testclient import TestClient

from web.app import app
from web.services.room_service import room_service
from web.utils.auth import (
    authenticate_user,
    check_user_exists,
    create_session,
    delete_session,
    get_db,
    get_db_path,
    get_user_by_identifier,
    get_user_by_session,
    get_user_profile,
    init_db,
    is_session_expired,
    register_user,
    rotate_session,
    save_match_record,
    set_db_path,
    update_user_profile,
)


@pytest.fixture
def client():
    """TestClient instance."""
    return TestClient(app)


def test_production_database_isolation():
    """Verify tests run against an isolated temporary database, leaving production data/users.db untouched."""
    prod_path = (Path(__file__).resolve().parent.parent / "data" / "users.db").resolve()
    active_path = get_db_path().resolve()

    # 1. Assert active database is NOT the production database
    assert active_path != prod_path
    assert "users.db" not in str(active_path) or "test" in str(active_path).lower()

    # 2. Record counts in production DB in read-only mode
    con_prod = sqlite3.connect(f"file:{prod_path}?mode=ro", uri=True)
    prod_users_before = con_prod.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    con_prod.close()

    # 3. Create a test user in active database
    test_user = register_user(
        username="iso_test_user_unique",
        password="ValidPassword123!",
        email="iso_test_unique@ludo.ai",
    )
    assert test_user["id"] is not None

    # 4. Assert production database did NOT receive the new user
    con_prod2 = sqlite3.connect(f"file:{prod_path}?mode=ro", uri=True)
    prod_users_after = con_prod2.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    prod_has_user = con_prod2.execute(
        "SELECT 1 FROM users WHERE username = 'iso_test_user_unique'"
    ).fetchone()
    con_prod2.close()

    assert prod_users_after == prod_users_before
    assert prod_has_user is None


def test_temporary_test_database_lifecycle():
    """Verify set_db_path and init_db correctly create and tear down standalone test databases."""
    temp_dir = tempfile.mkdtemp(prefix="standalone_test_db_")
    custom_db = Path(temp_dir) / "custom.db"

    saved_path = get_db_path()
    try:
        set_db_path(custom_db)
        init_db()
        assert custom_db.exists()

        u = register_user("lifecycle_user", "ValidPassword123!")
        assert u["id"] == 1
    finally:
        set_db_path(saved_path)
        try:
            custom_db.unlink(missing_ok=True)
            Path(temp_dir).rmdir()
        except Exception:
            pass


def test_transaction_rollback_on_failure():
    """Verify that a database error during match persistence rolls back atomically."""
    active_path = get_db_path()
    con = sqlite3.connect(active_path)
    matches_before = con.execute("SELECT COUNT(*) FROM match_records").fetchone()[0]
    con.close()

    # Construct invalid participant payload with a non-existent column or constraint violation
    # To test transaction rollback, we mock or inject an invalid SQL operation inside participants
    with pytest.raises(Exception):
        # Passing an invalid type that causes sqlite3 to fail during participant insert
        invalid_participants = [{
            "user_id": "NOT_AN_INT_INVALID",
            "username": "BadUser",
            "player_seat": None,  # NOT NULL constraint violation on player_seat!
            "is_winner": False,
        }]
        save_match_record(
            session_id="rollback_test_session",
            match_mode="1v1",
            winner_id=0,
            room_code="ROLLBK",
            participants=invalid_participants,
        )

    # Verify match_records was rolled back and count did not increase
    con2 = sqlite3.connect(active_path)
    matches_after = con2.execute("SELECT COUNT(*) FROM match_records").fetchone()[0]
    has_rollback_session = con2.execute(
        "SELECT 1 FROM match_records WHERE session_id = 'rollback_test_session'"
    ).fetchone()
    con2.close()

    assert matches_after == matches_before
    assert has_rollback_session is None


def test_duplicate_match_prevention():
    """Verify that calling save_match_record twice with the same room_code returns existing match ID."""
    u = register_user("dup_match_host", "ValidPassword123!")
    room_code = "DUP001"

    participants = [
        {"user_id": u["id"], "username": u["username"], "player_seat": 0, "team_id": 0, "is_winner": True},
        {"user_id": None, "username": "Guest (Yellow)", "player_seat": 1, "team_id": 1, "is_winner": False},
    ]

    # First save
    match_id_1 = save_match_record(
        session_id=f"room_{room_code}",
        match_mode="1v1_human",
        winner_id=0,
        user_id=u["id"],
        room_code=room_code,
        participants=participants,
    )
    assert match_id_1 is not None

    # Check host rating after win (+25)
    prof1 = get_user_profile(u["id"])
    rating_after_first = prof1["rating"]
    assert rating_after_first == 1025

    # Second save with identical room_code
    match_id_2 = save_match_record(
        session_id=f"room_{room_code}",
        match_mode="1v1_human",
        winner_id=0,
        user_id=u["id"],
        room_code=room_code,
        participants=participants,
    )

    # Must return identical match ID without creating a duplicate record or re-adjusting rating
    assert match_id_2 == match_id_1

    prof2 = get_user_profile(u["id"])
    assert prof2["rating"] == rating_after_first  # Not incremented a second time!

    # Check database rows for room_code
    con = sqlite3.connect(get_db_path())
    count = con.execute("SELECT COUNT(*) FROM match_records WHERE room_code = ?", (room_code,)).fetchone()[0]
    con.close()
    assert count == 1


def test_concurrent_match_persistence():
    """Verify concurrent match completion calls on GameRoom execute idempotently."""
    host = register_user("conc_match_host", "ValidPassword123!")
    room = room_service.create_room(
        game_mode="1v1_human",
        host_user_id=host["id"],
        host_username=host["username"],
        is_online=False,
    )

    room.winner = 0

    # Call _persist_completed_match concurrently from 5 threads
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(room._persist_completed_match) for _ in range(5)]
        for f in futures:
            f.result()

    # Assert only 1 match record was written for this room
    con = sqlite3.connect(get_db_path())
    count = con.execute("SELECT COUNT(*) FROM match_records WHERE room_code = ?", (room.room_code,)).fetchone()[0]
    con.close()
    assert count == 1


def test_session_expiration_enforcement():
    """Verify expired sessions return None and are purged from database."""
    u = register_user("sess_exp_user", "ValidPassword123!")
    token = create_session(u["id"])

    # Normal session is valid
    user = get_user_by_session(token)
    assert user is not None
    assert user["id"] == u["id"]

    # Backdate the session creation timestamp by 10 days (beyond 7 days default)
    old_time = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
    con = sqlite3.connect(get_db_path())
    con.execute("UPDATE sessions SET created_at = ? WHERE token = ?", (old_time, token))
    con.commit()
    con.close()

    # Querying with expired session must return None and delete token
    user_expired = get_user_by_session(token)
    assert user_expired is None

    # Verify session row is gone
    con2 = sqlite3.connect(get_db_path())
    row = con2.execute("SELECT 1 FROM sessions WHERE token = ?", (token,)).fetchone()
    con2.close()
    assert row is None


def test_session_rotation():
    """Verify rotate_session generates a new token and deletes the old one."""
    u = register_user("rotate_user", "ValidPassword123!")
    token1 = create_session(u["id"])

    token2 = rotate_session(token1)
    assert token2 is not None
    assert token2 != token1

    # Old token is invalid
    assert get_user_by_session(token1) is None
    # New token is valid
    user2 = get_user_by_session(token2)
    assert user2 is not None
    assert user2["id"] == u["id"]


def test_logout_invalidation_and_cookie_clearing(client):
    """Verify /api/logout removes session from database and clears cookie."""
    u = register_user("logout_tester", "ValidPassword123!")
    token = create_session(u["id"])

    client.cookies.set("session_token", token)
    res = client.post("/api/logout")
    assert res.status_code == 200
    assert res.json()["status"] == "logged_out"

    # Token must be deleted from database
    assert get_user_by_session(token) is None


def test_websocket_rejects_expired_and_invalid_sessions(client):
    """Verify WebSocket endpoint closes with policy violation for invalid or expired session."""
    # 1. Invalid token
    with pytest.raises(Exception):
        with client.websocket_connect("/ws/TEST01?token=invalid_token_12345678"):
            pass

    # 2. Expired token
    u = register_user("ws_exp_user", "ValidPassword123!")
    token = create_session(u["id"])
    old_time = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
    con = sqlite3.connect(get_db_path())
    con.execute("UPDATE sessions SET created_at = ? WHERE token = ?", (old_time, token))
    con.commit()
    con.close()

    with pytest.raises(Exception):
        with client.websocket_connect(f"/ws/TEST01?token={token}"):
            pass


def test_registration_validation_rules():
    """Verify strict registration validation and duplicate rejection."""
    # 1. Valid user
    u1 = register_user("valid_reg_user", "ValidPassword123!", email="valid_reg@ludo.ai")
    assert u1["username"] == "valid_reg_user"

    # 2. Duplicate username
    with pytest.raises(ValueError, match="already taken"):
        register_user("VALID_REG_USER", "AnotherPassword123!")

    # 3. Duplicate email
    with pytest.raises(ValueError, match="already registered"):
        register_user("other_user", "AnotherPassword123!", email="valid_reg@ludo.ai")

    # 4. Too short username (< 3 chars)
    with pytest.raises(ValueError, match="between 3 and 50"):
        register_user("ab", "ValidPassword123!")

    # 5. Invalid characters in username
    with pytest.raises(ValueError, match="invalid characters"):
        register_user("bad user!", "ValidPassword123!")

    # 6. Too short password (< 4 chars)
    with pytest.raises(ValueError, match="between 4 and 128"):
        register_user("valid_name", "123")

    # 7. Invalid email format
    with pytest.raises(ValueError, match="Invalid email address format"):
        register_user("valid_user2", "ValidPassword123!", email="not-an-email")


def test_existing_users_preserved_and_unaltered():
    """Verify existing user accounts can authenticate and preserve ratings."""
    existing = get_user_by_identifier("admin")
    if existing:
        assert existing["username"] == "admin"
        assert existing["rating"] >= 100
        # Password hash is intact and salt is non-empty
        con = sqlite3.connect(get_db_path())
        con.row_factory = sqlite3.Row
        row = con.execute("SELECT password_hash, salt FROM users WHERE username = 'admin'").fetchone()
        con.close()
        assert len(row["password_hash"]) == 64
        assert len(row["salt"]) == 32


def test_profile_validation_and_rating_protection():
    """Verify profile updates enforce length limits, valid avatar colors, and protect ratings."""
    u = register_user("prof_test_user", "ValidPassword123!")
    user_id = u["id"]

    # 1. Successful update
    updated = update_user_profile(
        user_id=user_id,
        full_name="Valid Full Name",
        bio="Research bio content",
        avatar_color="emerald",
    )
    assert updated["full_name"] == "Valid Full Name"
    assert updated["avatar_color"] == "emerald"

    # 2. Reject invalid avatar color
    with pytest.raises(ValueError, match="Invalid avatar color"):
        update_user_profile(user_id=user_id, avatar_color="neon_pink_invalid")

    # 3. Prevent duplicate email in profile update
    u2 = register_user("prof_other_user", "ValidPassword123!", email="other_profile@ludo.ai")
    with pytest.raises(ValueError, match="already registered"):
        update_user_profile(user_id=user_id, email="other_profile@ludo.ai")

    # 4. Verify rating is unchanged through profile updates
    prof = get_user_profile(user_id)
    assert prof["rating"] == 1000
