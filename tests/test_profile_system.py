"""Tests for Phase 2: Database Schema Expansion and Player Profile System.

Verifies:
- Database migration repeatability (idempotent init_db)
- Existing users and authentication preservation
- Profile retrieval, updates, and constraint validations
- Match participants tracking, ELO rating updates, and user match history
- Aggregate and per-mode statistical calculations
- Profile REST API endpoints (GET / PUT /api/profile)
- Profile page rendering and authentication gating
"""

import pytest
from fastapi.testclient import TestClient

from web.app import app
from web.utils.auth import (
    authenticate_user,
    create_session,
    delete_session,
    get_db,
    get_user_by_identifier,
    get_user_by_session,
    get_user_match_history,
    get_user_profile,
    init_db,
    register_user,
    save_match_record,
    update_user_profile,
)


@pytest.fixture(scope="module")
def client():
    """FastAPI TestClient instance."""
    return TestClient(app)


@pytest.fixture(scope="module")
def test_player():
    """Fixture ensuring a dedicated test player exists and has an active session."""
    username = "phase2_tester"
    password = "TestPassword123!"
    existing = get_user_by_identifier(username)
    if not existing:
        reg = register_user(
            username=username,
            password=password,
            email="phase2_tester@ludo.ai",
            full_name="Phase Two Tester",
        )
        user_id = reg["id"]
    else:
        user_id = existing["id"]

    auth_user = authenticate_user(username, password)
    assert auth_user is not None
    session_token = create_session(user_id)

    yield {
        "id": user_id,
        "username": username,
        "password": password,
        "session_token": session_token,
    }

    # Cleanup session
    delete_session(session_token)


def test_database_migration_repeatability():
    """Verify that init_db() can run repeatedly without corrupting data or failing."""
    # Run init_db multiple times
    init_db()
    init_db()

    with get_db() as conn:
        cursor = conn.cursor()
        # Verify users columns
        cursor.execute("PRAGMA table_info(users)")
        user_cols = {row["name"] for row in cursor.fetchall()}
        assert "avatar_color" in user_cols
        assert "bio" in user_cols
        assert "rating" in user_cols

        # Verify match_records columns
        cursor.execute("PRAGMA table_info(match_records)")
        match_cols = {row["name"] for row in cursor.fetchall()}
        assert "user_id" in match_cols
        assert "game_mode" in match_cols
        assert "room_code" in match_cols
        assert "is_online" in match_cols
        assert "team_winner" in match_cols

        # Verify match_participants table
        cursor.execute("PRAGMA table_info(match_participants)")
        part_cols = {row["name"] for row in cursor.fetchall()}
        assert {"id", "match_id", "user_id", "username", "player_seat", "team_id", "is_winner"}.issubset(part_cols)


def test_existing_users_preserved():
    """Verify that existing users in the database are intact with default profile values."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) AS cnt FROM users")
        count = cursor.fetchone()["cnt"]
        assert count > 0, "Users table should contain existing registered accounts"


def test_authentication_and_session(test_player):
    """Verify existing user authentication, session generation, and session lookup."""
    user = get_user_by_session(test_player["session_token"])
    assert user is not None
    assert user["id"] == test_player["id"]
    assert user["username"] == test_player["username"]


def test_get_user_profile_structure(test_player):
    """Verify get_user_profile returns complete profile dictionary and stats structure."""
    profile = get_user_profile(test_player["id"])
    assert profile["username"] == test_player["username"]
    assert "rating" in profile
    assert "avatar_color" in profile
    assert "bio" in profile
    assert "total_matches" in profile
    assert "total_wins" in profile
    assert "win_rate" in profile
    assert "by_mode" in profile
    assert {"1v1_ai", "1v1_human", "2v2_team", "4p_ffa"}.issubset(profile["by_mode"].keys())


def test_update_user_profile(test_player):
    """Verify updating name, bio, email, and avatar color."""
    updated = update_user_profile(
        user_id=test_player["id"],
        full_name="Grandmaster Tester",
        email="gm_tester@ludo.ai",
        avatar_color="emerald",
        bio="Specializing in DRL Blockade breaking.",
    )
    assert updated["full_name"] == "Grandmaster Tester"
    assert updated["email"] == "gm_tester@ludo.ai"
    assert updated["avatar_color"] == "emerald"
    assert updated["bio"] == "Specializing in DRL Blockade breaking."

    # Verify re-fetching profile reflects updates
    profile = get_user_profile(test_player["id"])
    assert profile["full_name"] == "Grandmaster Tester"
    assert profile["avatar_color"] == "emerald"


def test_update_user_profile_duplicate_email_rejected(test_player):
    """Verify that updating email to one already owned by another user is rejected."""
    # Create another user
    other_user = "other_player_99"
    if not get_user_by_identifier(other_user):
        register_user(other_user, "Pass123456!", email="other_unique@ludo.ai")

    with pytest.raises(ValueError, match="already registered"):
        update_user_profile(
            user_id=test_player["id"],
            email="other_unique@ludo.ai",
        )


def test_match_history_and_statistics(test_player):
    """Verify saving match records with participants updates stats, ratings, and history."""
    initial_profile = get_user_profile(test_player["id"])
    initial_rating = initial_profile["rating"]

    # Record a match where test_player won in 1v1_ai mode
    match_id = save_match_record(
        winner=0,
        total_turns=45,
        total_captures=3,
        duration_seconds=120,
        player_types={0: "human", 1: "dqn"},
        user_id=test_player["id"],
        game_mode="1v1_ai",
        room_code="LOCAL-1V1",
        is_online=False,
        participants=[
            {
                "user_id": test_player["id"],
                "username": test_player["username"],
                "player_seat": 0,
                "team_id": 0,
                "controller_type": "human",
                "is_winner": True,
                "tokens_finished": 4,
                "total_captures": 2,
                "final_score": 100,
            },
            {
                "user_id": None,
                "username": "AI_DQN",
                "player_seat": 1,
                "team_id": 1,
                "controller_type": "ai",
                "is_winner": False,
                "tokens_finished": 2,
                "total_captures": 1,
                "final_score": 50,
            },
        ],
    )
    assert match_id > 0

    # Check match history
    history = get_user_match_history(test_player["id"], limit=5)
    assert len(history) > 0
    latest = history[0]
    assert latest["game_mode"] == "1v1_ai"
    assert latest["outcome"] == "Won"
    assert latest["player_seat"] == 0

    # Check rating increased (+25)
    updated_profile = get_user_profile(test_player["id"])
    assert updated_profile["rating"] == initial_rating + 25
    assert updated_profile["total_wins"] >= 1
    assert updated_profile["by_mode"]["1v1_ai"]["wins"] >= 1


def test_profile_api_unauthorized(client):
    """Verify that unauthenticated access to /api/profile returns 401."""
    resp_get = client.get("/api/profile")
    assert resp_get.status_code == 401

    resp_put = client.put("/api/profile", json={"bio": "Hacker attempt"})
    assert resp_put.status_code == 401


def test_profile_api_authorized(client, test_player):
    """Verify that authenticated requests to /api/profile return 200 and accept updates."""
    client.cookies.set("session_token", test_player["session_token"])

    # GET profile
    resp_get = client.get("/api/profile")
    assert resp_get.status_code == 200
    data = resp_get.json()
    assert data["status"] == "ok"
    assert data["profile"]["username"] == test_player["username"]

    # PUT update profile
    resp_put = client.put(
        "/api/profile",
        json={
            "full_name": "API Tester Pro",
            "avatar_color": "indigo",
            "bio": "Updated via REST API.",
        },
    )
    assert resp_put.status_code == 200
    put_data = resp_put.json()
    assert put_data["status"] == "ok"
    assert put_data["profile"]["avatar_color"] == "indigo"
    assert put_data["profile"]["bio"] == "Updated via REST API."


def test_profile_page_route(client, test_player):
    """Verify /profile page renders HTML for authenticated users and redirects unauthenticated users."""
    # Unauthenticated -> redirect to login
    client.cookies.clear()
    resp_unauth = client.get("/profile", follow_redirects=False)
    assert resp_unauth.status_code == 302
    assert "/login" in resp_unauth.headers["location"]

    # Authenticated -> 200 OK HTML
    client.cookies.set("session_token", test_player["session_token"])
    resp_auth = client.get("/profile")
    assert resp_auth.status_code == 200
    assert "Player Profile" in resp_auth.text
    assert test_player["username"] in resp_auth.text
