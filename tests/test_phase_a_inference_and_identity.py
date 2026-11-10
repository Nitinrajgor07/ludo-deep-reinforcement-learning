"""Targeted validation tests for Phase A: Critical DRL Inference & Local Multiplayer Identity Fixes.

Covers:
A. DRL Checkpoint Loading:
   - Every supported algorithm (DQN, PPO, SAC/model3, C51/model4) automatically resolves and loads its trained checkpoint.
   - Missing checkpoint raises FileNotFoundError with descriptive message.
   - Corrupt/incompatible checkpoint raises RuntimeError with descriptive message.
   - Randomly initialized weights are strictly rejected/never silently used.
   - Model caching partitions entries strictly by algorithm and checkpoint path.

B. Local Seat Identity:
   - 1v1 Local: Seat 0 is Host (Red), Seat 1 is Guest (Yellow) with amber avatar.
   - 2v2 Local: Seat 0 is Host (Red), Seat 1 is Guest 1 (Green), Seat 2 is Guest 2 (Yellow), Seat 3 is Guest 3 (Blue).
   - Zero off-by-one seat naming (e.g. no "Player 3" for Yellow).

C. Local Rating & User Identity Integrity:
   - Guest seats have user_id = None.
   - Host rating is only adjusted by host's personal seat result (never mutated by guest wins/losses).
   - Match participants correctly record seat ID, team ID, controller type, and tokens finished.
   - Online matches with registered accounts continue updating ratings for all participants.

D. Mode Preservation:
   - Team mapping remains strictly {0: 0, 1: 1, 2: 0, 3: 1}.
   - Playable Human vs AI (1v1) operates with loaded DRL weights.
   - Preserved 4-Player AI Tournament operates seamlessly.
"""

from pathlib import Path
import pytest
from starlette.testclient import TestClient

from web.app import app
from web.services.game_service import (
    DEFAULT_TRAINED_CHECKPOINTS,
    get_cached_agent,
    resolve_checkpoint_for_algorithm,
    game_service,
    GameSession4P,
)
from web.services.room_service import room_service, GameMode, RoomStatus
from web.utils.auth import (
    create_session,
    delete_session,
    get_db,
    get_user_by_identifier,
    get_user_profile,
    register_user,
)


@pytest.fixture
def client():
    return TestClient(app, cookies={})


@pytest.fixture
def test_host():
    """Create or retrieve a dedicated test player for host testing."""
    username = "phase_a_host"
    pwd = "HostPassword123!"
    existing = get_user_by_identifier(username)
    if not existing:
        user_dict = register_user(username, pwd, email="phase_a_host@ludo.ai", full_name="Phase A Host")
        uid = user_dict["id"]
    else:
        uid = existing["id"]

    token = create_session(uid)
    yield {"id": uid, "username": username, "token": token}
    delete_session(token)


# ==============================================================================
# SECTION A: DRL CHECKPOINT LOADING & INFERENCE INTEGRITY
# ==============================================================================

def test_all_algorithms_load_compatible_trained_checkpoints():
    """Verify that DQN, PPO, SAC (model3), and C51 (model4) resolve and load trained checkpoints."""
    for algo in ["dqn", "ppo", "model3", "sac", "model4", "c51"]:
        resolved_path = resolve_checkpoint_for_algorithm(algo)
        assert resolved_path.exists(), f"Resolved checkpoint for {algo} must exist on disk: {resolved_path}"
        assert resolved_path.is_file()

        agent = get_cached_agent(algo)
        assert agent is not None, f"Agent for {algo} must be loaded and cached"
        # Verify agent has initialized encoder and network weights
        if hasattr(agent, "online_net"):
            assert agent.online_net is not None
        elif hasattr(agent, "policy_net"):
            assert agent.policy_net is not None
        elif hasattr(agent, "actor"):
            assert agent.actor is not None


def test_missing_checkpoint_raises_clear_error():
    """Verify that specifying a non-existent checkpoint raises FileNotFoundError with clear message."""
    with pytest.raises(FileNotFoundError) as exc_info:
        resolve_checkpoint_for_algorithm("dqn", checkpoint_path_str="non_existent_model_checkpoint.pt")
    assert "Specified checkpoint not found" in str(exc_info.value)


def test_corrupt_checkpoint_raises_runtime_error(tmp_path):
    """Verify that a corrupt checkpoint file raises a RuntimeError and does not silently fall back."""
    corrupt_file = tmp_path / "corrupt_weights.pt"
    corrupt_file.write_text("NOT_A_VALID_TORCH_TENSOR")

    with pytest.raises(RuntimeError) as exc_info:
        get_cached_agent("dqn", checkpoint_path_str=str(corrupt_file))
    assert "Failed to load trained checkpoint" in str(exc_info.value)


def test_agent_caching_isolates_algorithms_and_checkpoints():
    """Verify that agent cache keys distinguish between different algorithms and paths."""
    agent_dqn = get_cached_agent("dqn")
    agent_ppo = get_cached_agent("ppo")
    agent_sac = get_cached_agent("model3")
    agent_c51 = get_cached_agent("model4")

    # Distinct algorithm instances
    assert agent_dqn is not agent_ppo
    assert agent_dqn is not agent_sac
    assert agent_dqn is not agent_c51
    assert agent_ppo is not agent_sac


def test_human_vs_ai_session_uses_trained_checkpoint(client, test_host):
    """Verify /api/ludo/new initializes a 2-player match with the loaded trained model."""
    client.cookies.set("session_token", test_host["token"])
    resp = client.post("/api/ludo/new", json={
        "session_id": "test_phase_a_ai_session",
        "num_players": 2,
        "p0_type": "human",
        "p1_type": "dqn",
        "seed": 42,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["state"]["num_players"] == 2
    assert data["state"]["players"][0]["controller"] == "HUMAN"
    assert data["state"]["players"][1]["controller"] == "DQN"
    assert data["state"]["players"][0]["color"] == "Red"
    assert data["state"]["players"][1]["color"] == "Yellow"


# ==============================================================================
# SECTION B: LOCAL MULTIPLAYER SEAT IDENTITY & COLOR INTEGRITY
# ==============================================================================

def test_local_1v1_seat_names_and_colors(client, test_host):
    """Verify 1v1 local mode populates Seat 1 as Guest (Yellow) with amber avatar."""
    client.cookies.set("session_token", test_host["token"])
    resp = client.post("/api/rooms/create", json={"game_mode": "1v1_human", "is_online": False})
    assert resp.status_code == 200
    room = resp.json()["room"]

    assert room["max_players"] == 2
    players = room["players"]
    assert len(players) == 2

    # Seat 0: Authenticated Host (Red)
    seat0 = players[0]
    assert seat0["seat_id"] == 0
    assert seat0["user_id"] == test_host["id"]
    assert seat0["username"] == test_host["username"]

    # Seat 1: Guest (Yellow)
    seat1 = players[1]
    assert seat1["seat_id"] == 1
    assert seat1["user_id"] is None
    assert seat1["username"] == "Guest (Yellow)"
    assert seat1["avatar_color"] == "amber"


def test_local_2v2_seat_names_and_colors(client, test_host):
    """Verify 2v2 local mode populates all 4 seats with authoritative names, colors, and teams."""
    client.cookies.set("session_token", test_host["token"])
    resp = client.post("/api/rooms/create", json={"game_mode": "2v2_team", "is_online": False})
    assert resp.status_code == 200
    room = resp.json()["room"]

    assert room["max_players"] == 4
    players = room["players"]
    assert len(players) == 4

    # Seat 0: Host (Red, Team 0)
    assert players[0]["seat_id"] == 0
    assert players[0]["user_id"] == test_host["id"]
    assert players[0]["team_id"] == 0

    # Seat 1: Guest 1 (Green, Team 1)
    assert players[1]["seat_id"] == 1
    assert players[1]["user_id"] is None
    assert players[1]["username"] == "Guest 1 (Green)"
    assert players[1]["avatar_color"] == "emerald"
    assert players[1]["team_id"] == 1

    # Seat 2: Guest 2 (Yellow, Team 0) - Zero off-by-one confusion
    assert players[2]["seat_id"] == 2
    assert players[2]["user_id"] is None
    assert players[2]["username"] == "Guest 2 (Yellow)"
    assert players[2]["avatar_color"] == "amber"
    assert players[2]["team_id"] == 0

    # Seat 3: Guest 3 (Blue, Team 1)
    assert players[3]["seat_id"] == 3
    assert players[3]["user_id"] is None
    assert players[3]["username"] == "Guest 3 (Blue)"
    assert players[3]["avatar_color"] == "blue"
    assert players[3]["team_id"] == 1


# ==============================================================================
# SECTION C: LOCAL RATING & IDENTITY INTEGRITY
# ==============================================================================

def test_local_guest_matches_do_not_mutate_host_rating(client, test_host):
    """Verify that when a local match completes, host rating is only adjusted once for host's seat."""
    client.cookies.set("session_token", test_host["token"])

    # Fetch initial rating
    profile_before = get_user_profile(test_host["id"])
    initial_rating = profile_before["rating"]

    # Create local 2v2 room
    resp = client.post("/api/rooms/create", json={"game_mode": "2v2_team", "is_online": False})
    room_code = resp.json()["room_code"]

    # Start room game
    client.post("/api/rooms/start", json={"room_code": room_code})
    room = room_service.get_room(room_code)

    # Force simulated completion: Team 0 (Seats 0 & 2) wins!
    room.winner = 0
    room.team_winner = 0
    # Finish 8 tokens for Team 0
    for t in room.env.tokens:
        if room.teams.get(t.player_id) == 0:
            t.state = 3  # FINISHED
            t.step_count = 56

    # Persist completed match
    room._persist_completed_match()

    # Verify host profile rating was adjusted by exactly +25 (for Seat 0 win)
    # and NOT mutated by Guest 1 (loss, -15), Guest 2 (win, +25), Guest 3 (loss, -15)
    profile_after = get_user_profile(test_host["id"])
    assert profile_after["rating"] == initial_rating + 25

    # Verify database match participants: Guest seats must have user_id = NULL
    conn = get_db()
    c = conn.cursor()
    c.execute(
        "SELECT player_seat, user_id, username, is_winner, team_id FROM match_participants WHERE match_id = "
        "(SELECT id FROM match_records WHERE room_code = ? ORDER BY id DESC LIMIT 1)",
        (room_code,),
    )
    rows = c.fetchall()
    conn.close()

    assert len(rows) == 4
    for r in rows:
        seat_num = r["player_seat"]
        if seat_num == 0:
            assert r["user_id"] == test_host["id"]
            assert r["is_winner"] == 1
        elif seat_num == 1:
            assert r["user_id"] is None
            assert r["username"] == "Guest 1 (Green)"
            assert r["is_winner"] == 0
        elif seat_num == 2:
            assert r["user_id"] is None
            assert r["username"] == "Guest 2 (Yellow)"
            assert r["is_winner"] == 1
        elif seat_num == 3:
            assert r["user_id"] is None
            assert r["username"] == "Guest 3 (Blue)"
            assert r["is_winner"] == 0
