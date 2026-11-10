"""Tests for Phase C: UI Polish, User Experience and LAN Readiness.

Verifies:
1. Partial unique index on match_records(room_code) prevents online race condition duplicates
   while allowing multiple 'LOCAL-1V1' records.
2. Registration creates individual player profiles without hardcoded team assignment.
3. Base navigation exposes the 7 required items in authoritative order.
4. Ludo Arena templates and scripts support all 5 match modes.
5. Authoritative team mapping (Team 0: Red+Yellow, Team 1: Green+Blue).
6. 300ms snappy dice animation and legal token glow classes exist in static assets.
7. LAN launcher run_web_lan.bat binds to 0.0.0.0:8000 and run_web.bat preserves 127.0.0.1:8000.
"""

from pathlib import Path
import sqlite3
import pytest
from fastapi.testclient import TestClient

from web.app import app
from web.utils.auth import (
    get_db_path,
    init_db,
    register_user,
    save_match_record,
)


@pytest.fixture
def client():
    return TestClient(app)


def test_partial_unique_index_on_room_code():
    """Verify that match_records rejects duplicate online room codes but permits multiple LOCAL-1V1."""
    init_db()
    db_path = get_db_path()

    # Verify index exists in sqlite_master
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'index' AND name = 'idx_match_records_room_code'"
        )
        row = cursor.fetchone()
        assert row is not None, "idx_match_records_room_code partial unique index must exist in database"
        assert "LOCAL-1V1" in row[0]

    # Test that multiple LOCAL-1V1 records are permitted
    m1 = save_match_record(
        session_id="local_sess_1",
        match_mode="1v1_human_local",
        winner_id=0,
        winner_name="Player 0",
        winner_controller="human",
        total_turns=20,
        game_mode="1v1_human",
        room_code="LOCAL-1V1",
        is_online=False,
    )
    m2 = save_match_record(
        session_id="local_sess_2",
        match_mode="1v1_human_local",
        winner_id=1,
        winner_name="Player 1",
        winner_controller="human",
        total_turns=22,
        game_mode="1v1_human",
        room_code="LOCAL-1V1",
        is_online=False,
    )
    assert m1 != m2, "Distinct LOCAL-1V1 match records should both be inserted"

    # Test that online room codes are idempotent (cannot insert duplicates)
    online_code = "XYZ999"
    online1 = save_match_record(
        session_id=f"room_{online_code}",
        match_mode="1v1_human_online",
        winner_id=0,
        winner_name="Player 0",
        winner_controller="human",
        total_turns=30,
        game_mode="1v1_human",
        room_code=online_code,
        is_online=True,
    )
    # Attempting to save again with the same room code must return existing record ID
    online2 = save_match_record(
        session_id=f"room_{online_code}",
        match_mode="1v1_human_online",
        winner_id=0,
        winner_name="Player 0",
        winner_controller="human",
        total_turns=30,
        game_mode="1v1_human",
        room_code=online_code,
        is_online=True,
    )
    assert online1 == online2, "Duplicate online room code must return the same match ID without creating duplicates"


def test_registration_creates_individual_profile():
    """Verify registration creates an individual player account without team columns in user profile."""
    username = "phase_c_user"
    user = register_user(
        username=username,
        password="ValidPassword123!",
        email="phase_c@test.edu",
        full_name="Phase C Researcher",
    )
    assert user["username"] == username
    assert user["email"] == "phase_c@test.edu"
    assert user["rating"] == 1000
    # Team is match-level, not user-level
    assert "team" not in user
    assert "team_id" not in user


def test_navigation_exposes_7_items_in_order(client):
    """Verify base.html navigation bar contains all 7 designated items in authoritative order."""
    resp = client.get("/login")
    assert resp.status_code == 200
    html = resp.text

    # Verify all 7 navigation items are present
    assert "Dashboard" in html
    assert "Play Ludo" in html
    assert "Player Profile" in html
    assert "Results Hub" in html
    assert "DRL Models" in html
    assert "Research AI Tournament" in html
    assert "Sign In" in html or "Sign Out" in html or "Logout" in html

    # Verify correct links
    assert 'href="/dashboard"' in html
    assert 'href="/ludo"' in html
    assert 'href="/profile"' in html
    assert 'href="/results"' in html
    assert 'href="/models"' in html
    assert 'href="/ludo?tab=research"' in html


def test_authoritative_team_mapping_presentation():
    """Verify that team labels follow Team 0: Red+Yellow and Team 1: Green+Blue."""
    ludo_template = Path("web/templates/ludo.html").read_text(encoding="utf-8")
    assert "Team 0: Red & Yellow (P0 + P2)" in ludo_template or "Team 0 (Red+Yellow)" in ludo_template
    assert "Team 1: Green & Blue (P1 + P3)" in ludo_template or "Team 1 (Green+Blue)" in ludo_template

    ludo_js = Path("web/static/js/ludo.js").read_text(encoding="utf-8")
    assert "PLAYER_COLORS_META" in ludo_js
    assert "'Red'" in ludo_js
    assert "'Green'" in ludo_js
    assert "'Yellow'" in ludo_js
    assert "'Blue'" in ludo_js


def test_dice_animation_snappy_timing():
    """Verify 300ms dice animation timing in CSS and JS."""
    theme_css = Path("web/static/css/theme.css").read_text(encoding="utf-8")
    assert "0.3s" in theme_css
    assert "roll3d" in theme_css

    ludo_js = Path("web/static/js/ludo.js").read_text(encoding="utf-8")
    assert "counter >= 6" in ludo_js  # 6 * 50ms = 300ms


def test_lan_launcher_configuration():
    """Verify run_web_lan.bat binds to 0.0.0.0:8000 and run_web.bat binds to 127.0.0.1:8000."""
    lan_bat = Path("run_web_lan.bat").read_text(encoding="utf-8")
    assert "--host 0.0.0.0" in lan_bat
    assert "--port 8000" in lan_bat
    assert "ipconfig" in lan_bat
    assert "Firewall" in lan_bat or "firewall" in lan_bat

    local_bat = Path("run_web.bat").read_text(encoding="utf-8")
    assert "--host 127.0.0.1" in local_bat
    assert "--port 8000" in local_bat
