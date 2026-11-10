"""Comprehensive test suite for FastAPI Web Application, endpoints, templates, and Ludo Arena.

Strict Constraint: 100% Light Theme Only. Zero Streamlit Dependency.
"""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from web.app import app
from web.utils.data_loaders import (
    get_project_root,
    scan_available_checkpoints,
    load_pilot_summary,
    load_table_csv,
    CheckpointInfo,
)
from web.components.board_renderer import render_ludo_board_svg
from web.services.game_service import GameSession
from phd_ludo_drl.environment.ludo_env import LudoEnvironment

client = TestClient(app)


def test_project_root():
    """Verify that get_project_root locates the repository root correctly."""
    root = get_project_root()
    assert root.exists()
    assert (root / "phd_ludo_drl").exists()
    assert (root / "FINAL_EXPERIMENT_PROTOCOL.md").exists()


def test_scan_available_checkpoints():
    """Verify that scan_available_checkpoints locates checkpoints on disk."""
    checkpoints = scan_available_checkpoints()
    assert isinstance(checkpoints, list)
    assert len(checkpoints) > 0
    first = checkpoints[0]
    assert isinstance(first, CheckpointInfo)
    assert first.algorithm in ["dqn", "ppo", "model3", "model4", "unknown"]
    assert first.file_size_mb > 0.0


def test_load_pilot_summary():
    """Verify that pilot summary loads correctly."""
    df = load_pilot_summary()
    if df is not None:
        assert not df.empty


def test_load_table_csv():
    """Verify that results tables can be loaded into DataFrames."""
    df = load_table_csv("table_training_results")
    if df is not None:
        assert not df.empty


def test_render_ludo_board_svg():
    """Verify that render_ludo_board_svg outputs well-formed SVG string."""
    env = LudoEnvironment(seed=42)
    env.reset()
    svg = render_ludo_board_svg(
        tokens=env.tokens,
        current_player=env.current_player,
        dice_value=env.dice.value,
        legal_actions=env.get_legal_actions(),
        selected_action=None,
        winner=None,
        size=500,
    )
    assert isinstance(svg, str)
    assert svg.startswith("<svg")
    assert svg.endswith("</svg>")
    assert "RED YARD" in svg
    assert "GREEN YARD" in svg
    assert "GOAL" in svg
    assert "★" in svg


def test_game_session_initialization_and_step():
    """Verify that GameSession steps correctly and advances turns without illegal actions."""
    gs = GameSession(session_id="test_session", algorithm="dqn", opponent_type="random", seed=42)
    assert gs.env.current_player in [0, 1, 2, 3]
    assert not gs.is_terminal

    terminal, action = gs.step()
    assert isinstance(terminal, bool)
    if action is not None:
        assert action in [0, 1, 2, 3]
        assert len(gs.turn_history) >= 1
        last_turn = gs.turn_history[-1]
        assert "actor" in last_turn
        assert "reward" in last_turn


def test_game_session_different_algorithms():
    """Verify that GameSession can be instantiated with all 4 candidate models."""
    for algo in ["dqn", "ppo", "model3", "model4"]:
        gs = GameSession(session_id=f"test_{algo}", algorithm=algo, opponent_type="random", seed=101)
        assert gs.agent is not None
        assert gs.algorithm == algo
        terminal, action = gs.step()
        assert isinstance(terminal, bool)


def test_light_glass_theme_invariants():
    """Verify that theme CSS contains required glassmorphic variables and components."""
    theme_css = get_project_root() / "web" / "static" / "css" / "theme.css"
    assert theme_css.exists()
    content = theme_css.read_text(encoding="utf-8")

    assert "backdrop-filter" in content
    assert "--bg-base:" in content
    assert "--text-primary:" in content
    assert ".glass-card" in content


def test_fastapi_index_redirect():
    """Verify that GET / redirects to /dashboard with status 302."""
    res = client.get("/", follow_redirects=False)
    assert res.status_code == 302
    assert res.headers["location"] == "/dashboard"


@pytest.mark.parametrize("path", [
    "/dashboard",
    "/results",
    "/ludo",
    "/models",
])
def test_fastapi_pages_render_html(path):
    """Verify that the 4 primary HTML pages return status 200 with light theme markup."""
    res = client.get(path)
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert "PhD Ludo DRL" in res.text
    assert "/static/css/theme.css" in res.text
    # Strict light theme validation
    assert "dark-mode" not in res.text.lower()


@pytest.mark.parametrize("path", [
    "/training",
    "/evaluation",
    "/statistics",
    "/visualizations",
    "/about",
])
def test_fastapi_legacy_redirects(path):
    """Verify that legacy redundant routes issue status 302 redirects."""
    res = client.get(path, follow_redirects=False)
    assert res.status_code == 302
    assert res.headers["location"] in ["/results", "/dashboard"]


def test_fastapi_api_status():
    """Verify GET /api/status returns valid JSON telemetry."""
    res = client.get("/api/status")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "online"
    assert data["theme"] == "light"
    assert "checkpoint_count" in data
    assert "figure_count" in data
    assert "telemetry" in data


def test_fastapi_api_models():
    """Verify GET /api/models returns specifications for all 4 models."""
    res = client.get("/api/models")
    assert res.status_code == 200
    models = res.json()
    assert isinstance(models, list)
    assert len(models) == 4
    ids = [m["id"] for m in models]
    assert "dqn" in ids
    assert "ppo" in ids
    assert "model3" in ids
    assert "model4" in ids


def test_fastapi_api_checkpoints():
    """Verify GET /api/checkpoints returns valid checkpoints list."""
    res = client.get("/api/checkpoints")
    assert res.status_code == 200
    ckpts = res.json()
    assert isinstance(ckpts, list)
    assert len(ckpts) > 0


def test_fastapi_api_visualizations():
    """Verify GET /api/visualizations returns figure catalog."""
    res = client.get("/api/visualizations")
    assert res.status_code == 200
    figs = res.json()
    assert isinstance(figs, list)
    if figs:
        assert "url" in figs[0]
        assert "title" in figs[0]


def test_fastapi_api_ludo_lifecycle():
    """Verify full interactive Ludo game lifecycle via REST API."""
    # 1. Start new game
    new_res = client.post("/api/ludo/new", json={
        "session_id": "pytest_ludo_session",
        "algorithm": "dqn",
        "opponent_type": "random",
        "seed": 42,
    })
    assert new_res.status_code == 200
    data = new_res.json()
    assert "state" in data
    assert "svg" in data
    assert data["state"]["turn_count"] >= 0

    # 2. Get game state
    state_res = client.get("/api/ludo/state?session_id=pytest_ludo_session")
    assert state_res.status_code == 200
    assert state_res.json()["session_id"] == "pytest_ludo_session"

    # 2b. Test Manual Roll Dice endpoint
    roll_res = client.post("/api/ludo/roll", json={"session_id": "pytest_ludo_session"})
    assert roll_res.status_code == 200
    roll_data = roll_res.json()
    assert "dice_value" in roll_data
    assert 1 <= roll_data["dice_value"] <= 6

    # 3. Step AI move
    step_res = client.post("/api/ludo/step", json={"session_id": "pytest_ludo_session"})
    assert step_res.status_code == 200
    step_data = step_res.json()
    assert "action" in step_data

    # 4. Auto play 3 turns
    auto_res = client.post("/api/ludo/auto_play", json={
        "session_id": "pytest_ludo_session",
        "steps": 3,
    })
    assert auto_res.status_code == 200
    assert "state" in auto_res.json()


def test_fastapi_static_assets():
    """Verify static assets are accessible."""
    css_res = client.get("/static/css/theme.css")
    assert css_res.status_code == 200
    assert "text/css" in css_res.headers["content-type"]

    js_res = client.get("/static/js/main.js")
    assert js_res.status_code == 200

    ludo_js_res = client.get("/static/js/ludo.js")
    assert ludo_js_res.status_code == 200


def test_fastapi_logout():
    """Verify POST /api/logout endpoint."""
    res = client.post("/api/logout")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "logged_out"
    assert "redirect_url" in data


def test_fastapi_results_export_and_upload():
    """Verify result export as CSV/JSON and custom result upload storage."""
    # 1. Test CSV Export with existing table
    csv_res = client.get("/api/results/export/table1_descriptive_summary?format=csv")
    assert csv_res.status_code == 200
    assert "text/csv" in csv_res.headers["content-type"]
    assert "attachment; filename=table1_descriptive_summary.csv" in csv_res.headers["content-disposition"]
    assert len(csv_res.text) > 0

    # 2. Test JSON Export
    json_res = client.get("/api/results/export/table1_descriptive_summary?format=json")
    assert json_res.status_code == 200
    assert "application/json" in json_res.headers["content-type"]
    assert "attachment; filename=table1_descriptive_summary.json" in json_res.headers["content-disposition"]

    # 3. Test Invalid Table 404
    err_res = client.get("/api/results/export/non_existent_table_xyz")
    assert err_res.status_code == 404

    # 4. Test Result Upload (CSV)
    sample_csv = b"model,seed,win_rate,avg_turns\nDQN,42,0.85,120\nPPO,42,0.72,145\n"
    upload_res = client.post(
        "/api/results/upload",
        files={"file": ("test_custom_eval.csv", sample_csv, "text/csv")},
    )
    assert upload_res.status_code == 200
    up_data = upload_res.json()
    assert up_data["status"] == "success"
    assert up_data["file"]["name"] == "test_custom_eval.csv"
    assert up_data["file"]["rows"] == 2
    assert up_data["file"]["columns"] == 4

    # Verify newly stored table is immediately accessible
    verify_res = client.get("/api/results/export/test_custom_eval?format=csv")
    assert verify_res.status_code == 200
    assert "test_custom_eval.csv" in verify_res.headers["content-disposition"]


def test_sqlite_auth_lifecycle():
    """Verify registration, login, session persistence, and logout flow."""
    import uuid
    rand_suffix = uuid.uuid4().hex[:6]
    test_user = f"user_{rand_suffix}"
    test_email = f"user_{rand_suffix}@ludo.ai"

    # 1. GET login and register pages render properly
    login_get = client.get("/login")
    assert login_get.status_code == 200
    assert "Sign In" in login_get.text

    reg_get = client.get("/register")
    assert reg_get.status_code == 200
    assert "Create Account" in reg_get.text

    # 2. Register a new user
    reg_post = client.post(
        "/register",
        data={
            "username": test_user,
            "password": "secretpassword123",
            "email": test_email,
            "full_name": "Test Researcher",
        },
        follow_redirects=False,
    )
    assert reg_post.status_code == 303
    assert "/login" in reg_post.headers["location"]

    # 3. Duplicate registration rejected
    dup_post = client.post(
        "/register",
        data={
            "username": test_user,
            "password": "password456",
        },
    )
    assert dup_post.status_code == 400
    assert "already taken" in dup_post.text

    # 4. Login with bad password rejected
    bad_login = client.post(
        "/login",
        data={"username": test_user, "password": "wrongpassword"},
    )
    assert bad_login.status_code == 400
    assert "Invalid username or password" in bad_login.text

    # 5. Successful login sets session cookie
    good_login = client.post(
        "/login",
        data={"username": test_user, "password": "secretpassword123"},
        follow_redirects=False,
    )
    assert good_login.status_code == 303
    assert "session_token" in good_login.cookies

    # 6. Authenticated user viewing dashboard shows their name
    dash = client.get("/dashboard", cookies=good_login.cookies)
    assert dash.status_code == 200
    assert "Test Researcher" in dash.text
    assert "Sign Out" in dash.text

    # 7. Logout removes session
    logout_res = client.get("/logout", cookies=good_login.cookies, follow_redirects=False)
    assert logout_res.status_code == 302
    assert logout_res.headers.get("location") == "/dashboard"


def test_api_account_existence_check():
    """Verify /api/auth/check endpoint for existing and non-existing accounts."""
    # 1. Existing user via GET
    res_exist = client.get("/api/auth/check?identifier=test_phd_user")
    assert res_exist.status_code == 200
    data = res_exist.json()
    assert data["exists"] is True
    assert data["username"] == "test_phd_user"

    # 2. Non-existing user via GET
    res_none = client.get("/api/auth/check?identifier=definitely_not_existing_user_999")
    assert res_none.status_code == 200
    data_none = res_none.json()
    assert data_none["exists"] is False

    # 3. Existing user via POST JSON body
    res_post = client.post("/api/auth/check", json={"username": "test_phd_user"})
    assert res_post.status_code == 200
    assert res_post.json()["exists"] is True


def test_existence_based_login_and_register_guards():
    """Verify intelligent messaging when attempting to login non-existent or register existing user."""
    # Attempting to login with non-existent user prompts registration
    res_login_nonexist = client.post(
        "/login",
        data={"username": "ghost_user_xyz", "password": "any_password_123"},
    )
    assert res_login_nonexist.status_code == 400
    assert "No account found" in res_login_nonexist.text
    assert "register" in res_login_nonexist.text.lower()

    # Attempting to register existing user informs them to sign in
    res_reg_exist = client.post(
        "/register",
        data={"username": "test_phd_user", "password": "another_password"},
    )
    assert res_reg_exist.status_code == 400
    assert "already taken" in res_reg_exist.text or "already exists" in res_reg_exist.text


def test_protected_pages_unauthenticated_redirection():
    """Verify that unauthenticated access to protected research pages is gated and redirected to /login."""
    for path in ["/dashboard", "/results", "/models", "/ludo"]:
        res = client.get(path, follow_redirects=False)
        assert res.status_code == 302
        assert res.headers["location"] == f"/login?next={path}"
