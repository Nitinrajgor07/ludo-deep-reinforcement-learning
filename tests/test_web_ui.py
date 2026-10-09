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
