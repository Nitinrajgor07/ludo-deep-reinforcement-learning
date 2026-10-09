"""FastAPI REST API endpoints for PhD Ludo DRL demonstration.

Provides real empirical data, checkpoint inventory, hardware telemetry, and interactive 4-Player Ludo arena control.
100% Light Theme AI Research Standard. Zero Streamlit Dependency.
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from web.services.data_service import data_service, get_system_telemetry
from web.services.game_service import game_service

router = APIRouter(prefix="/api", tags=["API"])


class NewGameRequest(BaseModel):
    session_id: Optional[str] = None
    p0_type: Optional[str] = "dqn"
    p1_type: Optional[str] = "expert"
    p2_type: Optional[str] = "random"
    p3_type: Optional[str] = "expert"
    algorithm: Optional[str] = None       # Backward-compat alias for p0_type
    opponent_type: Optional[str] = None   # Backward-compat alias for p1_type
    checkpoint_path: Optional[str] = None
    seed: Optional[int] = 42


class PlayerActionRequest(BaseModel):
    session_id: str
    player_id: int = Field(..., ge=0, le=3)
    token_id: int = Field(..., ge=0, le=3)


class ManualStepRequest(BaseModel):
    session_id: str
    action: int = Field(..., ge=0, le=3)
    player_id: Optional[int] = None


class AutoPlayRequest(BaseModel):
    session_id: str
    steps: int = Field(5, ge=1, le=50)


class SessionActionRequest(BaseModel):
    session_id: str


@router.get("/status")
def get_status() -> Dict[str, Any]:
    """Return runtime system status, hardware telemetry, and artifact availability."""
    checkpoints = data_service.scan_checkpoints()
    figs = data_service.get_figures_manifest()
    tables = data_service.get_tables_manifest()
    telemetry = get_system_telemetry()
    return {
        "status": "online",
        "theme": "light",
        "checkpoint_count": len(checkpoints),
        "figure_count": len(figs),
        "table_count": len(tables),
        "telemetry": telemetry,
    }


@router.get("/models")
def get_models() -> List[Dict[str, Any]]:
    """Return technical specifications and checkpoint status for the 4 core RL models."""
    return data_service.get_model_specs()


@router.get("/checkpoints")
def get_checkpoints() -> List[Dict[str, Any]]:
    """Return list of all verified .pt model checkpoints."""
    return [ckpt.__dict__ for ckpt in data_service.scan_checkpoints()]


@router.get("/tables/{table_name}")
def get_table(table_name: str) -> Dict[str, Any]:
    """Return structured JSON data for a specific empirical results table."""
    data = data_service.load_table_json(table_name)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Table '{table_name}' not available yet.")
    return data


@router.get("/visualizations")
def get_visualizations(category: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return manifest of verified generated research figures."""
    figs = data_service.get_figures_manifest()
    if category:
        figs = [f for f in figs if f["category"].lower() == category.lower()]
    return figs


@router.get("/training")
def get_training_info() -> Dict[str, Any]:
    """Return training protocol status, pilot curves summary, and checkpoint telemetry."""
    pilot_df = data_service.load_table_dataframe("pilot_summary")
    return {
        "final_500k_completed": False,
        "protocol": "500,000 steps per model x 5 independent seeds (2,000,000 steps total)",
        "pilot_available": not pilot_df.empty,
        "pilot_runs_count": len(pilot_df) if not pilot_df.empty else 0,
        "checkpoints": [ckpt.__dict__ for ckpt in data_service.scan_checkpoints()],
    }


@router.get("/evaluation")
def get_evaluation_info() -> Dict[str, Any]:
    """Return evaluation metrics, comparing Preliminary/Pilot and Final states."""
    eval_df = data_service.load_table_dataframe("pilot_summary")
    return {
        "final_evaluation_completed": False,
        "protocol": "200,000 games (5,000 games x 5 seeds x 4 models x 2 opponents)",
        "pilot_data": eval_df.to_dict(orient="records") if not eval_df.empty else [],
    }


@router.get("/statistics")
def get_statistics_info() -> Dict[str, Any]:
    """Return statistical analysis tables and test summaries."""
    pilot = data_service.load_table_json("pilot_summary")
    robust = data_service.load_table_json("robustness_summary")
    return {
        "final_statistical_analysis_completed": False,
        "pilot_summary": pilot,
        "robustness_summary": robust,
    }


# --- Interactive 4-Player Ludo Arena Endpoints ---

@router.post("/ludo/new")
def new_game(req: NewGameRequest) -> Dict[str, Any]:
    """Initialize or reset an interactive 4-player Ludo match."""
    p0 = req.p0_type or req.algorithm or "dqn"
    p1 = req.p1_type or req.opponent_type or "expert"
    p2 = req.p2_type or "random"
    p3 = req.p3_type or "expert"

    player_types = {0: p0, 1: p1, 2: p2, 3: p3}

    session = game_service.get_or_create_session(
        session_id=req.session_id,
        player_types=player_types,
        checkpoint_path=req.checkpoint_path,
        seed=req.seed,
    )
    if req.session_id:
        session = game_service.reset_session(
            session_id=req.session_id,
            player_types=player_types,
            seed=req.seed,
        )

    summary = session.get_summary()
    return {
        "session_id": session.session_id,
        "current_player": summary["current_player"],
        "dice": summary["dice_value"],
        "legal_actions": summary["legal_actions"],
        "players": summary["players"],
        "state": summary,
        "svg": session.render_svg(),
    }


@router.get("/ludo/state")
def get_game_state(session_id: str = Query("default")) -> Dict[str, Any]:
    """Retrieve authoritative current game state and SVG board visualization."""
    session = game_service.get_or_create_session(session_id=session_id)
    summary = session.get_summary()
    return {
        "session_id": session.session_id,
        "current_player": summary["current_player"],
        "dice": summary["dice_value"],
        "legal_actions": summary["legal_actions"],
        "players": summary["players"],
        "state": summary,
        "svg": session.render_svg(),
    }


@router.post("/ludo/step")
def step_game(req: SessionActionRequest) -> Dict[str, Any]:
    """Advance the game by one AI / automated player turn."""
    return game_service.step_session(req.session_id)


@router.post("/ludo/action")
def player_action(req: PlayerActionRequest) -> Dict[str, Any]:
    """Execute a validated token action for a specific player."""
    try:
        return game_service.manual_step_session(req.session_id, req.player_id, req.token_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/ludo/manual_step")
def manual_step(req: ManualStepRequest) -> Dict[str, Any]:
    """Backward-compatible endpoint for token action."""
    session = game_service.get_or_create_session(session_id=req.session_id)
    pid = req.player_id if req.player_id is not None else session.env.current_player
    try:
        return game_service.manual_step_session(req.session_id, pid, req.action)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/ludo/auto_play")
def auto_play(req: AutoPlayRequest) -> Dict[str, Any]:
    """Run up to N turns sequentially across all 4 players."""
    return game_service.run_n_turns(req.session_id, req.steps)
