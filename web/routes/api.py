"""FastAPI REST API endpoints for PhD Ludo DRL demonstration.

Provides real empirical data, checkpoint inventory, hardware telemetry, and interactive 4-Player Ludo arena control.
100% Light Theme AI Research Standard. Zero Streamlit Dependency.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, File, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field

from web.services.data_service import data_service, get_system_telemetry
from web.services.game_service import game_service
from web.services.room_service import room_service
from web.utils.auth import (
    check_user_exists,
    delete_session,
    get_match_statistics_summary,
    get_recent_matches,
    get_user_by_identifier,
    get_user_by_session,
    get_user_match_history,
    get_user_profile,
    update_user_profile,
)

router = APIRouter(prefix="/api", tags=["API"])


class NewGameRequest(BaseModel):
    session_id: Optional[str] = None
    num_players: Optional[int] = 4
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


class AccountCheckRequest(BaseModel):
    identifier: Optional[str] = None
    username: Optional[str] = None
    email: Optional[str] = None


@router.get("/auth/check")
@router.post("/auth/check")
def check_account_existence(
    username: Optional[str] = Query(None),
    email: Optional[str] = Query(None),
    identifier: Optional[str] = Query(None),
    body: Optional[AccountCheckRequest] = None,
) -> Dict[str, Any]:
    """Check whether an account exists in the SQLite database by username or email.

    Accepts input via query parameters (?username=... or ?identifier=...)
    or via JSON body (POST).
    """
    target = (
        (body.identifier if body and body.identifier else None)
        or (body.username if body and body.username else None)
        or (body.email if body and body.email else None)
        or identifier
        or username
        or email
    )
    if not target or not target.strip():
        return {
            "exists": False,
            "identifier": "",
            "username": None,
            "full_name": None,
            "email": None,
            "message": "No username or email provided.",
        }

    target = target.strip()
    user = get_user_by_identifier(target)
    if user:
        return {
            "exists": True,
            "identifier": target,
            "username": user["username"],
            "full_name": user["full_name"],
            "email": user["email"],
            "message": f"Account '{user['username']}' found in database.",
        }
    return {
        "exists": False,
        "identifier": target,
        "username": target.lower(),
        "full_name": None,
        "email": None,
        "message": f"Account '{target}' not found in database.",
    }


class ProfileUpdateRequest(BaseModel):
    full_name: Optional[str] = None
    email: Optional[str] = None
    avatar_color: Optional[str] = None
    bio: Optional[str] = None


def _get_authenticated_user(request: Request) -> Dict[str, Any]:
    """Extract authenticated user from session cookie or Authorization Bearer header."""
    token = request.cookies.get("session_token")
    if not token:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required. Please sign in.")
    user = get_user_by_session(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired session. Please sign in again.")
    return user


@router.get("/profile")
def get_profile(request: Request) -> Dict[str, Any]:
    """Retrieve player profile, aggregate statistics, and recent match history."""
    user = _get_authenticated_user(request)
    profile = get_user_profile(user["id"])
    history = get_user_match_history(user["id"], limit=20)
    return {
        "status": "ok",
        "profile": profile,
        "history": history,
    }


@router.put("/profile")
@router.post("/profile")
def update_profile(request: Request, body: ProfileUpdateRequest) -> Dict[str, Any]:
    """Update profile fields (full_name, email, avatar_color, bio) for the authenticated player."""
    user = _get_authenticated_user(request)
    try:
        updated = update_user_profile(
            user_id=user["id"],
            full_name=body.full_name,
            email=body.email,
            avatar_color=body.avatar_color,
            bio=body.bio,
        )
        return {
            "status": "ok",
            "message": "Profile updated successfully.",
            "profile": updated,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/logout")
@router.get("/logout")
def logout(request: Request, response: Response) -> Dict[str, Any]:
    """Perform logout action, invalidate session in database, and clear cookie."""
    token = request.cookies.get("session_token")
    if not token:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()
    if token:
        delete_session(token)
    response.delete_cookie("session_token")
    return {
        "status": "logged_out",
        "message": "Signed out successfully.",
        "redirect_url": "/dashboard",
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


@router.get("/results/export/{table_name}")
def export_table(table_name: str, format: str = Query("csv", pattern="^(csv|json)$")):
    """Export and download empirical table as CSV or JSON file."""
    df = data_service.load_table_dataframe(table_name)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"Result table '{table_name}' not found.")

    clean_name = Path(table_name).stem
    if format == "json":
        json_content = df.to_json(orient="records", indent=2)
        return PlainTextResponse(
            content=json_content,
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename={clean_name}.json"},
        )
    else:
        csv_content = df.to_csv(index=False)
        return PlainTextResponse(
            content=csv_content,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={clean_name}.csv"},
        )


@router.post("/results/upload")
async def upload_result(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Store an uploaded empirical result dataset (CSV or JSON)."""
    filename = file.filename or "uploaded_result.csv"
    if not (filename.endswith(".csv") or filename.endswith(".json")):
        raise HTTPException(status_code=400, detail="Only .csv and .json files are accepted.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        saved_info = data_service.save_custom_result(filename, content)
        return {
            "status": "success",
            "message": f"Successfully stored result file '{filename}'.",
            "file": saved_info,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to process result file: {exc}")


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


@router.get("/ludo/matches")
def get_matches_history(limit: int = Query(20, ge=1, le=100)) -> Dict[str, Any]:
    """Retrieve realtime SQLite match history and statistics."""
    matches = get_recent_matches(limit=limit)
    summary = get_match_statistics_summary()
    return {
        "status": "success",
        "summary": summary,
        "matches": matches,
    }


# --- Interactive 4-Player Ludo Arena Endpoints ---

@router.post("/ludo/new")
def new_game(req: NewGameRequest) -> Dict[str, Any]:
    """Initialize or reset an interactive 4-player Ludo match."""
    p0 = req.p0_type or req.algorithm or "dqn"
    p1 = req.p1_type or req.opponent_type or "expert"
    p2 = req.p2_type or "random"
    p3 = req.p3_type or "expert"

    n_players = 2 if req.num_players == 2 else 4
    if n_players == 2:
        player_types = {0: p0, 1: p1}
    else:
        player_types = {0: p0, 1: p1, 2: p2, 3: p3}

    try:
        session = game_service.get_or_create_session(
            session_id=req.session_id,
            player_types=player_types,
            checkpoint_path=req.checkpoint_path,
            seed=req.seed,
            num_players=n_players,
        )
        if req.session_id:
            session = game_service.reset_session(
                session_id=req.session_id,
                player_types=player_types,
                seed=req.seed,
                num_players=n_players,
            )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))

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


@router.post("/ludo/roll")
def roll_dice(req: SessionActionRequest) -> Dict[str, Any]:
    """Roll dice for active player."""
    return game_service.roll_session_dice(req.session_id)


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


# ==============================================================================
# Multiplayer Room Management Endpoints
# ==============================================================================

class CreateRoomRequest(BaseModel):
    game_mode: str = "1v1_human"
    is_online: bool = True


class JoinRoomRequest(BaseModel):
    room_code: str
    requested_seat: Optional[int] = None


class LeaveRoomRequest(BaseModel):
    room_code: str


class ReadyRoomRequest(BaseModel):
    room_code: str
    is_ready: bool = True


class StartRoomRequest(BaseModel):
    room_code: str


class RoomActionRequest(BaseModel):
    room_code: str
    client_action_id: Optional[str] = None
    expected_action_seq: Optional[int] = None


class RoomMoveRequest(BaseModel):
    room_code: str
    token_id: int
    client_action_id: Optional[str] = None
    expected_action_seq: Optional[int] = None


@router.post("/rooms/create")
def create_room(req: CreateRoomRequest, request: Request) -> Dict[str, Any]:
    """Create a new multiplayer room (1v1 Human or 2v2 Team)."""
    user = _get_authenticated_user(request)
    try:
        room = room_service.create_room(
            game_mode=req.game_mode,
            host_user_id=user["id"],
            host_username=user["username"],
            is_online=req.is_online,
            host_avatar=user.get("avatar_color", "rose"),
            host_rating=user.get("rating", 1000),
        )
        return {
            "status": "ok",
            "room_code": room.room_code,
            "room": room.get_state_dict(),
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/rooms/join")
def join_room(req: JoinRoomRequest, request: Request) -> Dict[str, Any]:
    """Join an existing multiplayer room using its 6-character room code."""
    user = _get_authenticated_user(request)
    try:
        room, seat = room_service.join_room(
            room_code=req.room_code,
            user_id=user["id"],
            username=user["username"],
            avatar_color=user.get("avatar_color", "rose"),
            rating=user.get("rating", 1000),
            requested_seat=req.requested_seat,
        )
        return {
            "status": "ok",
            "seat": seat.to_dict(),
            "room": room.get_state_dict(),
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/rooms/info")
def get_room_info(room_code: str = Query(...)) -> Dict[str, Any]:
    """Retrieve current state and players for a room by code."""
    room = room_service.get_room(room_code)
    if not room:
        raise HTTPException(status_code=404, detail=f"Room '{room_code}' was not found.")
    return {
        "status": "ok",
        "room": room.get_state_dict(),
    }


@router.post("/rooms/leave")
def leave_room(req: LeaveRoomRequest, request: Request) -> Dict[str, Any]:
    """Leave a multiplayer room."""
    user = _get_authenticated_user(request)
    ok = room_service.leave_room(req.room_code, user["id"])
    return {
        "status": "ok",
        "left": ok,
    }


@router.post("/rooms/ready")
def set_room_ready(req: ReadyRoomRequest, request: Request) -> Dict[str, Any]:
    """Set ready state for authenticated player in room."""
    user = _get_authenticated_user(request)
    room = room_service.get_room(req.room_code)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found.")
    try:
        room.set_player_ready(user["id"], req.is_ready)
        return {
            "status": "ok",
            "room": room.get_state_dict(),
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/rooms/start")
def start_room_game(req: StartRoomRequest, request: Request) -> Dict[str, Any]:
    """Start the match (host only)."""
    user = _get_authenticated_user(request)
    room = room_service.get_room(req.room_code)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found.")
    try:
        room.start_game(user["id"])
        return {
            "status": "ok",
            "room": room.get_state_dict(),
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/rooms/roll")
def roll_room_dice(req: RoomActionRequest, request: Request) -> Dict[str, Any]:
    """Authoritative dice roll for active turn with replay protection (supports both Local and Online modes)."""
    user = _get_authenticated_user(request)
    room = room_service.get_room(req.room_code)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found.")
    try:
        res = room.roll_dice(
            user["id"],
            client_action_id=req.client_action_id,
            expected_action_seq=req.expected_action_seq,
        )
        return {
            "status": "ok",
            "event": res["event"],
            "dice": res["dice"],
            "legal_actions": res.get("legal_actions", []),
            "room": res["state"],
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/rooms/action")
def move_room_token(req: RoomMoveRequest, request: Request) -> Dict[str, Any]:
    """Authoritative token movement for active turn with replay protection (supports both Local and Online modes)."""
    user = _get_authenticated_user(request)
    room = room_service.get_room(req.room_code)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found.")
    try:
        res = room.move_token(
            user["id"],
            req.token_id,
            client_action_id=req.client_action_id,
            expected_action_seq=req.expected_action_seq,
        )
        return {
            "status": "ok",
            "event": res["event"],
            "token_id": req.token_id,
            "winner": res.get("winner"),
            "team_winner": res.get("team_winner"),
            "room": res["state"],
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
