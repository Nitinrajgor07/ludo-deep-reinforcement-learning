"""FastAPI WebSocket endpoints for authoritative real-time Ludo multiplayer synchronization.

Protocol Events:
- Client to Server:
  - JOIN_ROOM: {"type": "JOIN_ROOM"}
  - PLAYER_READY: {"type": "PLAYER_READY", "is_ready": bool}
  - START_GAME: {"type": "START_GAME"}
  - ROLL_DICE: {"type": "ROLL_DICE"}
  - MOVE_TOKEN: {"type": "MOVE_TOKEN", "token_id": int}
  - PING: {"type": "PING"}

- Server to Client:
  - GAME_STATE_UPDATE: {"type": "GAME_STATE_UPDATE", "event": str, "state": dict}
  - DICE_ROLLED: {"type": "DICE_ROLLED", "dice": int, "legal_actions": list, "state": dict}
  - PLAYER_READY: {"type": "PLAYER_READY", "user_id": int, "is_ready": bool, "state": dict}
  - PLAYER_DISCONNECTED: {"type": "PLAYER_DISCONNECTED", "username": str, "seat_id": int}
  - PLAYER_RECONNECTED: {"type": "PLAYER_RECONNECTED", "username": str, "seat_id": int}
  - GAME_FINISHED: {"type": "GAME_FINISHED", "winner": int, "team_winner": int, "state": dict}
  - ERROR: {"type": "ERROR", "message": str}
  - PONG: {"type": "PONG"}
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status

from web.services.room_service import GameRoom, RoomStatus, room_service
from web.utils.auth import get_user_by_session

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Multiplayer WebSockets"])


class ConnectionManager:
    """Manages active WebSockets, per-room serialization locks, and atomic room broadcasts."""

    def __init__(self):
        # room_code -> Dict[user_id, WebSocket]
        self._rooms: Dict[str, Dict[int, WebSocket]] = {}
        self._room_locks: Dict[str, asyncio.Lock] = {}
        self._lock = asyncio.Lock()

    def get_room_lock(self, room_code: str) -> asyncio.Lock:
        """Retrieve or create an async synchronization lock for room-level serialized transactions."""
        code = room_code.upper()
        if code not in self._room_locks:
            self._room_locks[code] = asyncio.Lock()
        return self._room_locks[code]

    async def connect(self, room_code: str, user_id: int, websocket: WebSocket) -> None:
        """Register authenticated WebSocket connection."""
        await websocket.accept()
        code = room_code.upper()
        async with self._lock:
            if code not in self._rooms:
                self._rooms[code] = {}
            self._rooms[code][user_id] = websocket

    async def disconnect(self, room_code: str, user_id: int) -> None:
        """Unregister WebSocket connection."""
        code = room_code.upper()
        async with self._lock:
            if code in self._rooms and user_id in self._rooms[code]:
                del self._rooms[code][user_id]
                if not self._rooms[code]:
                    del self._rooms[code]
                    self._room_locks.pop(code, None)

    async def broadcast(self, room_code: str, message: Dict[str, Any]) -> None:
        """Broadcast JSON message to all connected players in room."""
        code = room_code.upper()
        targets: List[WebSocket] = []
        async with self._lock:
            if code in self._rooms:
                targets = list(self._rooms[code].values())

        for ws in targets:
            try:
                await ws.send_json(message)
            except Exception:
                pass

    async def send_personal_message(self, room_code: str, user_id: int, message: Dict[str, Any]) -> None:
        """Send message to a specific user's WebSocket."""
        code = room_code.upper()
        ws: Optional[WebSocket] = None
        async with self._lock:
            if code in self._rooms:
                ws = self._rooms[code].get(user_id)

        if ws:
            try:
                await ws.send_json(message)
            except Exception:
                pass


manager = ConnectionManager()


def _authenticate_websocket(websocket: WebSocket) -> Optional[Dict[str, Any]]:
    """Authenticate WebSocket from secure cookie first, then Bearer header, then query token."""
    # 1. Prefer standard HTTP-only session cookie
    token = websocket.cookies.get("session_token")

    # 2. Check Authorization Bearer header
    if not token:
        auth_header = websocket.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()

    # 3. Fallback to query parameter (for test clients and environments where cookies are constrained)
    if not token:
        token = websocket.query_params.get("token")

    if not token or not isinstance(token, str) or len(token) < 16:
        return None

    # Retrieve and validate session in SQLite
    return get_user_by_session(token)


@router.websocket("/ws/rooms/{room_code}")
@router.websocket("/ws/{room_code}")
async def room_websocket_endpoint(websocket: WebSocket, room_code: str):
    """Authoritative real-time multiplayer WebSocket handler."""
    code = room_code.strip().upper()

    # 1. Authenticate user
    user = _authenticate_websocket(websocket)
    if not user:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Unauthorized session")
        return

    # 2. Check room existence
    room = room_service.get_room(code)
    if not room:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason=f"Room '{code}' does not exist")
        return

    # 3. Check seat assignment or auto-seat user in WAITING lobby
    seat = room.get_seat_for_user(user["id"])
    if not seat:
        if room.status == RoomStatus.WAITING and not room.is_full():
            try:
                seat = room.join(
                    user_id=user["id"],
                    username=user["username"],
                    avatar_color=user.get("avatar_color", "rose"),
                    rating=user.get("rating", 1000),
                )
            except ValueError as e:
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason=str(e))
                return
        elif room.status in [RoomStatus.IN_PROGRESS, RoomStatus.FINISHED, RoomStatus.ABANDONED, RoomStatus.EXPIRED]:
            await websocket.close(
                code=status.WS_1008_POLICY_VIOLATION,
                reason="Unauthorized: User is not an active participant in this match.",
            )
            return
        else:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Room is full")
            return

    # 4. Connect and broadcast reconnection/join
    await manager.connect(code, user["id"], websocket)
    seat.is_connected = True

    await manager.broadcast(
        code,
        {
            "type": "PLAYER_RECONNECTED" if seat.is_ready else "JOIN_ROOM",
            "username": user["username"],
            "seat_id": seat.seat_id,
            "team_id": seat.team_id,
            "state": room.get_state_dict(),
        },
    )

    # 5. Message event loop with per-room atomic serialization
    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
            except Exception:
                await manager.send_personal_message(code, user["id"], {"type": "ERROR", "message": "Invalid JSON format"})
                continue

            event_type = msg.get("type", "").upper()

            # Handle PING
            if event_type == "PING":
                await manager.send_personal_message(code, user["id"], {"type": "PONG"})
                continue

            # Handle PLAYER_READY
            elif event_type == "PLAYER_READY":
                is_ready = bool(msg.get("is_ready", True))
                async with manager.get_room_lock(code):
                    try:
                        room.set_player_ready(user["id"], is_ready)
                        await manager.broadcast(
                            code,
                            {
                                "type": "PLAYER_READY",
                                "user_id": user["id"],
                                "seat_id": seat.seat_id,
                                "is_ready": is_ready,
                                "state": room.get_state_dict(),
                            },
                        )
                    except ValueError as err:
                        await manager.send_personal_message(code, user["id"], {"type": "ERROR", "message": str(err)})

            # Handle START_GAME
            elif event_type == "START_GAME":
                async with manager.get_room_lock(code):
                    try:
                        room.start_game(user["id"])
                        await manager.broadcast(
                            code,
                            {
                                "type": "GAME_STATE_UPDATE",
                                "event": "GAME_STARTED",
                                "message": f"Game started by host @{user['username']}!",
                                "state": room.get_state_dict(),
                            },
                        )
                    except ValueError as err:
                        await manager.send_personal_message(code, user["id"], {"type": "ERROR", "message": str(err)})

            # Handle ROLL_DICE
            elif event_type == "ROLL_DICE":
                client_action_id = msg.get("client_action_id")
                expected_seq = msg.get("expected_action_seq")
                async with manager.get_room_lock(code):
                    try:
                        res = room.roll_dice(
                            user["id"],
                            client_action_id=client_action_id,
                            expected_action_seq=expected_seq,
                        )
                        await manager.broadcast(
                            code,
                            {
                                "type": "DICE_ROLLED" if res["event"] == "DICE_ROLLED" else "GAME_STATE_UPDATE",
                                "event": res["event"],
                                "dice": res["dice"],
                                "legal_actions": res.get("legal_actions", []),
                                "state": res["state"],
                            },
                        )
                    except ValueError as err:
                        await manager.send_personal_message(code, user["id"], {"type": "ERROR", "message": str(err)})

            # Handle MOVE_TOKEN
            elif event_type == "MOVE_TOKEN":
                token_id = msg.get("token_id")
                if token_id is None:
                    await manager.send_personal_message(code, user["id"], {"type": "ERROR", "message": "Missing 'token_id'"})
                    continue
                client_action_id = msg.get("client_action_id")
                expected_seq = msg.get("expected_action_seq")
                async with manager.get_room_lock(code):
                    try:
                        res = room.move_token(
                            user["id"],
                            int(token_id),
                            client_action_id=client_action_id,
                            expected_action_seq=expected_seq,
                        )
                        event_name = "GAME_FINISHED" if res.get("terminated") else "GAME_STATE_UPDATE"
                        await manager.broadcast(
                            code,
                            {
                                "type": event_name,
                                "event": res["event"],
                                "token_id": token_id,
                                "dice_used": res.get("dice_used"),
                                "winner": res.get("winner"),
                                "team_winner": res.get("team_winner"),
                                "state": res["state"],
                            },
                        )
                    except ValueError as err:
                        await manager.send_personal_message(code, user["id"], {"type": "ERROR", "message": str(err)})

            # Handle GET_STATE, SYNC, JOIN_ROOM
            elif event_type in ["GET_STATE", "SYNC", "JOIN_ROOM"]:
                await manager.send_personal_message(
                    code,
                    user["id"],
                    {"type": "GAME_STATE_UPDATE", "event": "SYNC", "state": room.get_state_dict()},
                )

            else:
                await manager.send_personal_message(
                    code,
                    user["id"],
                    {"type": "ERROR", "message": f"Unrecognized action '{event_type}'"},
                )

    except WebSocketDisconnect:
        await manager.disconnect(code, user["id"])
        seat.is_connected = False
        await manager.broadcast(
            code,
            {
                "type": "PLAYER_DISCONNECTED",
                "username": user["username"],
                "seat_id": seat.seat_id,
                "state": room.get_state_dict(),
            },
        )
    except Exception as exc:
        logger.error(f"WebSocket error in room {code}: {type(exc).__name__}")
        await manager.disconnect(code, user["id"])
        seat.is_connected = False
