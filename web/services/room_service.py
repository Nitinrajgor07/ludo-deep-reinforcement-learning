"""Room management service for multiplayer Ludo matches (Local and Online).

Supports:
- Human vs Human (1v1) — 2 Players
- Team vs Team (2v2) — 4 Players (Team 0: Seat 0 & 2; Team 1: Seat 1 & 3)

Provides authoritative room code generation, host management, seat allocation,
state synchronization, disconnect/reconnect tracking, and match record persistence.
Zero Streamlit dependency. 100% Light Theme AI Research Standard.
"""

from datetime import datetime, timezone
from enum import Enum
import secrets
import string
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from phd_ludo_drl.environment.constants import (
    DEFAULT_TEAMS_4P,
    NUM_PLAYERS,
    PLAYER_COLORS,
    PLAYER_START_SQUARES,
    PLAYER_START_SQUARES_4P,
    TEAM_NAMES_4P,
    TOKENS_PER_PLAYER,
)
from phd_ludo_drl.environment.ludo_env import LudoEnvironment
from web.components.board_renderer import render_ludo_board_svg
from web.utils.auth import save_match_record

# Room code alphabet: unambiguous uppercase letters and digits (omits O, 0, I, 1)
ROOM_CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
ROOM_CODE_LENGTH = 6

PLAYER_COLOR_NAMES_4P = ["Red", "Green", "Yellow", "Blue"]
PLAYER_COLOR_NAMES_2P = ["Red", "Yellow"]


class GameMode(str, Enum):
    """Supported multiplayer game modes."""
    ONE_VS_ONE_HUMAN = "1v1_human"
    TWO_VS_TWO_TEAM = "2v2_team"


class RoomStatus(str, Enum):
    """Lifecycle status of a game room."""
    WAITING = "waiting"          # Waiting for players to join
    READY = "ready"              # All seats filled, ready to start
    IN_PROGRESS = "in_progress"  # Game actively in progress
    FINISHED = "finished"        # Game completed
    ABANDONED = "abandoned"      # Host left or players abandoned
    EXPIRED = "expired"          # Room timed out


class TurnPhase(str, Enum):
    """Turn progression phases."""
    NEED_ROLL = "NEED_ROLL"      # Active player must roll the dice
    NEED_MOVE = "NEED_MOVE"      # Active player must select a token
    FINISHED = "FINISHED"        # Match is over


class PlayerSeat:
    """Represents an assigned seat in a room."""

    def __init__(
        self,
        seat_id: int,
        user_id: Optional[int] = None,
        username: str = "Player",
        team_id: Optional[int] = None,
        avatar_color: str = "rose",
        rating: int = 1000,
        is_ready: bool = False,
        controller_type: str = "human",
    ):
        self.seat_id = seat_id
        self.user_id = user_id
        self.username = username
        self.team_id = team_id
        self.avatar_color = avatar_color
        self.rating = rating
        self.is_ready = is_ready
        self.is_connected = True
        self.controller_type = controller_type
        self.connected_at = time.time()
        self.last_seen = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seat_id": self.seat_id,
            "user_id": self.user_id,
            "username": self.username,
            "team_id": self.team_id,
            "avatar_color": self.avatar_color,
            "rating": self.rating,
            "is_ready": self.is_ready,
            "is_connected": self.is_connected,
            "controller_type": self.controller_type,
        }


class GameRoom:
    """Authoritative game room managing lifecycle, seats, and Ludo environment."""

    def __init__(
        self,
        room_code: str,
        game_mode: str,
        host_user_id: int,
        host_username: str,
        is_online: bool = True,
        host_avatar: str = "rose",
        host_rating: int = 1000,
    ):
        self.room_code = room_code.upper()
        self.game_mode = game_mode
        self.host_user_id = host_user_id
        self.host_username = host_username
        self.is_online = is_online
        self.status = RoomStatus.WAITING
        self.created_at = time.time()
        self.last_activity = time.time()
        self.lock = threading.RLock()

        # Maximum player count & team setup
        if self.game_mode == GameMode.ONE_VS_ONE_HUMAN.value:
            self.max_players = 2
            self.teams: Optional[Dict[int, int]] = None
        elif self.game_mode == GameMode.TWO_VS_TWO_TEAM.value:
            self.max_players = 4
            self.teams = dict(DEFAULT_TEAMS_4P)  # {0: 0, 1: 1, 2: 0, 3: 1}
        else:
            raise ValueError(f"Unsupported room game mode: '{game_mode}'")

        # Initialize seats dictionary
        self.seats: Dict[int, Optional[PlayerSeat]] = {i: None for i in range(self.max_players)}

        # Assign Host to Seat 0
        host_team = self.teams[0] if self.teams else None
        self.seats[0] = PlayerSeat(
            seat_id=0,
            user_id=host_user_id,
            username=host_username,
            team_id=host_team,
            avatar_color=host_avatar,
            rating=host_rating,
            is_ready=True,
        )

        if not self.is_online:
            # Auto-populate local player seats for offline pass-and-play with authoritative names & colors
            if self.max_players == 2:
                # 1v1 Local: Seat 0 is Host (Red), Seat 1 is Guest (Yellow)
                self.seats[1] = PlayerSeat(
                    seat_id=1,
                    user_id=None,
                    username="Guest (Yellow)",
                    team_id=None,
                    avatar_color="amber",
                    rating=1000,
                    is_ready=True,
                )
            else:
                # 2v2 Local:
                # Seat 0: Host (Red, Team 0)
                # Seat 1: Guest 1 (Green, Team 1)
                # Seat 2: Guest 2 (Yellow, Team 0)
                # Seat 3: Guest 3 (Blue, Team 1)
                local_guest_configs = [
                    (1, "Guest 1 (Green)", "emerald", self.teams[1]),
                    (2, "Guest 2 (Yellow)", "amber", self.teams[2]),
                    (3, "Guest 3 (Blue)", "blue", self.teams[3]),
                ]
                for seat_i, name, avatar, t_id in local_guest_configs:
                    self.seats[seat_i] = PlayerSeat(
                        seat_id=seat_i,
                        user_id=None,
                        username=name,
                        team_id=t_id,
                        avatar_color=avatar,
                        rating=1000,
                        is_ready=True,
                    )
            self.status = RoomStatus.READY

        # Game state engine
        self.env: Optional[LudoEnvironment] = None
        self.turn_phase = TurnPhase.NEED_ROLL
        self.current_dice_value: Optional[int] = None
        self.legal_actions: List[int] = []
        self.action_seq: int = 0
        self.winner: Optional[int] = None
        self.team_winner: Optional[int] = None
        self.move_history: List[Dict[str, Any]] = []
        # Idempotency and replay prevention
        self.processed_action_ids: Set[str] = set()
        self.action_id_history: List[str] = []
        self.max_action_id_history: int = 500
        self._match_persisted: bool = False

        # Active WebSocket connections (user_id -> WebSocket)
        self.active_websockets: Dict[int, Any] = {}

    def _record_action_id(self, action_id: Optional[str]) -> None:
        """Register client action ID to prevent duplicates and replays."""
        if not action_id:
            return
        self.processed_action_ids.add(action_id)
        self.action_id_history.append(action_id)
        if len(self.action_id_history) > self.max_action_id_history:
            oldest = self.action_id_history.pop(0)
            self.processed_action_ids.discard(oldest)

    def touch(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = time.time()

    def get_player_count(self) -> int:
        """Return count of currently occupied seats."""
        return sum(1 for s in self.seats.values() if s is not None)

    def is_full(self) -> bool:
        """Check if all seats are occupied."""
        return self.get_player_count() >= self.max_players

    def get_seat_for_user(self, user_id: int) -> Optional[PlayerSeat]:
        """Find seat occupied by given user ID."""
        for seat in self.seats.values():
            if seat and seat.user_id == user_id:
                return seat
        return None

    def join(
        self,
        user_id: int,
        username: str,
        avatar_color: str = "rose",
        rating: int = 1000,
        requested_seat: Optional[int] = None,
    ) -> PlayerSeat:
        """Add user to room or reconnect if already joined."""
        with self.lock:
            self.touch()
            # 1. Check if user already has a seat (reconnect)
            existing_seat = self.get_seat_for_user(user_id)
            if existing_seat:
                existing_seat.is_connected = True
                existing_seat.last_seen = time.time()
                return existing_seat

            # 2. Check if room is full
            if self.is_full():
                raise ValueError(f"Room '{self.room_code}' is full (maximum {self.max_players} players).")

            # 3. Check if room is already completed or abandoned
            if self.status in [RoomStatus.FINISHED, RoomStatus.ABANDONED, RoomStatus.EXPIRED]:
                raise ValueError(f"Room '{self.room_code}' is no longer active ({self.status.value}).")

            # 4. Determine target seat
            target_seat_id = None
            if requested_seat is not None and 0 <= requested_seat < self.max_players:
                if self.seats[requested_seat] is None:
                    target_seat_id = requested_seat

            if target_seat_id is None:
                # Find first available seat
                for i in range(self.max_players):
                    if self.seats[i] is None:
                        target_seat_id = i
                        break

            if target_seat_id is None:
                raise ValueError(f"No available seats in room '{self.room_code}'.")

            # Assign team ID
            team_id = self.teams[target_seat_id] if self.teams else None

            # In local mode, auto-ready players; in online mode, player must ready up
            is_ready = True if not self.is_online else False

            seat = PlayerSeat(
                seat_id=target_seat_id,
                user_id=user_id,
                username=username,
                team_id=team_id,
                avatar_color=avatar_color,
                rating=rating,
                is_ready=is_ready,
            )
            self.seats[target_seat_id] = seat

            # Check if all players ready
            if self.is_full() and all(s.is_ready for s in self.seats.values() if s):
                self.status = RoomStatus.READY

            return seat

    def leave(self, user_id: int) -> Optional[int]:
        """Remove user from seat and handle host migration."""
        with self.lock:
            self.touch()
            seat = self.get_seat_for_user(user_id)
            if not seat:
                return None

            vacated_seat_id = seat.seat_id
            self.seats[vacated_seat_id] = None

            # If game is in progress, mark abandoned or leave empty
            if self.status == RoomStatus.IN_PROGRESS:
                # Game is forfeited / abandoned if critical player leaves
                self.status = RoomStatus.ABANDONED
            else:
                self.status = RoomStatus.WAITING

            # Host migration: If host left, transfer host to next connected player
            if user_id == self.host_user_id:
                new_host = None
                for s in self.seats.values():
                    if s and s.is_connected:
                        new_host = s
                        break
                if not new_host:
                    for s in self.seats.values():
                        if s:
                            new_host = s
                            break

                if new_host:
                    self.host_user_id = new_host.user_id
                    self.host_username = new_host.username
                else:
                    self.status = RoomStatus.ABANDONED

            return vacated_seat_id

    def set_player_ready(self, user_id: int, is_ready: bool = True) -> bool:
        """Toggle player ready state."""
        with self.lock:
            self.touch()
            if self.status in [RoomStatus.IN_PROGRESS, RoomStatus.FINISHED, RoomStatus.ABANDONED, RoomStatus.EXPIRED]:
                raise ValueError(f"Cannot change ready state: Game is already {self.status.value}.")

            seat = self.get_seat_for_user(user_id)
            if not seat:
                raise ValueError("User is not seated in this room.")

            seat.is_ready = is_ready
            if self.is_full() and all(s.is_ready for s in self.seats.values() if s):
                self.status = RoomStatus.READY
            elif self.status == RoomStatus.READY and not is_ready:
                self.status = RoomStatus.WAITING

            return seat.is_ready

    def start_game(self, requesting_user_id: int) -> bool:
        """Start the match if room is full and requester is host (or local mode)."""
        with self.lock:
            self.touch()
            if self.status == RoomStatus.FINISHED:
                raise ValueError("Cannot start game: Match is already finished.")
            if self.status == RoomStatus.IN_PROGRESS:
                raise ValueError("Game is already in progress.")
            if self.status in [RoomStatus.ABANDONED, RoomStatus.EXPIRED]:
                raise ValueError(f"Cannot start game: Room is {self.status.value}.")

            # Verify requester is host or local
            if requesting_user_id != self.host_user_id and self.is_online:
                raise ValueError("Only the room host can start the game.")

            # Verify all seats are filled
            if not self.is_full():
                raise ValueError(
                    f"Cannot start game: Room needs {self.max_players} players (currently {self.get_player_count()})."
                )

            # In online mode, verify all players are ready
            if self.is_online:
                unready = [s.username for s in self.seats.values() if s and not s.is_ready]
                if unready:
                    raise ValueError(f"Cannot start game: Waiting for players to ready up ({', '.join(unready)}).")

            # Initialize authoritative Ludo environment
            if self.game_mode == GameMode.ONE_VS_ONE_HUMAN.value:
                self.env = LudoEnvironment(
                    seed=int(time.time()),
                    max_turns=1000,
                    auto_advance_no_moves=False,
                    num_players=2,
                    teams=None,
                )
            else:
                self.env = LudoEnvironment(
                    seed=int(time.time()),
                    max_turns=1000,
                    auto_advance_no_moves=False,
                    num_players=4,
                    teams=dict(DEFAULT_TEAMS_4P),
                )

            self.status = RoomStatus.IN_PROGRESS
            self.turn_phase = TurnPhase.NEED_ROLL
            self.current_dice_value = None
            self.legal_actions = []
            self.action_seq = 0
            self.move_history = []
            return True

    def roll_dice(
        self,
        requesting_user_id: int,
        client_action_id: Optional[str] = None,
        expected_action_seq: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Roll dice for active player with strict server-side validation and replay protection."""
        with self.lock:
            self.touch()
            if self.status == RoomStatus.FINISHED:
                raise ValueError("Cannot roll dice: Match is already finished.")
            if self.status != RoomStatus.IN_PROGRESS or not self.env:
                raise ValueError(f"Game is not actively in progress (status: {self.status.value}).")

            # Replay protection: Check client action ID
            if client_action_id and client_action_id in self.processed_action_ids:
                raise ValueError(f"Duplicate or replayed action rejected (action_id: '{client_action_id}').")

            # Sequence synchronization: Reject stale actions
            if expected_action_seq is not None and expected_action_seq != self.action_seq:
                raise ValueError(
                    f"Stale action rejected: expected sequence {self.action_seq}, received {expected_action_seq}."
                )

            if self.turn_phase != TurnPhase.NEED_ROLL:
                raise ValueError("Dice has already been rolled for this turn. Awaiting token movement.")

            curr_player_seat = self.env.current_player
            seat = self.seats.get(curr_player_seat)

            # Validation: In online mode, user must own the active seat
            if self.is_online:
                if not seat or seat.user_id != requesting_user_id:
                    expected_name = seat.username if seat else f"Player {curr_player_seat}"
                    raise ValueError(f"It is {expected_name}'s turn (Seat {curr_player_seat}), not yours.")
            else:
                # In local mode, host can roll for all seats
                if requesting_user_id != self.host_user_id and (not seat or seat.user_id != requesting_user_id):
                    raise ValueError("Unauthorized player action.")

            # Record action ID
            self._record_action_id(client_action_id)

            # Roll dice authoritatively
            dice_val = self.env.dice.roll()
            self.current_dice_value = dice_val
            legal = self.env.get_legal_actions()
            self.legal_actions = legal

            # Check if player has legal moves
            if len(legal) == 0:
                # No legal moves possible: auto-advance turn
                self.turn_phase = TurnPhase.NEED_ROLL
                old_player = self.env.current_player
                self.env.current_player = (self.env.current_player + 1) % self.env.num_players
                self.env.turn_count += 1
                new_player = self.env.current_player

                log_entry = {
                    "turn": self.env.turn_count,
                    "player_seat": old_player,
                    "username": seat.username if seat else f"P{old_player}",
                    "dice": dice_val,
                    "action": "Pass",
                    "status": f"Rolled {dice_val} with no legal moves. Turn passed to Seat {new_player}.",
                    "timestamp": time.time(),
                }
                self.move_history.append(log_entry)
                self.action_seq += 1

                return {
                    "event": "TURN_PASSED",
                    "dice": dice_val,
                    "passed_seat": old_player,
                    "next_player_seat": new_player,
                    "legal_actions": [],
                    "state": self.get_state_dict(),
                }
            else:
                # Legal moves available: await token movement
                self.turn_phase = TurnPhase.NEED_MOVE
                return {
                    "event": "DICE_ROLLED",
                    "dice": dice_val,
                    "player_seat": curr_player_seat,
                    "legal_actions": legal,
                    "state": self.get_state_dict(),
                }

    def move_token(
        self,
        requesting_user_id: int,
        token_id: int,
        client_action_id: Optional[str] = None,
        expected_action_seq: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute validated token move for active player with replay protection."""
        with self.lock:
            self.touch()
            if self.status == RoomStatus.FINISHED:
                raise ValueError("Cannot move token: Match is already finished.")
            if self.status != RoomStatus.IN_PROGRESS or not self.env:
                raise ValueError(f"Game is not actively in progress (status: {self.status.value}).")

            # Replay protection: Check client action ID
            if client_action_id and client_action_id in self.processed_action_ids:
                raise ValueError(f"Duplicate or replayed action rejected (action_id: '{client_action_id}').")

            # Sequence synchronization: Reject stale actions
            if expected_action_seq is not None and expected_action_seq != self.action_seq:
                raise ValueError(
                    f"Stale action rejected: expected sequence {self.action_seq}, received {expected_action_seq}."
                )

            if self.turn_phase != TurnPhase.NEED_MOVE:
                raise ValueError("Must roll dice before moving a token.")

            curr_player_seat = self.env.current_player
            seat = self.seats.get(curr_player_seat)

            # Security validation: Verify player ownership of active seat
            if self.is_online:
                if not seat or seat.user_id != requesting_user_id:
                    expected_name = seat.username if seat else f"Player {curr_player_seat}"
                    raise ValueError(f"It is {expected_name}'s turn (Seat {curr_player_seat}), not yours.")
            else:
                if requesting_user_id != self.host_user_id and (not seat or seat.user_id != requesting_user_id):
                    raise ValueError("Unauthorized player action.")

            if token_id not in self.legal_actions:
                raise ValueError(
                    f"Token {token_id} is not a valid legal action for Seat {curr_player_seat}. "
                    f"Legal moves: {self.legal_actions}."
                )

            # Record action ID
            self._record_action_id(client_action_id)

            # Execute move in environment
            dice_used = self.current_dice_value or self.env.dice.value
            obs, reward, terminated, truncated, info = self.env.step(token_id)
            move_res = info.get("move_result")
            captures = len(move_res.captured_tokens) if move_res else 0

            # Formulate status description
            if terminated:
                self.status = RoomStatus.FINISHED
                self.turn_phase = TurnPhase.FINISHED
                self.winner = self.env.winner
                self.team_winner = self.env.team_winner
                status_desc = f"Won Match! (Player {self.winner})"
                if self.team_winner is not None:
                    status_desc = f"Team {self.team_winner} Victory!"
                # Persist match record to SQLite
                self._persist_completed_match()
            elif info.get("bonus_six"):
                self.turn_phase = TurnPhase.NEED_ROLL
                status_desc = f"Moved Token {token_id} (Rolled 6: Bonus Roll awarded!)"
            elif info.get("bonus_capture"):
                self.turn_phase = TurnPhase.NEED_ROLL
                status_desc = f"Captured opponent token! Bonus Roll awarded!"
            else:
                self.turn_phase = TurnPhase.NEED_ROLL
                status_desc = f"Moved Token {token_id}"

            self.action_seq += 1
            self.legal_actions = []

            log_entry = {
                "turn": self.env.turn_count,
                "player_seat": curr_player_seat,
                "username": seat.username if seat else f"P{curr_player_seat}",
                "dice": dice_used,
                "token_id": token_id,
                "action": f"Move Token {token_id}",
                "status": status_desc,
                "captures": captures,
                "timestamp": time.time(),
            }
            self.move_history.append(log_entry)

            return {
                "event": "GAME_FINISHED" if terminated else "TOKEN_MOVED",
                "player_seat": curr_player_seat,
                "token_id": token_id,
                "dice_used": dice_used,
                "terminated": terminated,
                "winner": self.winner,
                "team_winner": self.team_winner,
                "state": self.get_state_dict(),
            }

    def _persist_completed_match(self) -> None:
        """Record completed multiplayer match and update ratings in SQLite idempotently."""
        with self.lock:
            if self._match_persisted:
                return
            self._match_persisted = True

        try:
            participants = []
            for seat_id, seat in self.seats.items():
                if not seat:
                    continue
                # Determine if winner
                if self.team_winner is not None:
                    is_win = (seat.team_id == self.team_winner)
                else:
                    is_win = (seat_id == self.winner)

                # Count finished tokens
                tokens = [t for t in self.env.tokens if t.player_id == seat_id] if self.env else []
                fin_count = sum(1 for t in tokens if t.is_finished)

                participants.append({
                    "user_id": seat.user_id,
                    "username": seat.username,
                    "player_seat": seat_id,
                    "team_id": seat.team_id,
                    "controller_type": "human",
                    "is_winner": is_win,
                    "tokens_finished": fin_count,
                    "total_captures": 0,
                    "final_score": 100.0 if is_win else 0.0,
                })

            winner_name = None
            if self.winner is not None and self.seats.get(self.winner):
                winner_name = self.seats[self.winner].username

            save_match_record(
                session_id=f"room_{self.room_code}",
                match_mode=self.game_mode,
                winner_id=self.winner,
                winner_name=winner_name,
                winner_controller="HUMAN",
                total_turns=self.env.turn_count if self.env else 0,
                player_types={s: "human" for s in self.seats.keys()},
                player_scores={s: 100.0 if (self.seats[s] and participants[s]["is_winner"]) else 0.0 for s in self.seats.keys()},
                user_id=self.host_user_id,
                game_mode=self.game_mode,
                room_code=self.room_code,
                is_online=self.is_online,
                team_winner=self.team_winner,
                participants=participants,
            )
        except Exception:
            pass

    def render_svg(self, size: int = 560) -> str:
        """Generate authoritative SVG visualization for current board state."""
        if not self.env:
            return ""
        return render_ludo_board_svg(
            tokens=self.env.tokens,
            current_player=self.env.current_player,
            dice_value=self.current_dice_value or self.env.dice.value,
            legal_actions=self.legal_actions if self.turn_phase == TurnPhase.NEED_MOVE else [],
            selected_action=None,
            winner=self.winner,
            size=size,
        )

    def get_state_dict(self) -> Dict[str, Any]:
        """Return serializable summary of room and game state."""
        curr_p = self.env.current_player if self.env else 0
        active_seat = self.seats.get(curr_p) if self.env else None

        # Build player metadata list
        players = []
        for i in range(self.max_players):
            seat = self.seats.get(i)
            color_name = PLAYER_COLOR_NAMES_4P[i] if self.max_players == 4 else PLAYER_COLOR_NAMES_2P[i]
            tokens_fin = 0
            if self.env:
                tokens_fin = sum(1 for t in self.env.tokens if t.player_id == i and t.is_finished)

            players.append({
                "seat_id": i,
                "color": color_name,
                "occupied": seat is not None,
                "user_id": seat.user_id if seat else None,
                "username": seat.username if seat else "Open Seat",
                "team_id": seat.team_id if seat else (self.teams[i] if self.teams else None),
                "avatar_color": seat.avatar_color if seat else "slate",
                "rating": seat.rating if seat else 1000,
                "is_ready": seat.is_ready if seat else False,
                "is_connected": seat.is_connected if seat else False,
                "tokens_finished": tokens_fin,
            })

        return {
            "room_code": self.room_code,
            "game_mode": self.game_mode,
            "host_user_id": self.host_user_id,
            "host_username": self.host_username,
            "max_players": self.max_players,
            "player_count": self.get_player_count(),
            "is_online": self.is_online,
            "status": self.status.value,
            "turn_phase": self.turn_phase.value,
            "current_player": curr_p,
            "current_player_user_id": active_seat.user_id if active_seat else None,
            "current_player_username": active_seat.username if active_seat else None,
            "dice_value": self.current_dice_value,
            "legal_actions": self.legal_actions if self.turn_phase == TurnPhase.NEED_MOVE else [],
            "action_seq": self.action_seq,
            "turn_count": self.env.turn_count if self.env else 0,
            "winner": self.winner,
            "team_winner": self.team_winner,
            "players": players,
            "recent_history": self.move_history[-10:],
            "svg": self.render_svg() if self.status == RoomStatus.IN_PROGRESS else None,
        }


class RoomService:
    """Singleton service for managing active game rooms and room codes."""

    def __init__(self):
        self.rooms: Dict[str, GameRoom] = {}
        self.lock = threading.Lock()

    def generate_unique_code(self) -> str:
        """Generate a random unique 6-character room code."""
        with self.lock:
            for _ in range(1000):
                code = "".join(secrets.choice(ROOM_CODE_ALPHABET) for _ in range(ROOM_CODE_LENGTH))
                if code not in self.rooms:
                    return code
            # Fallback to timestamp suffix if space saturated
            return f"R{int(time.time()) % 100000:05d}"

    def create_room(
        self,
        game_mode: str,
        host_user_id: int,
        host_username: str,
        is_online: bool = True,
        host_avatar: str = "rose",
        host_rating: int = 1000,
    ) -> GameRoom:
        """Create a new game room and register host."""
        self.cleanup_expired_rooms()
        code = self.generate_unique_code()
        room = GameRoom(
            room_code=code,
            game_mode=game_mode,
            host_user_id=host_user_id,
            host_username=host_username,
            is_online=is_online,
            host_avatar=host_avatar,
            host_rating=host_rating,
        )
        with self.lock:
            self.rooms[code] = room
        return room

    def get_room(self, room_code: str) -> Optional[GameRoom]:
        """Fetch game room by code (case-insensitive)."""
        code = (room_code or "").strip().upper()
        return self.rooms.get(code)

    def join_room(
        self,
        room_code: str,
        user_id: int,
        username: str,
        avatar_color: str = "rose",
        rating: int = 1000,
        requested_seat: Optional[int] = None,
    ) -> Tuple[GameRoom, PlayerSeat]:
        """Add user to room by room code."""
        room = self.get_room(room_code)
        if not room:
            raise ValueError(f"Room code '{room_code}' was not found. Please verify the code.")

        seat = room.join(
            user_id=user_id,
            username=username,
            avatar_color=avatar_color,
            rating=rating,
            requested_seat=requested_seat,
        )
        return room, seat

    def leave_room(self, room_code: str, user_id: int) -> bool:
        """Remove user from room."""
        room = self.get_room(room_code)
        if not room:
            return False

        vacated = room.leave(user_id)
        # If room has no players left, clean it up
        if room.get_player_count() == 0:
            with self.lock:
                self.rooms.pop(room.room_code, None)
        return vacated is not None

    def cleanup_expired_rooms(self, max_idle_seconds: int = 3600) -> int:
        """Remove rooms inactive for more than max_idle_seconds."""
        now = time.time()
        to_delete = []
        with self.lock:
            for code, room in self.rooms.items():
                if now - room.last_activity > max_idle_seconds:
                    to_delete.append(code)
            for code in to_delete:
                del self.rooms[code]
        return len(to_delete)


# Singleton instance
room_service = RoomService()
