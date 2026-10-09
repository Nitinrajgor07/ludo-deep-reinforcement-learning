"""Game service managing environment interaction, 4-player turn progression, and AI agent inference.

Supports standard 4-Player Competitive Ludo Match:
- Player 0: Red
- Player 1: Green
- Player 2: Yellow
- Player 3: Blue

Strict Constraint: 100% Light Theme AI Research Standard. Zero Streamlit Dependency.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import uuid
import numpy as np

from phd_ludo_drl.environment.ludo_env import LudoEnvironment
from phd_ludo_drl.environment.token import Token
from phd_ludo_drl.environment.constants import TokenState, PLAYER_COLORS
from phd_ludo_drl.training.factory import create_agent, create_opponent
from phd_ludo_drl.agents.base_agent import BaseAgent
from phd_ludo_drl.opponents.base_opponent import BaseOpponent
from phd_ludo_drl.opponents.expert_opponent import ExpertOpponent
from phd_ludo_drl.opponents.random_opponent import RandomOpponent
from web.components.board_renderer import render_ludo_board_svg

# Color metadata
PLAYER_COLOR_NAMES = ["Red", "Green", "Yellow", "Blue"]

# In-memory agent cache to eliminate repeated disk reads
_AGENT_CACHE: Dict[str, BaseAgent] = {}


def get_cached_agent(algorithm: str, checkpoint_path_str: Optional[str] = None) -> BaseAgent:
    """Retrieve or instantiate a cached neural network agent."""
    cache_key = f"{algorithm.lower()}_{checkpoint_path_str or 'default'}"
    if cache_key not in _AGENT_CACHE:
        agent = create_agent(algorithm.lower())
        if checkpoint_path_str:
            p = Path(checkpoint_path_str)
            if p.exists():
                try:
                    agent.load_checkpoint(p)
                except Exception:
                    pass
        _AGENT_CACHE[cache_key] = agent
    return _AGENT_CACHE[cache_key]


class GameSession4P:
    """Encapsulates an authoritative 4-Player Ludo Match with independent agent controllers."""

    def __init__(
        self,
        session_id: str,
        player_types: Optional[Dict[int, str]] = None,
        checkpoint_path: Optional[str] = None,
        seed: Optional[int] = 42,
        algorithm: Optional[str] = None,
        opponent_type: Optional[str] = None,
    ):
        self.session_id = session_id
        # Default player setup: P0 = DQN, P1 = Expert, P2 = Random, P3 = Expert
        self.player_types = player_types or {0: "dqn", 1: "expert", 2: "random", 3: "expert"}
        if algorithm:
            self.player_types[0] = algorithm
        if opponent_type:
            self.player_types[1] = opponent_type

        self.algorithm = str(self.player_types.get(0, "dqn"))
        self.checkpoint_path = checkpoint_path
        self.seed = seed

        # Authoritative 4-Player Ludo Environment
        self.env = LudoEnvironment(
            seed=seed,
            max_turns=1000,
            auto_advance_no_moves=True,
            num_players=4,
        )

        # Controllers and Opponents
        self.controllers: Dict[int, Any] = {}
        self.shared_expert = ExpertOpponent(seed=seed)
        self.shared_random = RandomOpponent(seed=seed)
        self._setup_controllers()
        self.agent = self.controllers.get(0, None)

        # Telemetry & Statistics
        self.player_rewards: Dict[int, float] = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
        self.player_captures: Dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}
        self.turn_history: List[Dict[str, Any]] = []
        self.last_action_taken: Optional[int] = None
        self.last_actor: Optional[str] = None
        self.is_terminal: bool = False
        self.winner: Optional[int] = None

        self.reset(seed)

    def _setup_controllers(self) -> None:
        """Instantiate controllers for each of the 4 players."""
        for pid in range(4):
            ctype = self.player_types.get(pid, "random").lower()
            if ctype in ["dqn", "ppo", "model3", "sac", "model4", "c51"]:
                algo = "model3" if ctype == "sac" else ("model4" if ctype == "c51" else ctype)
                try:
                    self.controllers[pid] = get_cached_agent(algo, self.checkpoint_path)
                except Exception:
                    self.controllers[pid] = create_agent(algo)
            elif ctype == "expert":
                self.controllers[pid] = "expert"
            elif ctype == "human":
                self.controllers[pid] = "human"
            else:
                self.controllers[pid] = "random"

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset match to start state."""
        if seed is not None:
            self.seed = seed
        self.env.reset(seed=self.seed)
        self.player_rewards = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
        self.player_captures = {0: 0, 1: 0, 2: 0, 3: 0}
        self.turn_history = []
        self.last_action_taken = None
        self.last_actor = None
        self.is_terminal = False
        self.winner = None

    def _select_action_for_player(self, pid: int, legal_actions: List[int]) -> int:
        """Select action using player's configured controller."""
        controller = self.controllers.get(pid, "random")

        if controller == "human":
            # Default to first legal action if stepped automatically
            return legal_actions[0]

        if controller == "expert":
            return self.shared_expert.select_action(self.env.get_state_dict(), legal_actions)

        if controller == "random":
            return self.shared_random.select_action(self.env.get_state_dict(), legal_actions)

        # DRL Agent (BaseAgent)
        if isinstance(controller, BaseAgent):
            action_mask = self.env.get_action_mask()
            # Project state from perspective of player `pid`
            friendly = [t.copy() for t in self.env.tokens if t.player_id == pid]
            opponents = [t.copy() for t in self.env.tokens if t.player_id != pid][:4]
            projected_state = {
                "current_player": 0,
                "dice_value": self.env.dice.value,
                "tokens": friendly + opponents,
                "action_mask": action_mask,
                "is_terminal": self.env.is_terminal,
                "winner": 0 if self.env.winner == pid else (1 if self.env.winner is not None else None),
                "turn_count": self.env.turn_count,
            }
            try:
                act = controller.select_action(projected_state, action_mask, deterministic=True)
                if act in legal_actions:
                    return act
            except Exception:
                pass

        return legal_actions[0]

    def step(self) -> Tuple[bool, Optional[int]]:
        """Advance game by one player action."""
        if self.env.is_terminal:
            self.is_terminal = True
            self.winner = self.env.winner
            return True, None

        curr_p = self.env.current_player
        legal_actions = self.env.get_legal_actions()

        if not legal_actions:
            self.is_terminal = self.env.is_terminal
            self.winner = self.env.winner
            return self.is_terminal, None

        action = self._select_action_for_player(curr_p, legal_actions)
        return self._apply_action(curr_p, action)

    def execute_manual_action(self, player_id: int, token_id: int) -> Tuple[bool, Optional[int]]:
        """Execute a human player token move after strict server-side validation."""
        if self.env.is_terminal:
            return True, None

        if player_id != self.env.current_player:
            raise ValueError(
                f"Action rejected: It is Player {self.env.current_player}'s ({PLAYER_COLOR_NAMES[self.env.current_player]}) turn, "
                f"not Player {player_id}'s turn."
            )

        legal_actions = self.env.get_legal_actions()
        if token_id not in legal_actions:
            raise ValueError(
                f"Action rejected: Token {token_id} is not a legal move for Player {player_id}. "
                f"Legal actions: {legal_actions}."
            )

        return self._apply_action(player_id, token_id)

    def _apply_action(self, player_id: int, token_id: int) -> Tuple[bool, Optional[int]]:
        """Apply move, record telemetry, and update statistics."""
        color_name = PLAYER_COLOR_NAMES[player_id]
        ctype = str(self.player_types.get(player_id, "AI")).upper()
        actor_name = f"{ctype} (P{player_id} - {color_name})"

        dice_used = self.env.dice.value
        obs, reward, terminated, truncated, info = self.env.step(token_id)

        # Update player metrics
        self.player_rewards[player_id] += float(reward)
        move_res = info.get("move_result", None)
        captures_count = len(move_res.captured_tokens) if move_res else 0
        if captures_count > 0:
            self.player_captures[player_id] += captures_count

        self.last_action_taken = token_id
        self.last_actor = actor_name
        self.is_terminal = self.env.is_terminal
        self.winner = self.env.winner

        # Formulate status message
        if self.is_terminal and self.winner == player_id:
            status = "Won Match"
        elif move_res and move_res.finished:
            status = "Token Reached Goal"
        elif captures_count > 0:
            status = f"Captured {captures_count} Token(s)"
        elif move_res and move_res.released:
            status = "Released from Yard"
        else:
            status = "Advanced on Track"

        self.turn_history.append({
            "turn": self.env.turn_count,
            "player": f"P{player_id}",
            "actor": actor_name,
            "color": color_name,
            "dice": dice_used,
            "action": f"Move Token {token_id}",
            "reward": f"{reward:+.2f}",
            "capture": "Yes" if captures_count > 0 else "No",
            "bonus": "Yes" if (info.get("bonus_six", False) or info.get("bonus_capture", False)) else "No",
            "status": status,
        })

        return self.is_terminal, token_id

    def get_player_statistics(self) -> List[Dict[str, Any]]:
        """Return compact statistics for each of the 4 players."""
        stats: List[Dict[str, Any]] = []
        for pid in range(4):
            tokens = [t for t in self.env.tokens if t.player_id == pid]
            yard_count = sum(1 for t in tokens if t.is_in_yard)
            track_count = sum(1 for t in tokens if t.is_on_track or t.is_in_home)
            finished_count = sum(1 for t in tokens if t.is_finished)
            stats.append({
                "player_id": pid,
                "label": f"P{pid}",
                "color": PLAYER_COLOR_NAMES[pid],
                "controller": str(self.player_types.get(pid, "AI")).upper(),
                "yard_count": yard_count,
                "track_count": track_count,
                "finished_count": finished_count,
                "captures": self.player_captures[pid],
                "reward": round(self.player_rewards[pid], 2),
            })
        return stats

    def get_summary(self) -> Dict[str, Any]:
        """Return comprehensive JSON serializable game state."""
        curr_p = self.env.current_player
        legal = self.env.get_legal_actions()
        return {
            "session_id": self.session_id,
            "mode": "Interactive 4-Player Demonstration",
            "current_player": curr_p,
            "current_player_color": PLAYER_COLOR_NAMES[curr_p],
            "dice_value": self.env.dice.value,
            "legal_actions": legal,
            "is_terminal": self.is_terminal,
            "winner": self.winner,
            "winner_color": PLAYER_COLOR_NAMES[self.winner] if self.winner is not None else None,
            "turn_count": self.env.turn_count,
            "last_action": self.last_action_taken,
            "last_actor": self.last_actor,
            "players": self.get_player_statistics(),
            "recent_history": self.turn_history[-10:],
        }

    def render_svg(self, size: int = 560) -> str:
        """Render 4-player board as responsive SVG string."""
        return render_ludo_board_svg(
            tokens=self.env.tokens,
            current_player=self.env.current_player,
            dice_value=self.env.dice.value,
            legal_actions=self.env.get_legal_actions(),
            selected_action=self.last_action_taken,
            winner=self.winner,
            size=size,
        )


class GameService:
    """Singleton service managing interactive 4-Player Ludo arena matches."""

    def __init__(self):
        self._sessions: Dict[str, GameSession4P] = {}

    def get_or_create_session(
        self,
        session_id: Optional[str] = None,
        player_types: Optional[Dict[int, str]] = None,
        checkpoint_path: Optional[str] = None,
        seed: Optional[int] = 42,
    ) -> GameSession4P:
        if not session_id or session_id not in self._sessions:
            s_id = session_id or str(uuid.uuid4())[:8]
            session = GameSession4P(
                session_id=s_id,
                player_types=player_types,
                checkpoint_path=checkpoint_path,
                seed=seed,
            )
            self._sessions[s_id] = session
            return session

        session = self._sessions[session_id]
        if player_types is not None:
            session.player_types = player_types
            session._setup_controllers()
        return session

    def reset_session(
        self,
        session_id: str,
        player_types: Optional[Dict[int, str]] = None,
        seed: Optional[int] = None,
    ) -> GameSession4P:
        if session_id in self._sessions:
            session = self._sessions[session_id]
            if player_types:
                session.player_types = player_types
                session._setup_controllers()
            session.reset(seed)
            return session
        return self.get_or_create_session(session_id, player_types, seed=seed)

    def step_session(self, session_id: str) -> Dict[str, Any]:
        session = self.get_or_create_session(session_id)
        is_terminal, action = session.step()
        return {
            "is_terminal": is_terminal,
            "action": action,
            "state": session.get_summary(),
            "svg": session.render_svg(),
        }

    def manual_step_session(self, session_id: str, player_id: int, token_id: int) -> Dict[str, Any]:
        session = self.get_or_create_session(session_id)
        is_terminal, act = session.execute_manual_action(player_id, token_id)
        return {
            "is_terminal": is_terminal,
            "action": act,
            "state": session.get_summary(),
            "svg": session.render_svg(),
        }

    def run_n_turns(self, session_id: str, n: int = 5) -> Dict[str, Any]:
        session = self.get_or_create_session(session_id)
        for _ in range(n):
            if session.is_terminal:
                break
            session.step()
        return {
            "is_terminal": session.is_terminal,
            "state": session.get_summary(),
            "svg": session.render_svg(),
        }


# Global singleton instance
game_service = GameService()

# Backward-compatible alias
GameSession = GameSession4P
