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
from web.utils.auth import save_match_record

# Color metadata
PLAYER_COLOR_NAMES = ["Red", "Green", "Yellow", "Blue"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# In-memory agent cache to eliminate repeated disk reads
_AGENT_CACHE: Dict[str, BaseAgent] = {}

# Canonical verified default checkpoints for trained models
DEFAULT_TRAINED_CHECKPOINTS: Dict[str, Path] = {
    "dqn": PROJECT_ROOT / "experiments" / "smoke_tests" / "SMOKE_DQN_structured_cnn_42_20261108_221833" / "checkpoints" / "final_model.pt",
    "ppo": PROJECT_ROOT / "experiments" / "smoke_tests" / "SMOKE_PPO_structured_cnn_42_20261108_221835" / "checkpoints" / "final_model.pt",
    "model3": PROJECT_ROOT / "experiments" / "smoke_tests" / "SMOKE_MODEL3_structured_cnn_42_20261108_221836" / "checkpoints" / "final_model.pt",
    "sac": PROJECT_ROOT / "experiments" / "smoke_tests" / "SMOKE_MODEL3_structured_cnn_42_20261108_221836" / "checkpoints" / "final_model.pt",
    "model4": PROJECT_ROOT / "experiments" / "smoke_tests" / "SMOKE_MODEL4_structured_cnn_42_20261108_221836" / "checkpoints" / "final_model.pt",
    "c51": PROJECT_ROOT / "experiments" / "smoke_tests" / "SMOKE_MODEL4_structured_cnn_42_20261108_221836" / "checkpoints" / "final_model.pt",
}


def resolve_checkpoint_for_algorithm(algorithm: str, checkpoint_path_str: Optional[str] = None) -> Path:
    """Resolve and validate a trained neural network checkpoint path.

    If checkpoint_path_str is explicitly provided:
        Verifies that the file exists and is a file.
    If checkpoint_path_str is None:
        Resolves the canonical default trained checkpoint for the algorithm.

    Raises:
        FileNotFoundError: If the specified or default checkpoint does not exist on disk.
        ValueError: If the algorithm is unsupported.
    """
    algo_clean = algorithm.strip().lower()
    if checkpoint_path_str:
        p = Path(checkpoint_path_str)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(
                f"Specified checkpoint not found: '{checkpoint_path_str}' (resolved to '{p}')."
            )
        return p

    # Fallback to canonical default checkpoint
    default_p = DEFAULT_TRAINED_CHECKPOINTS.get(algo_clean)
    if default_p and default_p.exists() and default_p.is_file():
        return default_p

    # Search experiments directory for any compatible final_model or algorithm checkpoint
    candidate_checkpoints = list(PROJECT_ROOT.glob(f"experiments/**/{algo_clean}*.pt")) + \
                            list(PROJECT_ROOT.glob("experiments/**/final_model.pt"))
    for cand in candidate_checkpoints:
        if algo_clean in str(cand).lower() and cand.is_file():
            return cand

    raise FileNotFoundError(
        f"No valid trained checkpoint found for algorithm '{algorithm}'. "
        f"Checked default: '{default_p}'."
    )


def get_cached_agent(algorithm: str, checkpoint_path_str: Optional[str] = None) -> BaseAgent:
    """Retrieve or instantiate a cached neural network agent with strictly loaded trained weights.

    Guarantees:
    - Never silently falls back to randomly initialized weights.
    - Caches separate agent instances per algorithm and checkpoint path.
    - Validates checkpoint compatibility upon loading.
    """
    algo_clean = algorithm.strip().lower()
    checkpoint_path = resolve_checkpoint_for_algorithm(algo_clean, checkpoint_path_str)

    cache_key = f"{algo_clean}_{checkpoint_path.resolve()}"
    if cache_key not in _AGENT_CACHE:
        agent = create_agent(algo_clean)
        try:
            agent.load_checkpoint(checkpoint_path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load trained checkpoint for '{algorithm}' from '{checkpoint_path}': {exc}"
            ) from exc
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
        num_players: int = 4,
        auto_advance_no_moves: Optional[bool] = None,
    ):
        self.session_id = session_id
        self.num_players = 2 if num_players == 2 else 4
        # Default player setup
        if self.num_players == 2:
            self.player_types = player_types or {0: "human", 1: "dqn"}
        else:
            self.player_types = player_types or {0: "dqn", 1: "expert", 2: "random", 3: "expert"}

        if algorithm:
            self.player_types[0] = algorithm
        if opponent_type:
            self.player_types[1] = opponent_type

        self.algorithm = str(self.player_types.get(0, "dqn"))
        self.checkpoint_path = checkpoint_path
        self.seed = seed

        # Interactive human session determination
        has_human = any(str(pt).lower() == "human" for pt in self.player_types.values())
        if auto_advance_no_moves is None:
            self.auto_advance_no_moves = False if has_human else True
        else:
            self.auto_advance_no_moves = auto_advance_no_moves

        self.turn_phase: str = "NEED_ROLL"

        # Authoritative Ludo Environment
        self.env = LudoEnvironment(
            seed=seed,
            max_turns=1000,
            auto_advance_no_moves=self.auto_advance_no_moves,
            num_players=self.num_players,
        )

        # Controllers and Opponents
        self.controllers: Dict[int, Any] = {}
        self.shared_expert = ExpertOpponent(seed=seed)
        self.shared_random = RandomOpponent(seed=seed)
        self._setup_controllers()
        self.agent = self.controllers.get(0, None)

        # Telemetry & Statistics
        self.player_rewards: Dict[int, float] = {p: 0.0 for p in range(self.num_players)}
        self.player_captures: Dict[int, int] = {p: 0 for p in range(self.num_players)}
        self.turn_history: List[Dict[str, Any]] = []
        self.last_action_taken: Optional[int] = None
        self.last_actor: Optional[str] = None
        self.is_terminal: bool = False
        self.winner: Optional[int] = None

        self.reset(seed)

    def _setup_controllers(self) -> None:
        """Instantiate controllers for active players with strict trained weights."""
        for pid in range(self.num_players):
            ctype = self.player_types.get(pid, "random").lower()
            if ctype in ["dqn", "ppo", "model3", "sac", "model4", "c51"]:
                algo = "model3" if ctype == "sac" else ("model4" if ctype == "c51" else ctype)
                # Strictly load verified trained checkpoint; never fall back to random weights
                self.controllers[pid] = get_cached_agent(algo, self.checkpoint_path)
            elif ctype == "expert":
                self.controllers[pid] = "expert"
            elif ctype == "human":
                self.controllers[pid] = "human"
            else:
                self.controllers[pid] = "random"

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset match to start state."""
        has_human = any(str(pt).lower() == "human" for pt in self.player_types.values())
        self.auto_advance_no_moves = False if has_human else True
        self.env.auto_advance_no_moves = self.auto_advance_no_moves

        if seed is not None:
            self.seed = seed
        self.env.reset(seed=self.seed)
        self.turn_phase = "NEED_ROLL"
        self.player_rewards = {p: 0.0 for p in range(self.num_players)}
        self.player_captures = {p: 0 for p in range(self.num_players)}
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

    def _pass_turn_no_moves(self, player_id: int, dice_val: int) -> None:
        """Explicitly record a no-legal-moves event and advance turn to next player."""
        color_name = "Yellow" if (self.num_players == 2 and player_id == 1) else PLAYER_COLOR_NAMES[player_id]
        ctype = str(self.player_types.get(player_id, "AI")).upper()
        actor_name = f"{ctype} (P{player_id} - {color_name})"

        self.env.turn_count += 1

        # Check truncation limit
        if self.env.turn_count >= self.env.max_turns:
            self.env.is_terminal = True
            self.is_terminal = True

        self.turn_history.append({
            "turn": self.env.turn_count,
            "player": f"P{player_id}",
            "actor": actor_name,
            "color": color_name,
            "dice": dice_val,
            "action": "Pass (No Legal Move)",
            "reward": "+0.00",
            "capture": "No",
            "bonus": "No",
            "status": f"Rolled {dice_val} with no legal moves; turn passed",
        })

        self.last_action_taken = None
        self.last_actor = actor_name

        if not self.env.is_terminal:
            self.env.consecutive_sixes = 0
            self.env.bonus_six = False
            self.env.bonus_capture = False
            self.env.current_player = (self.env.current_player + 1) % self.num_players

        self.turn_phase = "NEED_ROLL"
        self.is_terminal = self.env.is_terminal
        self.winner = self.env.winner

    def step(self) -> Tuple[bool, Optional[int]]:
        """Advance game by one player action."""
        if self.env.is_terminal:
            self.is_terminal = True
            self.winner = self.env.winner
            return True, None

        curr_p = self.env.current_player

        if not self.auto_advance_no_moves:
            # Interactive session: active player (AI) rolls dice at start of turn
            dice_val = self.env.dice.roll()
            legal_actions = self.env.get_legal_actions()
            if not legal_actions:
                self._pass_turn_no_moves(curr_p, dice_val)
                return self.is_terminal, None

            self.turn_phase = "NEED_MOVE"
            action = self._select_action_for_player(curr_p, legal_actions)
            is_term, act = self._apply_action(curr_p, action)
            self.turn_phase = "NEED_ROLL"
            return is_term, act
        else:
            # Autonomous AI tournament session: dice was pre-rolled by env
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
            curr_c = "Yellow" if (self.num_players == 2 and self.env.current_player == 1) else PLAYER_COLOR_NAMES[self.env.current_player]
            raise ValueError(
                f"Action rejected: It is Player {self.env.current_player}'s ({curr_c}) turn, "
                f"not Player {player_id}'s turn."
            )

        if not self.auto_advance_no_moves and self.turn_phase != "NEED_MOVE":
            raise ValueError("Action rejected: Must roll dice before selecting a token move.")

        legal_actions = self.env.get_legal_actions()
        if token_id not in legal_actions:
            raise ValueError(
                f"Action rejected: Token {token_id} is not a legal move for Player {player_id}. "
                f"Legal actions: {legal_actions}."
            )

        is_term, act = self._apply_action(player_id, token_id)
        if not self.auto_advance_no_moves:
            self.turn_phase = "NEED_ROLL"
        return is_term, act

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
            # Auto-save completed match to SQLite for realtime match history
            try:
                # Detect match mode based on human/ai configurations
                human_count = sum(1 for p in self.player_types.values() if p.lower() == "human")
                if human_count == 2:
                    mode = "2-Player (2 Humans vs 2 AI)"
                elif human_count >= 3:
                    mode = "Multi-Human"
                elif human_count == 1:
                    mode = "1 Human vs 3 AI"
                else:
                    mode = "Full AI Tournament"

                winner_name = (
                    ("Yellow" if (self.num_players == 2 and self.winner == 1) else PLAYER_COLOR_NAMES[self.winner])
                    if self.winner is not None
                    else None
                )
                winner_ctrl = str(self.player_types.get(self.winner, "AI")).upper() if self.winner is not None else None
                save_match_record(
                    session_id=self.session_id,
                    match_mode=mode,
                    winner_id=self.winner,
                    winner_name=winner_name,
                    winner_controller=winner_ctrl,
                    total_turns=self.env.turn_count,
                    player_types={p: str(ct) for p, ct in self.player_types.items()},
                    player_scores={p: float(self.player_rewards[p]) for p in range(self.num_players)},
                )
            except Exception:
                pass
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
        """Return compact statistics for active players."""
        stats: List[Dict[str, Any]] = []
        for pid in range(self.num_players):
            tokens = [t for t in self.env.tokens if t.player_id == pid]
            yard_count = sum(1 for t in tokens if t.is_in_yard)
            track_count = sum(1 for t in tokens if t.is_on_track or t.is_in_home)
            finished_count = sum(1 for t in tokens if t.is_finished)
            color_name = "Yellow" if (self.num_players == 2 and pid == 1) else PLAYER_COLOR_NAMES[pid]
            stats.append({
                "player_id": pid,
                "label": f"P{pid}",
                "color": color_name,
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
        if not self.auto_advance_no_moves and self.turn_phase == "NEED_ROLL":
            legal = []
        else:
            legal = self.env.get_legal_actions()
        curr_ctrl = str(self.player_types.get(curr_p, "random")).lower()
        curr_color = "Yellow" if (self.num_players == 2 and curr_p == 1) else PLAYER_COLOR_NAMES[curr_p]
        win_color = (
            ("Yellow" if (self.num_players == 2 and self.winner == 1) else PLAYER_COLOR_NAMES[self.winner])
            if self.winner is not None
            else None
        )
        mode_str = "Human vs AI (1v1)" if self.num_players == 2 else "Interactive 4-Player Demonstration"
        return {
            "session_id": self.session_id,
            "mode": mode_str,
            "num_players": self.num_players,
            "current_player": curr_p,
            "current_player_color": curr_color,
            "current_player_controller": curr_ctrl,
            "is_human_turn": (curr_ctrl == "human"),
            "dice_value": self.env.dice.value,
            "legal_actions": legal,
            "turn_phase": self.turn_phase,
            "is_terminal": self.is_terminal,
            "winner": self.winner,
            "winner_color": win_color,
            "turn_count": self.env.turn_count,
            "last_action": self.last_action_taken,
            "last_actor": self.last_actor,
            "players": self.get_player_statistics(),
            "recent_history": self.turn_history[-10:],
        }

    def roll_dice(self) -> int:
        """Roll dice for active player and update legal actions."""
        if self.is_terminal:
            return self.env.dice.value
        val = self.env.dice.roll()
        curr_p = self.env.current_player

        # If auto advance is enabled (autonomous AI tournament), let env advance
        if self.env.auto_advance_no_moves:
            if len(self.env.get_legal_actions()) == 0:
                self.env._advance_until_legal_or_terminal()
                self.is_terminal = self.env.is_terminal
                self.winner = self.env.winner
        else:
            # Interactive session: check if active player has legal moves
            legal_actions = self.env.get_legal_actions()
            if len(legal_actions) == 0:
                self._pass_turn_no_moves(curr_p, val)
            else:
                self.turn_phase = "NEED_MOVE"

        return val

    def render_svg(self, size: int = 560) -> str:
        """Render board as responsive SVG string."""
        legal = [] if (not self.auto_advance_no_moves and self.turn_phase == "NEED_ROLL") else self.env.get_legal_actions()
        return render_ludo_board_svg(
            tokens=self.env.tokens,
            current_player=self.env.current_player,
            dice_value=self.env.dice.value,
            legal_actions=legal,
            selected_action=self.last_action_taken,
            winner=self.winner,
            size=size,
        )


class GameService:
    """Singleton service managing interactive Ludo arena matches."""

    def __init__(self):
        self._sessions: Dict[str, GameSession4P] = {}

    def get_or_create_session(
        self,
        session_id: Optional[str] = None,
        player_types: Optional[Dict[int, str]] = None,
        checkpoint_path: Optional[str] = None,
        seed: Optional[int] = 42,
        num_players: Optional[int] = None,
    ) -> GameSession4P:
        if session_id and session_id in self._sessions:
            session = self._sessions[session_id]
            # Only recreate if explicit num_players is requested and differs from existing session
            if num_players is not None and session.num_players != num_players:
                pass
            else:
                if player_types is not None:
                    session.player_types = player_types
                    has_human = any(str(pt).lower() == "human" for pt in session.player_types.values())
                    session.auto_advance_no_moves = False if has_human else True
                    session.env.auto_advance_no_moves = session.auto_advance_no_moves
                    session._setup_controllers()
                return session

        n_players = 2 if num_players == 2 else 4
        s_id = session_id or str(uuid.uuid4())[:8]
        session = GameSession4P(
            session_id=s_id,
            player_types=player_types,
            checkpoint_path=checkpoint_path,
            seed=seed,
            num_players=n_players,
        )
        self._sessions[s_id] = session
        return session

    def reset_session(
        self,
        session_id: str,
        player_types: Optional[Dict[int, str]] = None,
        seed: Optional[int] = None,
        num_players: Optional[int] = None,
    ) -> GameSession4P:
        n_players = 2 if num_players == 2 else 4
        if session_id in self._sessions and (num_players is None or self._sessions[session_id].num_players == n_players):
            session = self._sessions[session_id]
            if player_types:
                session.player_types = player_types
                session._setup_controllers()
            session.reset(seed)
            return session
        return self.get_or_create_session(session_id, player_types, seed=seed, num_players=n_players)

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

    def roll_session_dice(self, session_id: str) -> Dict[str, Any]:
        session = self.get_or_create_session(session_id)
        dice_val = session.roll_dice()
        return {
            "dice_value": dice_val,
            "is_terminal": session.is_terminal,
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
