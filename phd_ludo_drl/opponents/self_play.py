"""Standardized self-play arena and tournament evaluation interface."""

from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch

from phd_ludo_drl.environment.ludo_env import LudoEnvironment
from phd_ludo_drl.opponents.base_opponent import BaseOpponent
from phd_ludo_drl.agents.base_agent import BaseAgent


@dataclass
class GameRecord:
    """Summary of a single completed self-play game."""
    winner: Optional[int]
    turns: int
    p0_reward: float
    p1_reward: float
    p0_captures: int
    p1_captures: int
    terminated: bool
    truncated: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TournamentSummary:
    """Aggregate statistics over a series of self-play evaluation games."""
    total_games: int
    p0_wins: int
    p1_wins: int
    draws: int
    p0_win_rate: float
    p1_win_rate: float
    draw_rate: float
    avg_turns: float
    avg_p0_reward: float
    avg_p1_reward: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _get_player_action(
    player: Union[BaseOpponent, BaseAgent, Callable],
    state_dict: Dict[str, Any],
    legal_actions: List[int],
    action_mask: np.ndarray,
) -> int:
    """Universal adapter querying an action from BaseOpponent, BaseAgent, or Callable."""
    if not legal_actions:
        raise ValueError("Cannot query action when legal_actions is empty.")

    if isinstance(player, BaseOpponent):
        action = player.select_action(state_dict, legal_actions)
    elif isinstance(player, BaseAgent):
        # BaseAgent expects (state, action_mask)
        action = player.select_action(state_dict, action_mask, deterministic=True)
    elif callable(player):
        try:
            action = player(state_dict, legal_actions)
        except TypeError:
            action = player(state_dict)
    else:
        raise TypeError(f"Unsupported player type: {type(player).__name__}")

    # Enforce legal action invariant
    if action not in legal_actions:
        raise ValueError(f"Player {getattr(player, 'name', player)} selected illegal action {action}! Legal: {legal_actions}")

    return int(action)


class SelfPlayArena:
    """Manages head-to-head matches and tournaments between agents and opponents."""

    def __init__(self, env: Optional[LudoEnvironment] = None):
        self.env = env or LudoEnvironment(auto_advance_no_moves=True)

    def play_game(
        self,
        player_0: Union[BaseOpponent, BaseAgent, Callable],
        player_1: Union[BaseOpponent, BaseAgent, Callable],
        seed: Optional[int] = None,
        max_turns: int = 1000,
    ) -> GameRecord:
        """Run a single complete game between player_0 (Player 0) and player_1 (Player 1).

        Args:
            player_0: First player (Player 0).
            player_1: Second player (Player 1).
            seed: Optional seed for deterministic environment initialization.
            max_turns: Turn cutoff limit.

        Returns:
            GameRecord detailing outcome, rewards, and turn statistics.
        """
        self.env.max_turns = max_turns
        obs, info = self.env.reset(seed=seed)

        p0_total_reward = 0.0
        p1_total_reward = 0.0
        p0_captures = 0
        p1_captures = 0

        players = {0: player_0, 1: player_1}

        while not self.env.is_terminal:
            curr_player_id = self.env.current_player
            legal_actions = self.env.get_legal_actions()

            if not legal_actions:
                # No legal actions available; break to prevent deadlock
                break

            active_player = players[curr_player_id]
            action = _get_player_action(
                active_player,
                obs,
                legal_actions,
                self.env.get_action_mask(),
            )

            obs, reward, term, trunc, step_info = self.env.step(action)

            # Track rewards and captures
            if curr_player_id == 0:
                p0_total_reward += reward
            else:
                p1_total_reward += reward

            if "move_result" in step_info:
                caps = len(step_info["move_result"].captured_tokens)
                if curr_player_id == 0:
                    p0_captures += caps
                else:
                    p1_captures += caps

            if term or trunc:
                break

        return GameRecord(
            winner=self.env.winner,
            turns=self.env.turn_count,
            p0_reward=float(p0_total_reward),
            p1_reward=float(p1_total_reward),
            p0_captures=p0_captures,
            p1_captures=p1_captures,
            terminated=self.env.is_terminal and self.env.winner is not None,
            truncated=self.env.turn_count >= max_turns,
        )

    def play_tournament(
        self,
        player_a: Union[BaseOpponent, BaseAgent, Callable],
        player_b: Union[BaseOpponent, BaseAgent, Callable],
        num_games: int = 100,
        alternate_seats: bool = True,
        base_seed: int = 42,
    ) -> TournamentSummary:
        """Run a tournament of multiple games to assess comparative win rates.

        Args:
            player_a: Primary competitor (e.g. RL Model or Expert).
            player_b: Baseline opponent (e.g. Random or Heuristic).
            num_games: Total number of games.
            alternate_seats: If True, alternates seats (P0/P1) to eliminate first-player bias.
            base_seed: Base random seed for reproducible tournament episodes.

        Returns:
            TournamentSummary containing win rates and aggregated metrics.
        """
        a_wins = 0
        b_wins = 0
        draws = 0
        total_turns = 0
        total_a_reward = 0.0
        total_b_reward = 0.0

        for game_idx in range(num_games):
            seed = base_seed + game_idx

            if alternate_seats and (game_idx % 2 == 1):
                # Seat swap: Player B is P0, Player A is P1
                record = self.play_game(player_b, player_a, seed=seed)
                if record.winner == 1:
                    a_wins += 1
                elif record.winner == 0:
                    b_wins += 1
                else:
                    draws += 1
                total_a_reward += record.p1_reward
                total_b_reward += record.p0_reward
            else:
                # Standard seat: Player A is P0, Player B is P1
                record = self.play_game(player_a, player_b, seed=seed)
                if record.winner == 0:
                    a_wins += 1
                elif record.winner == 1:
                    b_wins += 1
                else:
                    draws += 1
                total_a_reward += record.p0_reward
                total_b_reward += record.p1_reward

            total_turns += record.turns

        return TournamentSummary(
            total_games=num_games,
            p0_wins=a_wins,
            p1_wins=b_wins,
            draws=draws,
            p0_win_rate=a_wins / num_games,
            p1_win_rate=b_wins / num_games,
            draw_rate=draws / num_games,
            avg_turns=total_turns / num_games,
            avg_p0_reward=total_a_reward / num_games,
            avg_p1_reward=total_b_reward / num_games,
        )
