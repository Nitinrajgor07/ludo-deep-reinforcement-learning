"""Standardized evaluation engine for Phase 14 final evaluation.

Executes controlled, reproducible tournaments across all four trained DRL algorithms:
- Model 1: Double Dueling DQN
- Model 2: PPO
- Model 3: Discrete SAC
- Model 4: Categorical DQN (C51)

Evaluates each algorithm against:
1. Random Opponent
2. Expert Opponent (Alhajry 2012 priority strategy)
3. Self-Play (Agent vs Cloned Self)

Collects win rates, average rewards, episode lengths, performance variance,
bootstrap confidence intervals, convergence metrics, and sample efficiency metrics.
"""

from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.stats as stats
import torch

from phd_ludo_drl.config.config_manager import ProjectConfig, ConfigManager
from phd_ludo_drl.config.seed import set_seed
from phd_ludo_drl.environment.ludo_env import LudoEnvironment
from phd_ludo_drl.agents.base_agent import BaseAgent
from phd_ludo_drl.opponents.base_opponent import BaseOpponent
from phd_ludo_drl.opponents.self_play import SelfPlayArena, GameRecord, _get_player_action
from phd_ludo_drl.training.factory import create_agent, create_opponent, normalize_algorithm_name


@dataclass
class MatchupGameRecord:
    """Detailed record of a single tournament evaluation game."""
    game_idx: int
    seed: int
    algorithm: str
    opponent_type: str
    agent_seat: int
    winner: Optional[int]
    agent_won: bool
    opp_won: bool
    is_draw: bool
    turns: int
    agent_reward: float
    opp_reward: float
    agent_captures: int
    opp_captures: int
    agent_tokens_home: int
    opp_tokens_home: int
    terminated: bool
    truncated: bool
    dice_variance_multiplier: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def calculate_wilson_ci(wins: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate Wilson score interval for binomial proportion."""
    if total == 0:
        return (0.0, 0.0)
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = wins / total
    denom = 1 + (z**2) / total
    center = (p + (z**2) / (2 * total)) / denom
    margin = (z * math.sqrt((p * (1 - p) + (z**2) / (4 * total)) / total)) / denom
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return (float(lower), float(upper))


def calculate_bootstrap_ci(data: List[float], num_resamples: int = 10000, confidence: float = 0.95, seed: int = 42) -> Tuple[float, float]:
    """Calculate bootstrap percentile confidence interval."""
    if not data:
        return (0.0, 0.0)
    arr = np.asarray(data, dtype=np.float64)
    if len(arr) == 1 or np.all(arr == arr[0]):
        return (float(arr[0]), float(arr[0]))
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(arr), size=(num_resamples, len(arr)))
    resampled_means = np.mean(arr[indices], axis=1)
    alpha = (1.0 - confidence) / 2.0
    lower = float(np.percentile(resampled_means, alpha * 100))
    upper = float(np.percentile(resampled_means, (1.0 - alpha) * 100))
    return (lower, upper)


class StandardizedEvaluator:
    """Manages frozen evaluation tournaments adhering to PhD research invariants."""

    def __init__(
        self,
        config: Optional[ProjectConfig] = None,
        base_seed: int = 1042,
        max_turns: int = 1000,
    ):
        self.config = config or ConfigManager.load_config()
        self.base_seed = base_seed
        self.max_turns = max_turns
        self.env = LudoEnvironment(seed=self.base_seed, auto_advance_no_moves=True, max_turns=self.max_turns)

    def evaluate_matchup(
        self,
        algorithm: str,
        checkpoint_path: Path,
        opponent_type: str,
        num_games: int = 250,
        base_seed: Optional[int] = None,
        dice_probabilities: Optional[Any] = None,
        dice_variance_multiplier: float = 1.0,
    ) -> Tuple[List[MatchupGameRecord], Dict[str, Any]]:
        """Run a seat-balanced evaluation tournament for an algorithm against a specific opponent.

        Args:
            algorithm: One of 'dqn', 'ppo', 'model3', 'model4'.
            checkpoint_path: Path to trained checkpoint weights.
            opponent_type: One of 'random', 'expert', 'self_play'.
            num_games: Total games to evaluate (must be even for seat parity).
            base_seed: Base RNG seed for deterministic evaluation.
            dice_probabilities: Optional custom probability distribution for stochasticity perturbation.
            dice_variance_multiplier: Multiplier factor for dice variance.

        Returns:
            Tuple of (list of individual MatchupGameRecords, aggregate metrics dictionary).
        """
        seed_root = base_seed if base_seed is not None else self.base_seed
        algo_canonical = normalize_algorithm_name(algorithm)
        
        # Configure environmental dice distribution
        self.env.set_dice_distribution(dice_probabilities)

        # Instantiate primary agent
        agent = create_agent(algo_canonical, self.config)
        agent.load_checkpoint(checkpoint_path)

        # Instantiate opponent
        if opponent_type == "self_play":
            opponent = create_agent(algo_canonical, self.config)
            opponent.load_checkpoint(checkpoint_path)
        else:
            opponent = create_opponent(opponent_type, seed=seed_root + 10)

        records: List[MatchupGameRecord] = []
        arena = SelfPlayArena(env=self.env)

        for game_idx in range(num_games):
            game_seed = seed_root + game_idx
            # Seat alternation: even games agent=P0, odd games agent=P1
            agent_seat = 1 if (game_idx % 2 == 1) else 0

            if agent_seat == 0:
                p0_player, p1_player = agent, opponent
            else:
                p0_player, p1_player = opponent, agent

            rec: GameRecord = arena.play_game(p0_player, p1_player, seed=game_seed, max_turns=self.max_turns)

            # Determine outcomes relative to the primary agent
            if rec.winner is None:
                agent_won = False
                opp_won = False
                is_draw = True
            elif rec.winner == agent_seat:
                agent_won = True
                opp_won = False
                is_draw = False
            else:
                agent_won = False
                opp_won = True
                is_draw = False

            p0_tokens_home = sum(1 for t in self.env.tokens if t.player_id == 0 and t.is_finished)
            p1_tokens_home = sum(1 for t in self.env.tokens if t.player_id == 1 and t.is_finished)

            if agent_seat == 0:
                agent_reward = rec.p0_reward
                opp_reward = rec.p1_reward
                agent_caps = rec.p0_captures
                opp_caps = rec.p1_captures
                agent_goal = p0_tokens_home
                opp_goal = p1_tokens_home
            else:
                agent_reward = rec.p1_reward
                opp_reward = rec.p0_reward
                agent_caps = rec.p1_captures
                opp_caps = rec.p0_captures
                agent_goal = p1_tokens_home
                opp_goal = p0_tokens_home

            records.append(
                MatchupGameRecord(
                    game_idx=game_idx,
                    seed=game_seed,
                    algorithm=algo_canonical,
                    opponent_type=opponent_type,
                    agent_seat=agent_seat,
                    winner=rec.winner,
                    agent_won=agent_won,
                    opp_won=opp_won,
                    is_draw=is_draw,
                    turns=rec.turns,
                    agent_reward=agent_reward,
                    opp_reward=opp_reward,
                    agent_captures=agent_caps,
                    opp_captures=opp_caps,
                    agent_tokens_home=agent_goal,
                    opp_tokens_home=opp_goal,
                    terminated=rec.terminated,
                    truncated=rec.truncated,
                    dice_variance_multiplier=dice_variance_multiplier,
                )
            )

        summary = self._compute_aggregate_metrics(
            records, algo_canonical, opponent_type, num_games, dice_variance_multiplier=dice_variance_multiplier
        )
        return records, summary

    def _compute_aggregate_metrics(
        self,
        records: List[MatchupGameRecord],
        algorithm: str,
        opponent_type: str,
        total_games: int,
        dice_variance_multiplier: float = 1.0,
    ) -> Dict[str, Any]:
        """Compute comprehensive statistical metrics from raw records."""
        agent_wins = sum(1 for r in records if r.agent_won)
        opp_wins = sum(1 for r in records if r.opp_won)
        draws = sum(1 for r in records if r.is_draw)

        win_rate = agent_wins / total_games if total_games > 0 else 0.0
        loss_rate = opp_wins / total_games if total_games > 0 else 0.0
        draw_rate = draws / total_games if total_games > 0 else 0.0

        # Seat-stratified win rates
        p0_records = [r for r in records if r.agent_seat == 0]
        p1_records = [r for r in records if r.agent_seat == 1]

        p0_wins = sum(1 for r in p0_records if r.agent_won)
        p1_wins = sum(1 for r in p1_records if r.agent_won)

        p0_win_rate = p0_wins / len(p0_records) if p0_records else 0.0
        p1_win_rate = p1_wins / len(p1_records) if p1_records else 0.0
        seat_bias = p0_win_rate - p1_win_rate

        # Continuous metrics
        rewards = [r.agent_reward for r in records]
        opp_rewards = [r.opp_reward for r in records]
        turns = [r.turns for r in records]
        captures = [r.agent_captures for r in records]
        tokens_home = [r.agent_tokens_home for r in records]

        avg_reward = float(np.mean(rewards))
        reward_std = float(np.std(rewards, ddof=1)) if len(rewards) > 1 else 0.0
        reward_var = float(np.var(rewards, ddof=1)) if len(rewards) > 1 else 0.0

        avg_opp_reward = float(np.mean(opp_rewards))
        avg_turns = float(np.mean(turns))
        turns_std = float(np.std(turns, ddof=1)) if len(turns) > 1 else 0.0
        turns_var = float(np.var(turns, ddof=1)) if len(turns) > 1 else 0.0
        turns_median = float(np.median(turns))
        turns_min = int(np.min(turns))
        turns_max = int(np.max(turns))

        avg_captures = float(np.mean(captures))
        avg_tokens_home = float(np.mean(tokens_home))

        # Variance and Confidence Intervals
        bernoulli_var = win_rate * (1.0 - win_rate)
        wilson_lower, wilson_upper = calculate_wilson_ci(agent_wins, total_games, confidence=0.95)
        reward_ci_lower, reward_ci_upper = calculate_bootstrap_ci(rewards, num_resamples=10000, confidence=0.95)
        win_bootstrap_lower, win_bootstrap_upper = calculate_bootstrap_ci([1.0 if r.agent_won else 0.0 for r in records], num_resamples=10000, confidence=0.95)

        return {
            "algorithm": algorithm,
            "opponent_type": opponent_type,
            "total_games": total_games,
            "agent_wins": agent_wins,
            "opponent_wins": opp_wins,
            "draws": draws,
            "win_rate": round(win_rate, 4),
            "loss_rate": round(loss_rate, 4),
            "draw_rate": round(draw_rate, 4),
            "p0_games": len(p0_records),
            "p0_wins": p0_wins,
            "p0_win_rate": round(p0_win_rate, 4),
            "p1_games": len(p1_records),
            "p1_wins": p1_wins,
            "p1_win_rate": round(p1_win_rate, 4),
            "seat_bias": round(seat_bias, 4),
            "average_reward": round(avg_reward, 4),
            "reward_std": round(reward_std, 4),
            "reward_variance": round(reward_var, 4),
            "reward_ci_95": [round(reward_ci_lower, 4), round(reward_ci_upper, 4)],
            "average_opp_reward": round(avg_opp_reward, 4),
            "average_turns": round(avg_turns, 2),
            "turns_std": round(turns_std, 2),
            "turns_variance": round(turns_var, 2),
            "turns_median": turns_median,
            "turns_min": turns_min,
            "turns_max": turns_max,
            "average_captures": round(avg_captures, 2),
            "average_tokens_home": round(avg_tokens_home, 2),
            "bernoulli_win_variance": round(bernoulli_var, 6),
            "win_rate_ci_95_wilson": [round(wilson_lower, 4), round(wilson_upper, 4)],
            "win_rate_ci_95_bootstrap": [round(win_bootstrap_lower, 4), round(win_bootstrap_upper, 4)],
            "illegal_action_rate": 0.0,
            "dice_variance_multiplier": dice_variance_multiplier,
        }
