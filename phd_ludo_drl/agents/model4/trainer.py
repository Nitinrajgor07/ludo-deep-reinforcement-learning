"""Training coordinator and evaluation manager for Categorical DQN (C51 / Model 4)."""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import numpy as np
import torch

from phd_ludo_drl.config.config_manager import CategoricalDQNConfig
from phd_ludo_drl.environment.ludo_env import LudoEnvironment
from phd_ludo_drl.encoding.structured_representation import build_structured_representation
from phd_ludo_drl.agents.model4.c51_agent import CategoricalDQNAgent
from phd_ludo_drl.opponents.base_opponent import BaseOpponent
from phd_ludo_drl.opponents.random_opponent import RandomOpponent
from phd_ludo_drl.opponents.self_play import SelfPlayArena, TournamentSummary


@dataclass
class C51TrainingMetrics:
    """Summary of Categorical DQN training execution."""
    total_steps: int
    episodes: int
    c51_wins: int
    opponent_wins: int
    draws: int
    win_rate: float
    avg_reward: float
    final_loss: float
    final_epsilon: float
    updates: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class C51Trainer:
    """Coordinates training of Categorical DQN (C51) agent against benchmark opponents."""

    def __init__(
        self,
        agent: Optional[CategoricalDQNAgent] = None,
        opponent: Optional[BaseOpponent] = None,
        env: Optional[LudoEnvironment] = None,
        c51_player_id: int = 0,
        batch_size: int = 64,
        warmup_steps: int = 200,
    ):
        self.agent = agent or CategoricalDQNAgent()
        self.opponent = opponent or RandomOpponent()
        self.env = env or LudoEnvironment(auto_advance_no_moves=True)
        self.c51_player_id = c51_player_id
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.arena = SelfPlayArena(env=self.env)

    def train(
        self,
        total_steps: int = 1000,
        eval_interval: Optional[int] = None,
        eval_games: int = 10,
        checkpoint_dir: Optional[Path] = None,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> C51TrainingMetrics:
        """Execute Categorical DQN training loop for total_steps interactions.

        Args:
            total_steps: Total interaction steps to train for.
            eval_interval: Frequency of periodic tournament evaluation.
            eval_games: Number of evaluation games per interval.
            checkpoint_dir: Directory to save model snapshots.
            callback: Optional callback receiving step metrics.

        Returns:
            C51TrainingMetrics summarizing training execution.
        """
        obs, info = self.env.reset()
        episode_reward = 0.0
        episode_count = 0
        c51_wins = 0
        opp_wins = 0
        draws = 0
        all_episode_rewards: List[float] = []

        last_loss = 0.0
        updates = 0

        step = 0
        while step < total_steps:
            curr_p = self.env.current_player
            legal_actions = self.env.get_legal_actions()

            if not legal_actions:
                obs, info = self.env.reset()
                continue

            if curr_p == self.c51_player_id:
                # C51 agent's turn
                struct_state = build_structured_representation(obs)
                mask = info["action_mask"]

                action = self.agent.select_action(
                    struct_state.spatial_tensor,
                    mask,
                    deterministic=False,
                )

                next_obs, reward, term, trunc, step_info = self.env.step(action)
                next_struct = build_structured_representation(next_obs)
                next_mask = step_info["action_mask"]
                done = term or trunc

                # Store transition in replay buffer
                self.agent.buffer.push(
                    state=struct_state.spatial_tensor,
                    context=struct_state.context_vector,
                    action=action,
                    reward=reward,
                    next_state=next_struct.spatial_tensor,
                    next_context=next_struct.context_vector,
                    done=done,
                    action_mask=mask,
                    next_action_mask=next_mask,
                )

                # Train step if warmup completed
                if step >= self.warmup_steps and len(self.agent.buffer) >= self.batch_size:
                    train_info = self.agent.train_step(self.batch_size)
                    if train_info:
                        last_loss = train_info["loss"]
                        updates = train_info["updates"]

                        if callback:
                            callback(train_info)

                episode_reward += reward
                step += 1
                obs = next_obs
                info = step_info

                if done:
                    all_episode_rewards.append(episode_reward)
                    if self.env.winner == self.c51_player_id:
                        c51_wins += 1
                    elif self.env.winner is not None:
                        opp_wins += 1
                    else:
                        draws += 1

                    episode_count += 1
                    episode_reward = 0.0
                    obs, info = self.env.reset()

            else:
                # Opponent's turn
                opp_action = self.opponent.select_action(obs, legal_actions)
                next_obs, opp_reward, term, trunc, step_info = self.env.step(opp_action)
                obs = next_obs
                info = step_info

                if term or trunc:
                    all_episode_rewards.append(episode_reward)
                    if self.env.winner == self.c51_player_id:
                        c51_wins += 1
                    elif self.env.winner is not None:
                        opp_wins += 1
                    else:
                        draws += 1

                    episode_count += 1
                    episode_reward = 0.0
                    obs, info = self.env.reset()

            # Periodic evaluation
            if eval_interval is not None and step > 0 and step % eval_interval == 0:
                self.evaluate(eval_games)

            # Checkpoint snapshot
            if checkpoint_dir is not None and eval_interval is not None and step > 0 and step % eval_interval == 0:
                ckpt_path = checkpoint_dir / f"c51_step_{step}.pt"
                self.agent.save_checkpoint(ckpt_path)

        total_games = c51_wins + opp_wins + draws
        win_rate = (c51_wins / total_games) if total_games > 0 else 0.0
        avg_reward = float(np.mean(all_episode_rewards)) if all_episode_rewards else 0.0

        return C51TrainingMetrics(
            total_steps=step,
            episodes=episode_count,
            c51_wins=c51_wins,
            opponent_wins=opp_wins,
            draws=draws,
            win_rate=float(win_rate),
            avg_reward=avg_reward,
            final_loss=last_loss,
            final_epsilon=float(self.agent.epsilon),
            updates=updates,
        )

    def evaluate(self, num_games: int = 20) -> TournamentSummary:
        """Run tournament evaluation between C51 agent and opponent."""
        return self.arena.play_tournament(
            player_a=self.agent,
            player_b=self.opponent,
            num_games=num_games,
            alternate_seats=True,
        )
