"""Training coordinator and rollout execution for PPO baseline."""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import numpy as np
import torch

from phd_ludo_drl.config.config_manager import PPOConfig
from phd_ludo_drl.environment.ludo_env import LudoEnvironment
from phd_ludo_drl.encoding.structured_representation import build_structured_representation
from phd_ludo_drl.agents.ppo.ppo_agent import PPOAgent
from phd_ludo_drl.agents.ppo.rollout_buffer import PPORolloutBuffer
from phd_ludo_drl.opponents.base_opponent import BaseOpponent
from phd_ludo_drl.opponents.random_opponent import RandomOpponent
from phd_ludo_drl.opponents.self_play import SelfPlayArena, TournamentSummary


@dataclass
class PPOTrainingMetrics:
    """Summary of PPO training execution."""
    total_steps: int
    episodes: int
    ppo_wins: int
    opponent_wins: int
    draws: int
    win_rate: float
    avg_reward: float
    final_loss: float
    final_policy_loss: float
    final_value_loss: float
    final_entropy: float
    updates: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PPOTrainer:
    """Coordinates PPO rollout collection, GAE calculation, and policy updates."""

    def __init__(
        self,
        agent: Optional[PPOAgent] = None,
        opponent: Optional[BaseOpponent] = None,
        env: Optional[LudoEnvironment] = None,
        ppo_player_id: int = 0,
        rollout_steps: int = 128,
    ):
        self.agent = agent or PPOAgent()
        self.opponent = opponent or RandomOpponent()
        self.env = env or LudoEnvironment(auto_advance_no_moves=True)
        self.ppo_player_id = ppo_player_id
        self.rollout_steps = rollout_steps
        self.arena = SelfPlayArena(env=self.env)
        self.buffer = PPORolloutBuffer(capacity=self.rollout_steps)

    def train(
        self,
        total_steps: int = 1000,
        eval_interval: Optional[int] = None,
        eval_games: int = 10,
        checkpoint_dir: Optional[Path] = None,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> PPOTrainingMetrics:
        """Execute PPO training loop for total_steps interactions.

        Args:
            total_steps: Total interaction steps of the PPO agent.
            eval_interval: Frequency of periodic evaluation (optional).
            eval_games: Number of evaluation games per interval.
            checkpoint_dir: Directory to save model snapshots.
            callback: Optional callback receiving step metrics.

        Returns:
            PPOTrainingMetrics summarizing training execution.
        """
        obs, info = self.env.reset()
        episode_reward = 0.0
        episode_count = 0
        ppo_wins = 0
        opp_wins = 0
        draws = 0
        all_episode_rewards: List[float] = []

        last_loss = 0.0
        last_policy_loss = 0.0
        last_value_loss = 0.0
        last_entropy = 0.0
        updates = 0

        step = 0
        while step < total_steps:
            curr_p = self.env.current_player
            legal_actions = self.env.get_legal_actions()

            if not legal_actions:
                obs, info = self.env.reset()
                continue

            if curr_p == self.ppo_player_id:
                # PPO's turn: collect on-policy transition
                struct_state = build_structured_representation(obs)
                mask = info["action_mask"]

                action, log_prob, value = self.agent.step(
                    state=struct_state.spatial_tensor,
                    action_mask=mask,
                    deterministic=False,
                )

                next_obs, reward, term, trunc, step_info = self.env.step(action)
                done = term or trunc

                self.buffer.add(
                    state=struct_state.spatial_tensor,
                    context=struct_state.context_vector,
                    action=action,
                    action_mask=mask,
                    reward=reward,
                    value=value,
                    log_prob=log_prob,
                    done=done,
                )

                episode_reward += reward
                step += 1
                obs = next_obs
                info = step_info

                if done:
                    all_episode_rewards.append(episode_reward)
                    if self.env.winner == self.ppo_player_id:
                        ppo_wins += 1
                    elif self.env.winner is not None:
                        opp_wins += 1
                    else:
                        draws += 1

                    episode_count += 1
                    episode_reward = 0.0
                    obs, info = self.env.reset()

                # If rollout buffer is full or we reached total_steps, perform PPO update
                if len(self.buffer) == self.rollout_steps or step >= total_steps:
                    # Compute bootstrap value for the final state if not done
                    if done:
                        last_value = 0.0
                    else:
                        last_struct = build_structured_representation(obs)
                        last_value = self.agent.get_value(last_struct.spatial_tensor)

                    self.buffer.compute_gae(
                        last_value=last_value,
                        last_done=done,
                        gamma=self.agent.config.gamma,
                        gae_lambda=self.agent.config.gae_lambda,
                    )

                    update_metrics = self.agent.update(self.buffer)
                    if update_metrics:
                        last_loss = update_metrics["loss"]
                        last_policy_loss = update_metrics["policy_loss"]
                        last_value_loss = update_metrics["value_loss"]
                        last_entropy = update_metrics["entropy"]
                        updates = update_metrics["updates"]

                    self.buffer.reset()

                    if callback:
                        callback({
                            "step": step,
                            "updates": updates,
                            "loss": last_loss,
                            "policy_loss": last_policy_loss,
                            "value_loss": last_value_loss,
                            "entropy": last_entropy,
                        })

            else:
                # Opponent's turn
                opp_action = self.opponent.select_action(obs, legal_actions)
                next_obs, opp_reward, term, trunc, step_info = self.env.step(opp_action)
                obs = next_obs
                info = step_info

                if term or trunc:
                    all_episode_rewards.append(episode_reward)
                    if self.env.winner == self.ppo_player_id:
                        ppo_wins += 1
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

            # Checkpoint saving
            if checkpoint_dir is not None and eval_interval is not None and step > 0 and step % eval_interval == 0:
                ckpt_path = checkpoint_dir / f"ppo_step_{step}.pt"
                self.agent.save_checkpoint(ckpt_path)

        total_games = ppo_wins + opp_wins + draws
        win_rate = (ppo_wins / total_games) if total_games > 0 else 0.0
        avg_reward = float(np.mean(all_episode_rewards)) if all_episode_rewards else 0.0

        return PPOTrainingMetrics(
            total_steps=step,
            episodes=episode_count,
            ppo_wins=ppo_wins,
            opponent_wins=opp_wins,
            draws=draws,
            win_rate=float(win_rate),
            avg_reward=avg_reward,
            final_loss=last_loss,
            final_policy_loss=last_policy_loss,
            final_value_loss=last_value_loss,
            final_entropy=last_entropy,
            updates=updates,
        )

    def evaluate(self, num_games: int = 20) -> TournamentSummary:
        """Run tournament evaluation between PPO agent and opponent."""
        return self.arena.play_tournament(
            player_a=self.agent,
            player_b=self.opponent,
            num_games=num_games,
            alternate_seats=True,
        )

