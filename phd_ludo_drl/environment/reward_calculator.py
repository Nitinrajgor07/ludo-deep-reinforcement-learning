"""Domain-grounded reward calculation matching published IEEE paper values."""

from typing import Dict, List, Optional
from phd_ludo_drl.config.config_manager import RewardConfig
from phd_ludo_drl.environment.rules import MoveResult
from phd_ludo_drl.environment.token import Token


class LudoRewardCalculator:
    """Calculates shaped reward based on move events and terminal state."""

    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig()

    def compute_move_reward(
        self,
        move_result: MoveResult,
        is_terminal: bool,
        winner: Optional[int],
    ) -> float:
        """Compute immediate shaped reward for the acting player."""
        reward = 0.0

        if move_result.released:
            reward += self.config.release

        if move_result.moved_to_safe:
            reward += self.config.defend

        if len(move_result.captured_tokens) > 0:
            reward += self.config.capture * len(move_result.captured_tokens)

        # Progress reward for advancing towards home / goal
        if not move_result.released and move_result.new_step > move_result.previous_step:
            reward += self.config.advance

        if move_result.formed_blockade:
            reward += self.config.blockade

        if is_terminal and winner is not None:
            if winner == move_result.player_id:
                reward += self.config.win
            else:
                reward += self.config.loss

        return float(reward)

    def get_illegal_action_penalty(self) -> float:
        """Penalty applied when an agent attempts an illegal action."""
        return float(self.config.illegal_penalty)
