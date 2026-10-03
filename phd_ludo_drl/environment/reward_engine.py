"""Comprehensive Reward Engine implementing the published IEEE baseline reward framework.

Supports all 8 primary reward events:
- Token release (+0.25)
- Defensive move (+0.20)
- Capture (+0.15)
- Progress (+0.10)
- Blockade (+0.05)
- Win (+1.00)
- Token captured (-0.25 for victim)
- Loss (-1.00)
- Out-of-distribution illegal penalty (-0.50)

Ensures strict parity across all 4 RL models (DQN, PPO, Model 3, Model 4).
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
from phd_ludo_drl.config.config_manager import RewardConfig
from phd_ludo_drl.environment.rules import MoveResult
from phd_ludo_drl.environment.token import Token


@dataclass
class RewardBreakdown:
    """Itemized breakdown of reward components for research logging."""
    release: float = 0.0
    defend: float = 0.0
    capture: float = 0.0
    advance: float = 0.0
    blockade: float = 0.0
    win: float = 0.0
    knocked: float = 0.0
    loss: float = 0.0
    illegal_penalty: float = 0.0
    total: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


class RewardEngine:
    """Authoritative Reward Engine implementing the published benchmark values."""

    def __init__(
        self,
        config: Optional[RewardConfig] = None,
        mode: str = "shaped",
    ):
        """Initialize Reward Engine.

        Args:
            config: RewardConfig dataclass with baseline values.
            mode: "shaped" (published default), "sparse" (win/loss only), or "dense" (shaped + progress bonus).
        """
        self.config = config or RewardConfig()
        self.mode = mode.lower()
        if self.mode not in ("shaped", "sparse", "dense"):
            raise ValueError(f"Unknown reward mode '{mode}'. Expected 'shaped', 'sparse', or 'dense'.")

    def compute_move_reward(
        self,
        move_result: MoveResult,
        is_terminal: bool,
        winner: Optional[int],
    ) -> float:
        """Compute the scalar reward for the acting player."""
        breakdown, _ = self.compute_player_rewards_breakdown(move_result, is_terminal, winner)
        return breakdown.total

    def compute_player_rewards(
        self,
        move_result: MoveResult,
        is_terminal: bool,
        winner: Optional[int],
    ) -> Tuple[float, float]:
        """Compute simultaneous rewards for Player 0 and Player 1.

        Returns:
            Tuple of (p0_reward, p1_reward).
        """
        p0_breakdown, p1_breakdown = self.compute_player_rewards_breakdown(move_result, is_terminal, winner)
        return p0_breakdown.total, p1_breakdown.total

    def compute_player_rewards_breakdown(
        self,
        move_result: MoveResult,
        is_terminal: bool,
        winner: Optional[int],
    ) -> Tuple[RewardBreakdown, RewardBreakdown]:
        """Compute detailed itemized reward breakdowns for both players."""
        acting_p = move_result.player_id
        opponent_p = 1 - acting_p

        acting_bd = RewardBreakdown()
        opponent_bd = RewardBreakdown()

        # Sparse mode: only evaluates terminal win/loss
        if self.mode == "sparse":
            if is_terminal and winner is not None:
                if winner == acting_p:
                    acting_bd.win = self.config.win
                    opponent_bd.loss = self.config.loss
                else:
                    acting_bd.loss = self.config.loss
                    opponent_bd.win = self.config.win
            acting_bd.total = acting_bd.win + acting_bd.loss
            opponent_bd.total = opponent_bd.win + opponent_bd.loss
            p0 = acting_bd if acting_p == 0 else opponent_bd
            p1 = opponent_bd if acting_p == 0 else acting_bd
            return p0, p1

        # --- SHAPED / DENSE MODE ---

        # 1. Token Release (+0.25)
        if move_result.released:
            acting_bd.release = self.config.release

        # 2. Defensive Move (+0.20)
        if move_result.moved_to_safe:
            acting_bd.defend = self.config.defend

        # 3. Capture (+0.15 for capturer) & Token Captured (-0.25 for victim)
        num_captured = len(move_result.captured_tokens)
        if num_captured > 0:
            acting_bd.capture = self.config.capture * num_captured
            opponent_bd.knocked = self.config.knocked * num_captured  # -0.25 * num_captured

        # 4. Progress / Advance (+0.10)
        if not move_result.released and move_result.new_step > move_result.previous_step:
            step_diff = move_result.new_step - move_result.previous_step
            if self.mode == "dense":
                # Dense mode: scales with steps advanced
                acting_bd.advance = self.config.advance + (step_diff * 0.02)
            else:
                acting_bd.advance = self.config.advance

        # 5. Blockade (+0.05)
        if move_result.formed_blockade:
            acting_bd.blockade = self.config.blockade

        # 6. Win (+1.00) and Loss (-1.00)
        if is_terminal and winner is not None:
            if winner == acting_p:
                acting_bd.win = self.config.win
                opponent_bd.loss = self.config.loss
            else:
                acting_bd.loss = self.config.loss
                opponent_bd.win = self.config.win

        # Calculate totals
        acting_bd.total = float(
            acting_bd.release
            + acting_bd.defend
            + acting_bd.capture
            + acting_bd.advance
            + acting_bd.blockade
            + acting_bd.win
            + acting_bd.knocked
            + acting_bd.loss
        )

        opponent_bd.total = float(
            opponent_bd.release
            + opponent_bd.defend
            + opponent_bd.capture
            + opponent_bd.advance
            + opponent_bd.blockade
            + opponent_bd.win
            + opponent_bd.knocked
            + opponent_bd.loss
        )

        p0_bd = acting_bd if acting_p == 0 else opponent_bd
        p1_bd = opponent_bd if acting_p == 0 else acting_bd
        return p0_bd, p1_bd

    def get_illegal_action_penalty(self) -> float:
        """Penalty applied when an illegal action is attempted."""
        return float(self.config.illegal_penalty)
