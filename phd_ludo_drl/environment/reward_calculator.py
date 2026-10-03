"""Backward-compatible reward calculator wrapper delegating to RewardEngine."""

from typing import Optional
from phd_ludo_drl.config.config_manager import RewardConfig
from phd_ludo_drl.environment.rules import MoveResult
from phd_ludo_drl.environment.reward_engine import RewardEngine, RewardBreakdown


class LudoRewardCalculator(RewardEngine):
    """Backward-compatible alias for RewardEngine."""

    def __init__(self, config: Optional[RewardConfig] = None, mode: str = "shaped"):
        super().__init__(config=config, mode=mode)
