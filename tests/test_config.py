"""Tests for configuration manager, schema constraints, and immutability."""

from dataclasses import FrozenInstanceError
import pytest
from pathlib import Path
from phd_ludo_drl.config.config_manager import (
    ConfigManager,
    ProjectConfig,
    EnvironmentConfig,
    RewardConfig,
    DQNConfig,
    PPOConfig,
)


def test_default_config_loading():
    """Verify loading from default_config.yaml matches published paper parameters."""
    cfg = ConfigManager.load_config()
    assert isinstance(cfg, ProjectConfig)
    assert cfg.name == "phd_ludo_drl"

    # Environment defaults
    assert cfg.environment.num_players == 2
    assert cfg.environment.tokens_per_player == 4
    assert cfg.environment.track_length == 52

    # Reward values matching published paper
    assert cfg.reward.release == 0.25
    assert cfg.reward.defend == 0.20
    assert cfg.reward.capture == 0.15
    assert cfg.reward.advance == 0.10
    assert cfg.reward.blockade == 0.05
    assert cfg.reward.win == 1.00
    assert cfg.reward.knocked == -0.25
    assert cfg.reward.loss == -1.00

    # DQN defaults
    assert cfg.dqn.tau == 0.005
    assert cfg.dqn.buffer_capacity == 50000

    # PPO defaults
    assert cfg.ppo.num_parallel_envs == 8
    assert cfg.ppo.clip_epsilon == 0.2
    assert cfg.ppo.gae_lambda == 0.95

    # Evaluation games protocol
    assert cfg.evaluation.benchmark_games == 5000


def test_config_immutability():
    """Verify frozen dataclass prevents accidental runtime modification."""
    cfg = ConfigManager.load_config()
    with pytest.raises(FrozenInstanceError):
        cfg.environment.num_players = 4  # type: ignore

    with pytest.raises(FrozenInstanceError):
        cfg.reward.win = 2.0  # type: ignore


def test_config_save_and_reload(tmp_path: Path):
    """Verify round-trip YAML serialization of ProjectConfig."""
    cfg = ConfigManager.load_config()
    target_file = tmp_path / "saved_config.yaml"

    ConfigManager.save_config(cfg, target_file)
    assert target_file.exists()

    reloaded = ConfigManager.load_config(target_file)
    assert reloaded.to_dict() == cfg.to_dict()
