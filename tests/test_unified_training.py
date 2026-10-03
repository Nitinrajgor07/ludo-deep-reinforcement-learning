"""Unit tests for the Unified Training Framework."""

from pathlib import Path
import pytest
import torch

from phd_ludo_drl.config.config_manager import ConfigManager
from phd_ludo_drl.environment.ludo_env import LudoEnvironment
from phd_ludo_drl.agents.dqn.dqn_agent import DQNAgent
from phd_ludo_drl.agents.ppo.ppo_agent import PPOAgent
from phd_ludo_drl.agents.model3.sac_agent import DiscreteSACAgent
from phd_ludo_drl.agents.model4.c51_agent import CategoricalDQNAgent
from phd_ludo_drl.opponents.random_opponent import RandomOpponent
from phd_ludo_drl.opponents.expert_opponent import ExpertOpponent
from phd_ludo_drl.training.factory import (
    create_agent,
    create_opponent,
    create_trainer,
    normalize_algorithm_name,
    SUPPORTED_ALGORITHMS,
)
from phd_ludo_drl.training.unified_trainer import UnifiedTrainingRunner
from train import build_arg_parser


def test_factory_create_all_four_agents():
    """Verify factory initializes all 4 models with identical latent dimensions."""
    cfg = ConfigManager.load_config()

    agent_dqn = create_agent("dqn", cfg)
    assert isinstance(agent_dqn, DQNAgent)
    assert agent_dqn.latent_dim == 128

    agent_ppo = create_agent("ppo", cfg)
    assert isinstance(agent_ppo, PPOAgent)
    assert agent_ppo.latent_dim == 128

    agent_m3 = create_agent("model3", cfg)
    assert isinstance(agent_m3, DiscreteSACAgent)
    assert agent_m3.latent_dim == 128

    agent_m4 = create_agent("model4", cfg)
    assert isinstance(agent_m4, CategoricalDQNAgent)
    assert agent_m4.latent_dim == 128


def test_factory_create_opponents():
    """Verify opponent factory creates valid benchmark opponents."""
    opp_rand = create_opponent("random", seed=42)
    assert isinstance(opp_rand, RandomOpponent)

    opp_exp = create_opponent("expert", seed=42)
    assert isinstance(opp_exp, ExpertOpponent)

    with pytest.raises(ValueError):
        create_opponent("unknown_opponent")


def test_factory_create_trainers():
    """Verify trainer factory binds agents, opponents, and environments."""
    env = LudoEnvironment(auto_advance_no_moves=True)
    opp = RandomOpponent()
    cfg = ConfigManager.load_config()

    for algo in SUPPORTED_ALGORITHMS:
        agent = create_agent(algo, cfg)
        trainer = create_trainer(algo, agent, opp, env, cfg)
        assert hasattr(trainer, "train")
        assert hasattr(trainer, "evaluate")


def test_unified_training_runner_execution(tmp_path):
    """Verify UnifiedTrainingRunner executes and persists artifacts for all 4 models."""
    runner = UnifiedTrainingRunner(base_dir=tmp_path)

    for algo in SUPPORTED_ALGORITHMS:
        result = runner.run_experiment(
            algorithm=algo,
            seed=42,
            total_steps=15,
            opponent_type="random",
            eval_games=2,
            batch_size=4,
            warmup_steps=5,
            rollout_steps=8,
        )

        assert result["metadata"]["algorithm"] == algo
        assert result["training_metrics"]["total_steps"] == 15
        assert "p0_win_rate" in result["evaluation_metrics"]

        # Check saved artifacts
        res_dir = tmp_path / result["artifacts"]["results_dir"]
        assert (res_dir / "summary.json").exists()


def test_cli_argument_parser():
    """Verify command line argument parser configurations."""
    parser = build_arg_parser()

    args = parser.parse_args(["--algorithm", "dqn", "--seed", "101", "--steps", "500"])
    assert args.algorithm == "dqn"
    assert args.seed == 101
    assert args.steps == 500

    args_ppo = parser.parse_args(["--algorithm", "ppo", "--opponent", "expert"])
    assert args_ppo.algorithm == "ppo"
    assert args_ppo.opponent == "expert"
