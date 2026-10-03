"""Factory for initializing agents, opponents, environments, and trainers."""

from typing import Any, Dict, Optional, Tuple, Type
import torch

from phd_ludo_drl.config.config_manager import ProjectConfig, ConfigManager
from phd_ludo_drl.environment.ludo_env import LudoEnvironment
from phd_ludo_drl.agents.base_agent import BaseAgent
from phd_ludo_drl.opponents.base_opponent import BaseOpponent
from phd_ludo_drl.opponents.random_opponent import RandomOpponent
from phd_ludo_drl.opponents.expert_opponent import ExpertOpponent

# Model 1: Double Dueling DQN
from phd_ludo_drl.agents.dqn.dqn_agent import DQNAgent
from phd_ludo_drl.agents.dqn.trainer import DQNTrainer

# Model 2: PPO
from phd_ludo_drl.agents.ppo.ppo_agent import PPOAgent
from phd_ludo_drl.agents.ppo.trainer import PPOTrainer

# Model 3: Discrete Soft Actor-Critic
from phd_ludo_drl.agents.model3.sac_agent import DiscreteSACAgent
from phd_ludo_drl.agents.model3.trainer import DiscreteSACTrainer

# Model 4: Categorical DQN (C51)
from phd_ludo_drl.agents.model4.c51_agent import CategoricalDQNAgent
from phd_ludo_drl.agents.model4.trainer import C51Trainer


SUPPORTED_ALGORITHMS = ["dqn", "ppo", "model3", "model4"]

ALGORITHM_ALIASES = {
    "dqn": "dqn",
    "double_dueling_dqn": "dqn",
    "ppo": "ppo",
    "model3": "model3",
    "sac": "model3",
    "discrete_sac": "model3",
    "model4": "model4",
    "c51": "model4",
    "c51_dqn": "model4",
    "categorical_dqn": "model4",
}


def normalize_algorithm_name(name: str) -> str:
    """Normalize algorithm identifier to one of the canonical 4 models."""
    cleaned = name.strip().lower()
    if cleaned not in ALGORITHM_ALIASES:
        valid_options = ", ".join(SUPPORTED_ALGORITHMS)
        raise ValueError(f"Unknown algorithm '{name}'. Supported algorithms: {valid_options}")
    return ALGORITHM_ALIASES[cleaned]


def create_agent(
    algorithm: str,
    config: Optional[ProjectConfig] = None,
    device: Optional[torch.device] = None,
) -> BaseAgent:
    """Instantiate one of the 4 benchmark RL agents with shared encoder parity.

    Args:
        algorithm: One of ['dqn', 'ppo', 'model3', 'model4'].
        config: Research configuration.
        device: Torch computation device.

    Returns:
        Instance of BaseAgent subclass.
    """
    cfg = config or ConfigManager.load_config()
    algo = normalize_algorithm_name(algorithm)

    if algo == "dqn":
        return DQNAgent(config=cfg.dqn, latent_dim=cfg.encoding.latent_dim, device=device)
    elif algo == "ppo":
        return PPOAgent(config=cfg.ppo, latent_dim=cfg.encoding.latent_dim, device=device)
    elif algo == "model3":
        return DiscreteSACAgent(config=cfg.model3, latent_dim=cfg.encoding.latent_dim, device=device)
    elif algo == "model4":
        return CategoricalDQNAgent(config=cfg.model4, latent_dim=cfg.encoding.latent_dim, device=device)
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")


def create_opponent(opponent_type: str, seed: Optional[int] = None) -> BaseOpponent:
    """Instantiate a standardized benchmark opponent.

    Args:
        opponent_type: 'random' or 'expert'.
        seed: Random seed for opponent stochasticity.

    Returns:
        Instance of BaseOpponent subclass.
    """
    opp = opponent_type.strip().lower()
    if opp == "random":
        return RandomOpponent(seed=seed)
    elif opp == "expert":
        return ExpertOpponent(seed=seed)
    else:
        raise ValueError(f"Unknown opponent type '{opponent_type}'. Must be 'random' or 'expert'.")


def create_trainer(
    algorithm: str,
    agent: BaseAgent,
    opponent: BaseOpponent,
    env: LudoEnvironment,
    config: Optional[ProjectConfig] = None,
    batch_size: Optional[int] = None,
    warmup_steps: Optional[int] = None,
    rollout_steps: Optional[int] = None,
) -> Any:
    """Instantiate the appropriate algorithm trainer with uniform parameters.

    Args:
        algorithm: Algorithm identifier.
        agent: Initialized BaseAgent.
        opponent: Initialized BaseOpponent.
        env: Initialized LudoEnvironment.
        config: Research configuration.
        batch_size: Optional batch size override.
        warmup_steps: Optional warmup steps override.
        rollout_steps: Optional rollout steps override (PPO).

    Returns:
        Algorithm-specific trainer instance with standard train/evaluate interface.
    """
    cfg = config or ConfigManager.load_config()
    algo = normalize_algorithm_name(algorithm)

    if algo == "dqn":
        bs = batch_size or cfg.dqn.batch_size
        return DQNTrainer(
            agent=agent,  # type: ignore
            opponent=opponent,
            env=env,
            batch_size=bs,
            warmup_steps=warmup_steps if warmup_steps is not None else 200,
        )
    elif algo == "ppo":
        rs = rollout_steps or cfg.ppo.rollout_steps_per_env
        return PPOTrainer(
            agent=agent,  # type: ignore
            opponent=opponent,
            env=env,
            rollout_steps=rs,
        )
    elif algo == "model3":
        bs = batch_size or cfg.model3.batch_size
        return DiscreteSACTrainer(
            agent=agent,  # type: ignore
            opponent=opponent,
            env=env,
            batch_size=bs,
            warmup_steps=warmup_steps if warmup_steps is not None else cfg.model3.warmup_steps,
        )
    elif algo == "model4":
        bs = batch_size or cfg.model4.batch_size
        return C51Trainer(
            agent=agent,  # type: ignore
            opponent=opponent,
            env=env,
            batch_size=bs,
            warmup_steps=warmup_steps if warmup_steps is not None else cfg.model4.warmup_steps,
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")
