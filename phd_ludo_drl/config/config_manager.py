"""Centralized configuration manager with schema validation and immutability controls."""

from dataclasses import dataclass, field, asdict
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


@dataclass(frozen=True)
class EnvironmentConfig:
    num_players: int = 2
    tokens_per_player: int = 4
    track_length: int = 52
    home_column_length: int = 5
    safe_squares: List[int] = field(default_factory=lambda: [0, 8, 13, 21, 26, 34, 39, 47])
    max_turns_per_game: int = 1000


@dataclass(frozen=True)
class EncodingConfig:
    type: str = "structured_cnn"
    latent_dim: int = 128
    cnn_channels: int = 4
    cnn_filters: List[int] = field(default_factory=lambda: [32, 64])


@dataclass(frozen=True)
class RewardConfig:
    release: float = 0.25
    defend: float = 0.20
    capture: float = 0.15
    advance: float = 0.10
    blockade: float = 0.05
    win: float = 1.00
    knocked: float = -0.25
    loss: float = -1.00
    illegal_penalty: float = -0.50


@dataclass(frozen=True)
class DQNConfig:
    architecture: str = "double_dueling"
    hidden_units: int = 256
    learning_rate: float = 1e-4
    gamma: float = 0.99
    tau: float = 0.005
    buffer_capacity: int = 50000
    batch_size: int = 64
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 100000


@dataclass(frozen=True)
class PPOConfig:
    hidden_units: int = 128
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coeff: float = 0.01
    value_coeff: float = 0.5
    num_parallel_envs: int = 8
    rollout_steps_per_env: int = 128
    num_epochs: int = 4
    mini_batch_size: int = 256
    max_grad_norm: float = 0.5


@dataclass(frozen=True)
class DiscreteSACConfig:
    name: str = "discrete_sac"
    hidden_units: int = 128
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    buffer_capacity: int = 50000
    batch_size: int = 64
    initial_alpha: float = 0.2
    tune_alpha: bool = True
    target_entropy_ratio: float = 0.98
    warmup_steps: int = 200


@dataclass(frozen=True)
class CategoricalDQNConfig:
    name: str = "c51_dqn"
    hidden_units: int = 128
    learning_rate: float = 1e-4
    gamma: float = 0.99
    tau: float = 0.005
    num_atoms: int = 51
    v_min: float = -20.0
    v_max: float = 30.0
    buffer_capacity: int = 50000
    batch_size: int = 64
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 100000
    warmup_steps: int = 200


@dataclass(frozen=True)
class TrainingConfig:
    total_steps: int = 500000
    checkpoint_interval: int = 25000
    eval_interval: int = 10000
    eval_games_during_train: int = 100
    seeds: List[int] = field(default_factory=lambda: [42, 101, 202, 303, 404])


@dataclass(frozen=True)
class EvaluationConfig:
    benchmark_games: int = 5000
    opponents: List[str] = field(default_factory=lambda: ["random", "expert"])


@dataclass(frozen=True)
class ProjectConfig:
    name: str = "phd_ludo_drl"
    version: str = "0.1.0"
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    encoding: EncodingConfig = field(default_factory=EncodingConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    dqn: DQNConfig = field(default_factory=DQNConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    model3: DiscreteSACConfig = field(default_factory=DiscreteSACConfig)
    model4: CategoricalDQNConfig = field(default_factory=CategoricalDQNConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ConfigManager:
    """Manages loading, validation, and serialization of research experiment configs."""

    @staticmethod
    def get_default_config_path() -> Path:
        return Path(__file__).parent / "default_config.yaml"

    @classmethod
    def load_config(cls, path: Optional[Path] = None) -> ProjectConfig:
        """Load configuration from a YAML file or fall back to defaults."""
        config_path = path or cls.get_default_config_path()
        if not os.path.exists(config_path):
            return ProjectConfig()

        with open(config_path, "r", encoding="utf-8") as f:
            raw_dict = yaml.safe_load(f) or {}

        env_cfg = EnvironmentConfig(**raw_dict.get("environment", {}))
        enc_cfg = EncodingConfig(**raw_dict.get("encoding", {}))
        rew_cfg = RewardConfig(**raw_dict.get("reward", {}))
        dqn_cfg = DQNConfig(**raw_dict.get("dqn", {}))
        ppo_cfg = PPOConfig(**raw_dict.get("ppo", {}))
        m3_cfg = DiscreteSACConfig(**raw_dict.get("model3", {}))
        m4_cfg = CategoricalDQNConfig(**raw_dict.get("model4", {}))
        trn_cfg = TrainingConfig(**raw_dict.get("training", {}))
        evl_cfg = EvaluationConfig(**raw_dict.get("evaluation", {}))

        proj_meta = raw_dict.get("project", {})
        return ProjectConfig(
            name=proj_meta.get("name", "phd_ludo_drl"),
            version=proj_meta.get("version", "0.1.0"),
            environment=env_cfg,
            encoding=enc_cfg,
            reward=rew_cfg,
            dqn=dqn_cfg,
            ppo=ppo_cfg,
            model3=m3_cfg,
            model4=m4_cfg,
            training=trn_cfg,
            evaluation=evl_cfg,
        )



    @staticmethod
    def save_config(config: ProjectConfig, target_path: Path) -> None:
        """Save a frozen configuration snapshot to YAML for experiment logging."""
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)
