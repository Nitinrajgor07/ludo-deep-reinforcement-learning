"""Unified training runner for executing standardized experiments across all 4 RL models."""

import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, Optional
import torch

from phd_ludo_drl.config.seed import set_seed
from phd_ludo_drl.config.config_manager import ConfigManager, ProjectConfig
from phd_ludo_drl.environment.ludo_env import LudoEnvironment
from phd_ludo_drl.experiments.experiment_id import ExperimentIdentifier, ExperimentMetadata
from phd_ludo_drl.logs.logger import setup_logger
from phd_ludo_drl.training.factory import (
    create_agent,
    create_opponent,
    create_trainer,
    normalize_algorithm_name,
    SUPPORTED_ALGORITHMS,
)


class UnifiedTrainingRunner:
    """Orchestrates end-to-end training runs for DQN, PPO, Model 3, and Model 4.

    Ensures 100% empirical control:
    - Same environment and seed registry
    - Same structured CNN encoder backbone
    - Same action masking and reward engine
    - Same opponent suite and self-play arena
    - Automated logging, checkpointing, and artifact serialization
    """

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or Path.cwd()

    def run_experiment(
        self,
        algorithm: str,
        seed: int = 42,
        total_steps: int = 1000,
        opponent_type: str = "random",
        eval_interval: Optional[int] = None,
        eval_games: int = 10,
        config: Optional[ProjectConfig] = None,
        description: str = "",
        device: Optional[torch.device] = None,
        batch_size: Optional[int] = None,
        warmup_steps: Optional[int] = None,
        rollout_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute a standardized experiment from start to finish.

        Args:
            algorithm: One of ['dqn', 'ppo', 'model3', 'model4'].
            seed: Master random seed.
            total_steps: Total interaction steps to train.
            opponent_type: 'random' or 'expert'.
            eval_interval: Interaction interval between periodic evaluations.
            eval_games: Number of evaluation games per tournament.
            config: Optional ProjectConfig override.
            description: Description of the experiment.
            device: Torch computation device.
            batch_size: Batch size override.
            warmup_steps: Warmup steps override.
            rollout_steps: Rollout steps override (for PPO).

        Returns:
            Dictionary containing metadata, training metrics, and evaluation summaries.
        """
        algo = normalize_algorithm_name(algorithm)
        cfg = config or ConfigManager.load_config()

        # 1. Deterministic Seeding
        set_seed(seed)

        # 2. Experiment Metadata & Directory Structure
        metadata: ExperimentMetadata = ExperimentIdentifier.create(
            algorithm=algo,
            seed=seed,
            encoding=cfg.encoding.type,
            description=description or f"Unified benchmark run for {algo.upper()}",
            config_dict=cfg.to_dict(),
        )

        paths = ExperimentIdentifier.resolve_paths(self.base_dir, metadata.experiment_id)

        # 3. Structured Logging
        logger = setup_logger(
            name=metadata.experiment_id,
            log_dir=paths["logs"],
            log_to_console=True,
        )
        logger.info(f"Initialized Experiment: {metadata.experiment_id}")
        logger.info(f"Algorithm: {algo.upper()} | Seed: {seed} | Steps: {total_steps} | Opponent: {opponent_type}")

        # 4. Save Frozen Configuration Snapshot
        ConfigManager.save_config(cfg, paths["root"] / "config_frozen.yaml")

        # 5. Environment & Entities Initialization
        env = LudoEnvironment(seed=seed, auto_advance_no_moves=True)
        agent = create_agent(algorithm=algo, config=cfg, device=device)
        opponent = create_opponent(opponent_type=opponent_type, seed=seed + 10)

        # 6. Unified Trainer Initialization
        trainer = create_trainer(
            algorithm=algo,
            agent=agent,
            opponent=opponent,
            env=env,
            config=cfg,
            batch_size=batch_size,
            warmup_steps=warmup_steps,
            rollout_steps=rollout_steps,
        )

        logger.info(f"Starting training loop ({total_steps} steps)...")
        start_time = time.time()

        # 7. Execute Training Loop
        training_metrics = trainer.train(
            total_steps=total_steps,
            eval_interval=eval_interval,
            eval_games=eval_games,
            checkpoint_dir=paths["checkpoints"],
            callback=lambda step_info: logger.debug(f"Step update: {step_info}"),
        )

        logger.info(f"Training completed: {training_metrics.total_steps} steps, {training_metrics.episodes} episodes.")

        # 8. Final Checkpoint
        final_ckpt_path = paths["checkpoints"] / "final_model.pt"
        agent.save_checkpoint(final_ckpt_path)
        logger.info(f"Saved final checkpoint to {final_ckpt_path}")

        # 9. Post-Training Tournament Evaluation
        logger.info(f"Executing post-training evaluation ({eval_games} games vs {opponent_type})...")
        eval_summary = trainer.evaluate(num_games=eval_games)
        logger.info(
            f"Eval Outcome: Win Rate = {eval_summary.p0_win_rate:.2%} | "
            f"Avg Turns = {eval_summary.avg_turns:.1f} | Avg Reward = {eval_summary.avg_p0_reward:.2f}"
        )

        runtime_seconds = round(time.time() - start_time, 2)
        logger.info(f"Training and evaluation completed in {runtime_seconds:.2f}s")

        # 10. Compile and Save Results Artifact
        result_payload = {
            "metadata": metadata.to_dict(),
            "config": {
                "algorithm": algo,
                "seed": seed,
                "total_steps": total_steps,
                "opponent_type": opponent_type,
                "eval_games": eval_games,
            },
            "runtime_seconds": runtime_seconds,
            "training_metrics": training_metrics.to_dict(),
            "evaluation_metrics": eval_summary.to_dict(),
            "artifacts": {
                "checkpoint": str(final_ckpt_path.relative_to(self.base_dir)),
                "log_dir": str(paths["logs"].relative_to(self.base_dir)),
                "results_dir": str(paths["results"].relative_to(self.base_dir)),
            },
        }

        results_json_path = paths["results"] / "summary.json"
        with open(results_json_path, "w", encoding="utf-8") as f:
            json.dump(result_payload, f, indent=2)

        logger.info(f"Saved complete experiment summary to {results_json_path}")
        return result_payload
