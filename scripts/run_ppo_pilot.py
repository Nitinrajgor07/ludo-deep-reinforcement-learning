"""Small pilot training script for PPO baseline verification."""

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from phd_ludo_drl.config.seed import set_seed
from phd_ludo_drl.config.config_manager import ConfigManager
from phd_ludo_drl.environment.ludo_env import LudoEnvironment
from phd_ludo_drl.agents.ppo.ppo_agent import PPOAgent
from phd_ludo_drl.agents.ppo.trainer import PPOTrainer
from phd_ludo_drl.opponents.random_opponent import RandomOpponent


def run_ppo_pilot(steps: int = 300, seed: int = 42) -> dict:
    set_seed(seed)
    cfg = ConfigManager.load_config()

    env = LudoEnvironment(seed=seed, auto_advance_no_moves=True)
    agent = PPOAgent(config=cfg.ppo, latent_dim=128)
    opponent = RandomOpponent(seed=seed + 10)

    # Use rollout buffer of 64 for pilot run
    trainer = PPOTrainer(
        agent=agent,
        opponent=opponent,
        env=env,
        rollout_steps=64,
    )

    print(f"Starting PPO Pilot Training ({steps} interaction steps)...")
    metrics = trainer.train(total_steps=steps, eval_interval=150, eval_games=10)
    print("Evaluating trained PPO checkpoint...")
    eval_summary = trainer.evaluate(num_games=20)

    result = {
        "pilot_config": {
            "total_steps": steps,
            "rollout_steps": 64,
            "num_epochs": cfg.ppo.num_epochs,
            "mini_batch_size": cfg.ppo.mini_batch_size,
            "learning_rate": cfg.ppo.learning_rate,
            "gamma": cfg.ppo.gamma,
            "gae_lambda": cfg.ppo.gae_lambda,
            "clip_epsilon": cfg.ppo.clip_epsilon,
            "entropy_coeff": cfg.ppo.entropy_coeff,
            "value_coeff": cfg.ppo.value_coeff,
            "seed": seed,
        },
        "training_metrics": metrics.to_dict(),
        "evaluation_metrics": eval_summary.to_dict(),
    }

    out_path = Path("experiments/ppo_pilot_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("Pilot Results:", json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    run_ppo_pilot(steps=300)
