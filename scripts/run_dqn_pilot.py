"""Small pilot training script for Double Dueling DQN baseline verification."""

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from phd_ludo_drl.config.seed import set_seed
from phd_ludo_drl.config.config_manager import ConfigManager
from phd_ludo_drl.environment.ludo_env import LudoEnvironment
from phd_ludo_drl.agents.dqn.dqn_agent import DQNAgent
from phd_ludo_drl.agents.dqn.trainer import DQNTrainer
from phd_ludo_drl.opponents.random_opponent import RandomOpponent


def run_dqn_pilot(steps: int = 300, seed: int = 42) -> dict:
    set_seed(seed)
    cfg = ConfigManager.load_config()

    env = LudoEnvironment(seed=seed, auto_advance_no_moves=True)
    agent = DQNAgent(config=cfg.dqn, latent_dim=128)
    opponent = RandomOpponent(seed=seed + 10)

    trainer = DQNTrainer(
        agent=agent,
        opponent=opponent,
        env=env,
        batch_size=32,
        warmup_steps=50,
    )

    print(f"Starting DQN Pilot Training ({steps} interaction steps)...")
    metrics = trainer.train(total_steps=steps, eval_interval=150, eval_games=10)
    print("Evaluating trained checkpoint...")
    eval_summary = trainer.evaluate(num_games=20)

    result = {
        "pilot_config": {
            "total_steps": steps,
            "warmup_steps": 50,
            "batch_size": 32,
            "learning_rate": cfg.dqn.learning_rate,
            "gamma": cfg.dqn.gamma,
            "tau": cfg.dqn.tau,
            "per_alpha": cfg.dqn.per_alpha,
            "buffer_capacity": cfg.dqn.buffer_capacity,
            "seed": seed,
        },
        "training_metrics": metrics.to_dict(),
        "evaluation_metrics": eval_summary.to_dict(),
    }

    out_path = Path("experiments/dqn_pilot_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("Pilot Results:", json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    run_dqn_pilot(steps=300)
