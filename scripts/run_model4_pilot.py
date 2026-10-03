"""Small pilot training script for Model 4 (Categorical DQN / C51) baseline verification."""

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from phd_ludo_drl.config.seed import set_seed
from phd_ludo_drl.config.config_manager import ConfigManager
from phd_ludo_drl.environment.ludo_env import LudoEnvironment
from phd_ludo_drl.agents.model4.c51_agent import CategoricalDQNAgent
from phd_ludo_drl.agents.model4.trainer import C51Trainer
from phd_ludo_drl.opponents.random_opponent import RandomOpponent


def run_model4_pilot(steps: int = 300, seed: int = 42) -> dict:
    set_seed(seed)
    cfg = ConfigManager.load_config()

    env = LudoEnvironment(seed=seed, auto_advance_no_moves=True)
    agent = CategoricalDQNAgent(config=cfg.model4, latent_dim=128)
    opponent = RandomOpponent(seed=seed + 10)

    trainer = C51Trainer(
        agent=agent,
        opponent=opponent,
        env=env,
        batch_size=32,
        warmup_steps=50,
    )

    print(f"Starting Model 4 (Categorical DQN / C51) Pilot Training ({steps} interaction steps)...")
    metrics = trainer.train(total_steps=steps, eval_interval=150, eval_games=10)
    print("Evaluating trained C51 checkpoint...")
    eval_summary = trainer.evaluate(num_games=20)

    result = {
        "pilot_config": {
            "algorithm": "Categorical DQN (C51)",
            "total_steps": steps,
            "warmup_steps": 50,
            "batch_size": 32,
            "learning_rate": cfg.model4.learning_rate,
            "gamma": cfg.model4.gamma,
            "tau": cfg.model4.tau,
            "num_atoms": cfg.model4.num_atoms,
            "v_min": cfg.model4.v_min,
            "v_max": cfg.model4.v_max,
            "buffer_capacity": cfg.model4.buffer_capacity,
            "seed": seed,
        },
        "training_metrics": metrics.to_dict(),
        "evaluation_metrics": eval_summary.to_dict(),
    }

    out_path = Path("experiments/model4_pilot_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("Pilot Results:", json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    run_model4_pilot(steps=300)
