"""Small pilot training script for Model 3 (Discrete SAC) baseline verification."""

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from phd_ludo_drl.config.seed import set_seed
from phd_ludo_drl.config.config_manager import ConfigManager
from phd_ludo_drl.environment.ludo_env import LudoEnvironment
from phd_ludo_drl.agents.model3.sac_agent import DiscreteSACAgent
from phd_ludo_drl.agents.model3.trainer import DiscreteSACTrainer
from phd_ludo_drl.opponents.random_opponent import RandomOpponent


def run_model3_pilot(steps: int = 300, seed: int = 42) -> dict:
    set_seed(seed)
    cfg = ConfigManager.load_config()

    env = LudoEnvironment(seed=seed, auto_advance_no_moves=True)
    agent = DiscreteSACAgent(config=cfg.model3, latent_dim=128)
    opponent = RandomOpponent(seed=seed + 10)

    trainer = DiscreteSACTrainer(
        agent=agent,
        opponent=opponent,
        env=env,
        batch_size=32,
        warmup_steps=50,
    )

    print(f"Starting Model 3 (Discrete SAC) Pilot Training ({steps} interaction steps)...")
    metrics = trainer.train(total_steps=steps, eval_interval=150, eval_games=10)
    print("Evaluating trained Discrete SAC checkpoint...")
    eval_summary = trainer.evaluate(num_games=20)

    result = {
        "pilot_config": {
            "algorithm": "Discrete SAC",
            "total_steps": steps,
            "warmup_steps": 50,
            "batch_size": 32,
            "actor_lr": cfg.model3.actor_lr,
            "critic_lr": cfg.model3.critic_lr,
            "alpha_lr": cfg.model3.alpha_lr,
            "gamma": cfg.model3.gamma,
            "tau": cfg.model3.tau,
            "initial_alpha": cfg.model3.initial_alpha,
            "target_entropy_ratio": cfg.model3.target_entropy_ratio,
            "buffer_capacity": cfg.model3.buffer_capacity,
            "seed": seed,
        },
        "training_metrics": metrics.to_dict(),
        "evaluation_metrics": eval_summary.to_dict(),
    }

    out_path = Path("experiments/model3_pilot_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("Pilot Results:", json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    run_model3_pilot(steps=300)
