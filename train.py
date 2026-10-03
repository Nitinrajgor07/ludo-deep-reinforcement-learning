"""Root Command-Line Interface for the Unified PhD Ludo DRL Training Framework.

Usage examples:
    python train.py --algorithm dqn
    python train.py --algorithm ppo
    python train.py --algorithm model3
    python train.py --algorithm model4
"""

import argparse
import json
from pathlib import Path
import sys
import torch

from phd_ludo_drl.config.config_manager import ConfigManager
from phd_ludo_drl.training.unified_trainer import UnifiedTrainingRunner
from phd_ludo_drl.training.factory import SUPPORTED_ALGORITHMS


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct command-line argument parser for experiment execution."""
    parser = argparse.ArgumentParser(
        description="PhD Ludo DRL: Unified Training Framework for 4-Model Comparative Study",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--algorithm", "-a",
        type=str,
        required=True,
        help=f"RL Algorithm to train. Options: {SUPPORTED_ALGORITHMS}",
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed for environment and model determinism.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=300,
        help="Total interaction steps of the agent.",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=None,
        help="Step interval for intermediate tournament evaluations.",
    )
    parser.add_argument(
        "--eval-games",
        type=int,
        default=10,
        help="Number of games to evaluate in each tournament.",
    )
    parser.add_argument(
        "--opponent",
        type=str,
        default="random",
        choices=["random", "expert"],
        help="Benchmark opponent architecture.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom YAML configuration file.",
    )
    parser.add_argument(
        "--description",
        type=str,
        default="",
        help="Short description of the experiment.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Computation device ('cpu', 'cuda'). Defaults to auto-detection.",
    )

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else None
    cfg = ConfigManager.load_config(config_path)

    device = torch.device(args.device) if args.device else None

    runner = UnifiedTrainingRunner()
    result = runner.run_experiment(
        algorithm=args.algorithm,
        seed=args.seed,
        total_steps=args.steps,
        opponent_type=args.opponent,
        eval_interval=args.eval_interval,
        eval_games=args.eval_games,
        config=cfg,
        description=args.description,
        device=device,
    )

    print("\n" + "=" * 60)
    print(f"EXPERIMENT COMPLETED: {result['metadata']['experiment_id']}")
    print("=" * 60)
    print(f"Algorithm:       {result['metadata']['algorithm'].upper()}")
    print(f"Total Steps:     {result['training_metrics']['total_steps']}")
    print(f"Episodes:        {result['training_metrics']['episodes']}")
    print(f"Agent Wins:      {result['training_metrics'].get('dqn_wins', result['training_metrics'].get('ppo_wins', result['training_metrics'].get('sac_wins', result['training_metrics'].get('c51_wins', 0))))}")
    print(f"Opponent Wins:   {result['training_metrics']['opponent_wins']}")
    print(f"Eval Win Rate:   {result['evaluation_metrics']['p0_win_rate']:.2%}")
    print(f"Artifacts:       {result['artifacts']['results_dir']}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
