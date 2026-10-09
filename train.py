"""Root Command-Line Interface for the Unified PhD Ludo DRL Production Training Pipeline.

Supports:
- Individual model runs across any of the 3 standardized run types:
    --run-type smoke_test
    --run-type pilot_run
    --run-type final_experiment
- 4-Model Smoke Test validation:
    python train.py --smoke-test
- 20-Run Final Training Campaign (4 models x 5 master seeds, 500k steps each):
    python train.py --campaign

Strict Governance:
- A smoke test is NEVER labelled as a final experiment.
- Raw runs are protected against accidental overwriting.
- Checkpoints, logs, manifests, and Git provenance are automatically tracked.
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Optional
import torch

from phd_ludo_drl.config.config_manager import ConfigManager
from phd_ludo_drl.training.factory import SUPPORTED_ALGORITHMS, normalize_algorithm_name
from phd_ludo_drl.training.production_pipeline import (
    ProductionTrainingPipeline,
    RunType,
    MASTER_SEEDS,
    FROZEN_PROTOCOL_VERSION,
    FINAL_EXPERIMENT_STEPS,
)


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct command-line argument parser for production pipeline execution."""
    parser = argparse.ArgumentParser(
        description=f"PhD Ludo DRL: Production Training Pipeline (Protocol v{FROZEN_PROTOCOL_VERSION})",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--algorithm", "-a",
        type=str,
        default="dqn",
        help=f"RL Algorithm to train. Options: {SUPPORTED_ALGORITHMS}",
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help=f"Random seed. Protocol master seeds: {MASTER_SEEDS}",
    )
    parser.add_argument(
        "--run-type", "-r",
        type=str,
        default="smoke_test",
        choices=["smoke_test", "pilot_run", "final_experiment"],
        help="Classification of the run. Note: Smoke tests can NEVER be labelled as final experiments.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Total interaction steps. Defaults to 200 for smoke_test, 25,000 for pilot_run, 500,000 for final_experiment.",
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
        default=None,
        help="Number of games to evaluate in each tournament. Defaults to 2 for smoke_test, 20 for pilot, 100 for final.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=None,
        help="Step frequency for periodic checkpoints. Defaults to 100 for smoke, 5,000 for pilot, 25,000 for final.",
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Attempt failure recovery: resume from latest checkpoint.",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to specific run directory to resume from.",
    )
    parser.add_argument(
        "--campaign",
        action="store_true",
        help="Execute the full 20-run final training campaign (4 models x 5 master seeds, 500k steps each).",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Execute an automated smoke test across all 4 models to validate pipeline integrity.",
    )

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    pipeline = ProductionTrainingPipeline()

    # 1. Quick Automated Smoke Test Across All 4 Models
    if args.smoke_test:
        print("\n" + "=" * 80)
        print("LAUNCHING 4-MODEL PIPELINE SMOKE TEST VALIDATION")
        print("=" * 80)
        results = pipeline.run_smoke_test(
            algorithms=SUPPORTED_ALGORITHMS,
            steps=args.steps or 100,
            seed=args.seed,
            checkpoint_interval=50,
            eval_games=2,
        )
        print("\nAll 4 model smoke tests validated successfully!")
        for algo, res in results.items():
            print(f"  {algo.upper()}: {res['artifacts']['root_dir']}")
        return

    # 2. Large-Scale Final Training Campaign (20 Runs)
    if args.campaign:
        print("\n" + "=" * 80)
        print(f"LAUNCHING 20-RUN FINAL TRAINING CAMPAIGN ({FINAL_EXPERIMENT_STEPS:,} steps/run)")
        print(f"Protocol: {FROZEN_PROTOCOL_VERSION}")
        print("=" * 80)
        status = pipeline.run_campaign(
            algorithms=SUPPORTED_ALGORITHMS,
            seeds=MASTER_SEEDS,
            total_steps=FINAL_EXPERIMENT_STEPS,
            resume=True,
        )
        print(f"Campaign status: {status['status']}")
        return

    # 3. Individual Run Execution
    rtype = RunType(args.run_type)
    if args.steps is not None:
        total_steps = args.steps
    elif rtype == RunType.SMOKE_TEST:
        total_steps = 200
    elif rtype == RunType.PILOT_RUN:
        total_steps = 25_000
    else:
        total_steps = FINAL_EXPERIMENT_STEPS

    eval_games = args.eval_games or (2 if rtype == RunType.SMOKE_TEST else (20 if rtype == RunType.PILOT_RUN else 100))
    ckpt_interval = args.checkpoint_interval or (100 if rtype == RunType.SMOKE_TEST else (5000 if rtype == RunType.PILOT_RUN else 25000))

    config_path = Path(args.config) if args.config else None
    cfg = ConfigManager.load_config(config_path)
    device = torch.device(args.device) if args.device else None
    resume_dir = Path(args.resume_from) if args.resume_from else None

    result = pipeline.run_experiment(
        algorithm=args.algorithm,
        seed=args.seed,
        total_steps=total_steps,
        run_type=rtype,
        opponent_type=args.opponent,
        eval_interval=args.eval_interval,
        eval_games=eval_games,
        checkpoint_interval=ckpt_interval,
        config=cfg,
        description=args.description,
        device=device,
        resume=args.resume or (resume_dir is not None),
        resume_from_dir=resume_dir,
    )

    print("\n" + "=" * 80)
    print(f"EXPERIMENT COMPLETED: {result['metadata']['experiment_id']}")
    print("=" * 80)
    print(f"Run Type:        {result['metadata']['run_type'].upper()}")
    print(f"Algorithm:       {result['metadata']['algorithm'].upper()}")
    print(f"Master Seed:     {result['metadata']['seed']}")
    print(f"Total Steps:     {result['training_metrics']['total_steps']}")
    print(f"Episodes:        {result['training_metrics']['episodes']}")
    print(f"Eval Win Rate:   {result['evaluation_metrics']['p0_win_rate']:.2%}")
    print(f"Artifacts Root:  {result['artifacts']['root_dir']}")
    print(f"Summary JSON:    {result['artifacts']['summary']}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
