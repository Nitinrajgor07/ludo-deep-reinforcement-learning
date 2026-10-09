"""Standalone Runner for Phase 9 Production Training Pipeline.

Usage:
    # 1. Run 4-model validation smoke test (verifies complete pipeline)
    python scripts/run_production_pipeline.py --smoke-test

    # 2. Run single model smoke test
    python scripts/run_production_pipeline.py --algorithm dqn --seed 42 --run-type smoke_test --steps 100

    # 3. Launch 20-run final training campaign (500k steps each, 4 models x 5 master seeds)
    python scripts/run_production_pipeline.py --campaign

    # 4. Resume an interrupted run
    python scripts/run_production_pipeline.py --resume --resume-from experiments/final_experiments/<RUN_DIR>
"""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from phd_ludo_drl.training.production_pipeline import (
    ProductionTrainingPipeline,
    RunType,
    MASTER_SEEDS,
    FINAL_EXPERIMENT_STEPS,
    FROZEN_PROTOCOL_VERSION,
)
from phd_ludo_drl.training.factory import SUPPORTED_ALGORITHMS


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Phase 9 Production Training Pipeline (Protocol v{FROZEN_PROTOCOL_VERSION})",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--algorithm", "-a", type=str, default="dqn", choices=SUPPORTED_ALGORITHMS)
    parser.add_argument("--seed", "-s", type=int, default=42, choices=MASTER_SEEDS)
    parser.add_argument("--run-type", "-r", type=str, default="smoke_test", choices=["smoke_test", "pilot_run", "final_experiment"])
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true", help="Run 4-model validation smoke test")
    parser.add_argument("--campaign", action="store_true", help="Run full 20-run final benchmark campaign")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--resume-from", type=str, default=None, help="Path to run directory to resume")

    args = parser.parse_args()
    pipeline = ProductionTrainingPipeline(base_dir=PROJECT_ROOT)

    if args.smoke_test:
        print("Executing automated 4-model smoke test...")
        results = pipeline.run_smoke_test(
            algorithms=SUPPORTED_ALGORITHMS,
            steps=args.steps or 100,
            seed=args.seed,
            checkpoint_interval=50,
            eval_games=2,
        )
        print(f"Smoke test successfully verified {len(results)} algorithms!")
        return

    if args.campaign:
        print(f"Launching 20-run final campaign ({FINAL_EXPERIMENT_STEPS:,} steps per run)...")
        status = pipeline.run_campaign(
            algorithms=SUPPORTED_ALGORITHMS,
            seeds=MASTER_SEEDS,
            total_steps=FINAL_EXPERIMENT_STEPS,
            resume=True,
        )
        print(f"Campaign finished with status: {status['status']}")
        return

    rtype = RunType(args.run_type)
    steps = args.steps or (200 if rtype == RunType.SMOKE_TEST else (25_000 if rtype == RunType.PILOT_RUN else FINAL_EXPERIMENT_STEPS))
    ckpt_interval = 100 if rtype == RunType.SMOKE_TEST else (5000 if rtype == RunType.PILOT_RUN else 25000)
    eval_games = 2 if rtype == RunType.SMOKE_TEST else (20 if rtype == RunType.PILOT_RUN else 100)
    resume_dir = Path(args.resume_from) if args.resume_from else None

    result = pipeline.run_experiment(
        algorithm=args.algorithm,
        seed=args.seed,
        total_steps=steps,
        run_type=rtype,
        checkpoint_interval=ckpt_interval,
        eval_games=eval_games,
        resume=args.resume or (resume_dir is not None),
        resume_from_dir=resume_dir,
    )
    print(f"Run finished: {result['metadata']['experiment_id']}")
    print(f"Summary: {result['artifacts']['summary']}")


if __name__ == "__main__":
    main()
