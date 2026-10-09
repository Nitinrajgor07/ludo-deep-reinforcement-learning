"""PHASE 10 — FINAL EVALUATION RUNNER

Command-line orchestrator for Phase 10 Final Evaluation.
Strictly adheres to:
- FINAL_EXPERIMENT_PROTOCOL.md (Version 1.0.0 FROZEN)
- PhD Research Benchmark Specification

Protocol Requirements:
- 4 models: Double Dueling DQN, PPO, Discrete SAC (Model 3), C51 (Model 4)
- 5 seeds: [42, 101, 202, 303, 404]
- 5,000 games vs Random Opponent
- 5,000 games vs Expert Opponent (Alhajry 2012 priority strategy)
- Total: 200,000 evaluation games across 20 trained models
- Balanced seating: 50% Player 0, 50% Player 1
- Fixed evaluation policy: deterministic action selection
- Independent game seeds: unique per game
- Raw game-level persistence (CSV and JSONL)
- Never overwrite raw evaluation data
- Checkpoint verification: STOP and report if any 500k checkpoint is missing.
  Do NOT substitute a pilot checkpoint and label it final.

Usage:
    # 1. Audit checkpoint readiness (inspects all 20 required 500k checkpoints):
    python scripts/run_phase10_final_evaluation.py --check-readiness

    # 2. Execute final evaluation (strictly enforces 500k checkpoint verification):
    python scripts/run_phase10_final_evaluation.py --run

    # 3. Dry-run pre-flight validation:
    python scripts/run_phase10_final_evaluation.py --dry-run
"""

import argparse
from datetime import datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from phd_ludo_drl.evaluation.final_evaluation_pipeline import (
    FinalEvaluationPipeline,
    FinalEvaluationConfig,
    CheckpointValidator,
    MissingCheckpointError,
    RawDataProtectedError,
    BENCHMARK_MODELS,
    MASTER_SEEDS,
    EVALUATION_OPPONENTS,
    GAMES_PER_OPPONENT,
    REQUIRED_FINAL_STEPS,
    TOTAL_PLANNED_GAMES,
    FROZEN_PROTOCOL_VERSION,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Phase 10 Final Evaluation Pipeline (Protocol v{FROZEN_PROTOCOL_VERSION})",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--check-readiness",
        action="store_true",
        default=False,
        help="Audit repository for required 500k checkpoints and print diagnostic report without running games.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        default=False,
        help="Execute the final evaluation pipeline (strictly aborts if any 500k checkpoint is missing).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Perform pre-flight verification without executing game simulations.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=BENCHMARK_MODELS,
        choices=BENCHMARK_MODELS,
        help="List of model algorithms to evaluate.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=MASTER_SEEDS,
        choices=MASTER_SEEDS,
        help="List of master random seeds to evaluate.",
    )
    parser.add_argument(
        "--games-per-opponent",
        type=int,
        default=GAMES_PER_OPPONENT,
        help="Number of evaluation games per opponent (5,000 required for final evaluation).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/final_evaluation",
        help="Directory to store raw and processed evaluation artifacts.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    cfg = FinalEvaluationConfig(
        models=args.models,
        seeds=args.seeds,
        opponents=EVALUATION_OPPONENTS,
        games_per_opponent=args.games_per_opponent,
        required_min_steps=REQUIRED_FINAL_STEPS,
        output_root=Path(args.output_dir),
    )

    pipeline = FinalEvaluationPipeline(base_dir=PROJECT_ROOT, config=cfg)

    # If run without flags or with --check-readiness, perform the audit
    if args.check_readiness or (not args.run and not args.dry_run):
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Auditing Phase 10 Checkpoint Readiness...")
        report = pipeline.get_readiness_report()
        print(report)
        readiness = pipeline.check_readiness()
        if not readiness["all_ready"]:
            print(
                f"\nAudit Result: INCOMPLETE ({readiness['ready_count']}/{readiness['total_required']} ready).\n"
                f"Missing / Insufficient: {readiness['missing_count']} missing, {readiness['pilot_count']} pilot/intermediate.\n"
                f"STOPPING per research protocol: Required 500k checkpoints are missing. Pilot checkpoints cannot be substituted."
            )
            return 1
        else:
            print("\nAudit Result: COMPLETE (20/20 ready). Ready for 200,000 evaluation games.")
            return 0

    if args.dry_run:
        try:
            pipeline.run_pipeline(dry_run=True)
            return 0
        except MissingCheckpointError as e:
            print(f"\n[STOPPED - MISSING CHECKPOINT]\n{e}")
            return 1

    if args.run:
        try:
            pipeline.run_pipeline(dry_run=False)
            return 0
        except MissingCheckpointError as e:
            print(f"\n[STOPPED - MISSING CHECKPOINT]\n{e}")
            return 1
        except RawDataProtectedError as e:
            print(f"\n[STOPPED - DATA PROTECTION]\n{e}")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
