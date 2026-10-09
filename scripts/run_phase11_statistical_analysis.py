"""PHASE 11 — STATISTICAL ANALYSIS RUNNER

Command-line orchestrator for Phase 11 Statistical Analysis.
Strictly adheres to:
- FINAL_EXPERIMENT_PROTOCOL.md Section 20
- Phase 11 Requirements:
  * Uses ONLY raw evaluation data
  * Distinguishes PRELIMINARY/PILOT from FINAL EXPERIMENT
  * If final raw data is missing, reports analysis as INCOMPLETE rather than fabricating values
  * Zero manually typed numbers; 100% reproducible

Usage:
    # 1. Audit raw evaluation data readiness:
    python scripts/run_phase11_statistical_analysis.py --audit

    # 2. Run final statistical analysis (strictly requires final 200k raw data):
    python scripts/run_phase11_statistical_analysis.py --mode final

    # 3. Run analysis on preliminary / pilot data (clearly labelled as pilot):
    python scripts/run_phase11_statistical_analysis.py --mode preliminary
"""

import argparse
from datetime import datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from phd_ludo_drl.analysis.statistical_pipeline import (
    StatisticalAnalysisPipeline,
    DataTier,
    IncompleteRawDataError,
    TOTAL_REQUIRED_FINAL_GAMES,
    FROZEN_PROTOCOL_VERSION,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Phase 11 Statistical Analysis Pipeline (Protocol v{FROZEN_PROTOCOL_VERSION})",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="final",
        choices=["final", "preliminary", "audit"],
        help="Analysis mode: 'final' (strictly enforces 200k 500k-step raw data), 'preliminary' (analyzes pilot data with explicit disclaimer), 'audit' (checks data readiness only).",
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        default=False,
        help="Audit raw data completeness and exit.",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=None,
        help="Custom raw evaluation data directory to scan.",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="results/processed",
        help="Directory to save structured statistical outputs.",
    )
    parser.add_argument(
        "--tables-dir",
        type=str,
        default="results/tables",
        help="Directory to save publication tables.",
    )
    parser.add_argument(
        "--master-seed",
        type=int,
        default=42,
        help="Master seed for bootstrap resampling.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    raw_dirs = [Path(args.raw_dir)] if args.raw_dir else None
    require_final = (args.mode == "final") and not args.audit

    pipeline = StatisticalAnalysisPipeline(
        raw_dirs=raw_dirs,
        processed_dir=args.processed_dir,
        tables_dir=args.tables_dir,
        master_seed=args.master_seed,
        require_final_data=require_final,
    )

    if args.audit or args.mode == "audit":
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Auditing Raw Evaluation Data for Statistical Pipeline...")
        audit = pipeline.audit_raw_data()
        print("=" * 88)
        print("PHASE 11 RAW EVALUATION DATA AUDIT REPORT")
        print(f"Protocol Version: {audit['protocol_version']}")
        print(f"Audit Timestamp: {audit['audit_timestamp_utc']}")
        print(f"Total Games Discovered: {audit['total_games_ingested']:,}")
        print(f"Final Experiment Games: {audit['final_experiment_games']:,} / {TOTAL_REQUIRED_FINAL_GAMES:,}")
        print(f"Preliminary Pilot Games: {audit['preliminary_pilot_games']:,}")
        print(f"Data Tier: {audit['data_tier']}")
        print(f"Algorithms Present: {audit['models_present']}")
        print(f"Seeds Present: {audit['seeds_present']}")
        print(f"Opponents Present: {audit['opponents_present']}")
        print("-" * 88)
        if not audit["is_final_complete"]:
            print(
                f"STATUS: INCOMPLETE (Final raw evaluation data is missing or partial).\n"
                f"Required: {TOTAL_REQUIRED_FINAL_GAMES:,} games across 4 models x 5 master seeds from 500k checkpoints.\n"
                f"Protocol Directive: Final statistical analysis is marked INCOMPLETE.\n"
                f"Under NO circumstances will substitute or fabricated values be generated."
            )
            return 1
        else:
            print("STATUS: COMPLETE (All 200,000 final games verified). Ready for final statistical analysis.")
            return 0

    # Run analysis
    result = pipeline.run()

    if result.get("status") == "INCOMPLETE":
        print(
            "\n[STATISTICAL ANALYSIS STATUS: INCOMPLETE]\n"
            "Final raw evaluation data is missing. The analysis has been reported as INCOMPLETE.\n"
            "Substitute values were NOT generated in accordance with protocol safety directives."
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
