"""PHASE 12 — ROBUSTNESS AND ABLATION RUNNER

Command-line orchestrator for Phase 12 Robustness and Ablation Experiments.
Strictly adheres to:
- FINAL_EXPERIMENT_PROTOCOL.md Section 18 & Section 19
- Phase 12 Requirements:
  * Explicit hypothesis for every ablation
  * Controlled single-variable change holding other conditions constant
  * Documented master seed
  * Raw data persistence
  * Reproducible statistical analysis
  * Zero selective reporting: Honest disclosure of negative and unfavorable findings

Usage:
    # 1. List all formal hypotheses and experiment specifications:
    python scripts/run_phase12_robustness_ablation.py --list

    # 2. Run full robustness and ablation analysis suite:
    python scripts/run_phase12_robustness_ablation.py --analyze

    # 3. Analyze a specific experiment:
    python scripts/run_phase12_robustness_ablation.py --experiment action_masking
"""

import argparse
from datetime import datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from phd_ludo_drl.analysis.robustness_ablation_framework import (
    RobustnessAblationFramework,
    RESEARCH_EXPERIMENTS,
    FROZEN_PROTOCOL_VERSION,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Phase 12 Robustness & Ablation Runner (Protocol v{FROZEN_PROTOCOL_VERSION})",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--list",
        action="store_true",
        default=False,
        help="List all registered ablation and robustness hypotheses with controlled variables and fixed conditions.",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        default=False,
        help="Execute full robustness and ablation statistical analysis and output publication tables.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        choices=list(RESEARCH_EXPERIMENTS.keys()),
        help="Run analysis for a specific registered experiment.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/tables",
        help="Directory to store publication tables.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    framework = RobustnessAblationFramework(base_dir=PROJECT_ROOT)

    # 1. List Hypotheses
    if args.list or (not args.analyze and not args.experiment):
        print("=" * 88)
        print(f"PHASE 12 ROBUSTNESS & ABLATION EXPERIMENTAL REGISTRY (Protocol v{FROZEN_PROTOCOL_VERSION})")
        print("=" * 88)
        for key, hyp in RESEARCH_EXPERIMENTS.items():
            print(f"\n[{hyp.experiment_id}] {hyp.name} ({hyp.category.value})")
            print(f"  Component Studied:   {hyp.component_studied}")
            print(f"  Controlled Variable: {hyp.controlled_variable} ({hyp.baseline_value} -> {hyp.ablated_value})")
            print(f"  Documented Seed:     {hyp.documented_seed}")
            print(f"  Null Hypothesis H0:  {hyp.h0_null}")
            print(f"  Alt Hypothesis H1:   {hyp.h1_alternative}")
            print(f"  Fixed Conditions:    {hyp.fixed_conditions}")
        print("\n" + "=" * 88)
        print("To run statistical evaluation: python scripts/run_phase12_robustness_ablation.py --analyze")
        return 0

    # 2. Run Analysis
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Executing Phase 12 Robustness & Ablation Analysis...")
    records = framework.run_all_analyses()
    tables = framework.generate_summary_tables(records)

    print("\n" + "=" * 88)
    print("PHASE 12 EXPERIMENTAL OUTCOMES & NEGATIVE FINDINGS AUDIT")
    print("=" * 88)
    for r in records:
        sig_str = "p < 0.05 (*)" if r.is_statistically_significant else "p >= 0.05 (ns)"
        neg_str = "[NEGATIVE FINDING]" if r.is_negative_finding else "[AFFIRMATIVE]"
        print(f"\n{neg_str} {r.name}:")
        print(f"  Baseline:   {r.baseline_value:.2%} ({r.baseline_condition})")
        print(f"  Ablated:    {r.ablated_value:.2%} ({r.ablated_condition})")
        print(f"  Impact:     {r.measured_difference:+.2%} ({r.pct_impact:+.1f}%) | Test: {r.test_statistic:.4f} ({sig_str})")
        print(f"  Effect:     {r.effect_size_measure} = {r.effect_size_value:+.4f} ({r.effect_size_interpretation})")
        if r.is_negative_finding:
            print(f"  Scientific Note: {r.negative_finding_detail}")

    print("\n" + "=" * 88)
    print("ANALYSIS COMPLETED SUCCESSFULLY")
    print(f"Tables saved: {tables['csv']}")
    print(f"Markdown table: {tables['md']}")
    print(f"LaTeX table: {tables['tex']}")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    sys.exit(main())
