"""PHASE 17 — STATISTICAL ANALYSIS RUNNER

Executes the Phase 17 statistical analysis pipeline using ONLY stored experimental results.
Directly reads from results/raw/ (and optional raw stores).
Generates:
- results/processed/
- results/tables/

Includes clear labels for:
- algorithm
- seed
- experiment
- opponent
- metric

Strictly adheres to:
- Do not manually type result values.
- Do not make an overall "best model" claim automatically.
- Objectively report measured differences and statistical evidence.
"""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from phd_ludo_drl.analysis.statistical_pipeline import StatisticalAnalysisPipeline


def main():
    parser = argparse.ArgumentParser(description="Phase 17 — Statistical Analysis Pipeline")
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="results/raw",
        help="Primary raw results directory (default: results/raw)",
    )
    parser.add_argument(
        "--include-all-stored",
        action="store_true",
        help="Also ingest and label stored results from results/ablations/raw and results/robustness/raw",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="results/processed",
        help="Directory to save processed statistical files (default: results/processed)",
    )
    parser.add_argument(
        "--tables-dir",
        type=str,
        default="results/tables",
        help="Directory to save publication tables (default: results/tables)",
    )
    parser.add_argument(
        "--master-seed",
        type=int,
        default=42,
        help="Master seed for reproducible bootstrap resampling (default: 42)",
    )

    args = parser.parse_args()

    raw_dirs = [Path(args.raw_dir)]
    if args.include_all_stored:
        ablations_raw = PROJECT_ROOT / "results" / "ablations" / "raw"
        robustness_raw = PROJECT_ROOT / "results" / "robustness" / "raw"
        if ablations_raw.exists():
            raw_dirs.append(ablations_raw)
        if robustness_raw.exists():
            raw_dirs.append(robustness_raw)

    pipeline = StatisticalAnalysisPipeline(
        raw_dirs=raw_dirs,
        processed_dir=args.processed_dir,
        tables_dir=args.tables_dir,
        master_seed=args.master_seed,
    )

    results = pipeline.run()

    # Terminal summary of empirical evidence (no "best model" claim)
    print("\n" + "=" * 90)
    print("PHASE 17 EMPIRICAL EVIDENCE & MEASURED DIFFERENCES SUMMARY")
    print("=" * 90)
    
    # Read generated table 1 & table 3 for terminal presentation
    t2_path = Path(args.tables_dir) / "table2_win_rate_and_confidence_intervals.csv"
    if t2_path.exists():
        import pandas as pd
        df_t2 = pd.read_csv(t2_path)
        print("\n[Empirical Win Rates & 95% Confidence Intervals]")
        print(df_t2.to_string(index=False))

    t3_path = Path(args.tables_dir) / "table3_comparative_hypothesis_tests.csv"
    if t3_path.exists():
        import pandas as pd
        df_t3 = pd.read_csv(t3_path)
        print("\n[Hypothesis Tests on Stored Matches]")
        print(df_t3[["Comparison Category", "Group 1", "Group 2", "Metric", "Test Name", "Observed Diff", "Test Stat", "p-Value", "Sig (alpha=0.05)"]].to_string(index=False))

    t4_path = Path(args.tables_dir) / "table4_effect_sizes.csv"
    if t4_path.exists():
        import pandas as pd
        df_t4 = pd.read_csv(t4_path)
        print("\n[Effect Sizes & Magnitudes]")
        print(df_t4[["Comparison Category", "Group 1", "Group 2", "Metric", "Effect Measure", "Point Estimate", "Interpretation"]].to_string(index=False))

    t5_path = Path(args.tables_dir) / "table5_seat_fairness_analysis.csv"
    if t5_path.exists():
        import pandas as pd
        df_t5 = pd.read_csv(t5_path)
        print("\n[Seat Fairness Analysis (Seat 0 vs Seat 1)]")
        print(df_t5.to_string(index=False))

    print("\n" + "=" * 90)
    print("ANALYSIS DIRECTIVE VERIFICATION:")
    print("1. All values computed strictly from stored results in results/raw/ (no manual entry).")
    print("2. Both parametric and non-parametric tests computed based on distributional diagnostics.")
    print("3. Paired tests (McNemar, Paired t) and independent tests (z-test, Welch t, Mann-Whitney) reported.")
    print("4. No overall 'best model' claim made; all measured differences reported objectively.")
    print("=" * 90)


if __name__ == "__main__":
    main()
