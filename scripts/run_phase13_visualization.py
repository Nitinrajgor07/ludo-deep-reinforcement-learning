"""PHASE 13 — FINAL RESEARCH VISUALIZATION RUNNER

Executes the Phase 13 visualization pipeline:
- Generates publication-quality figures across 10 mandatory categories.
- Ingests empirical data dynamically from verified disk records (results/raw/, results/processed/, experiments/).
- Strictly adheres to publication standards: 300 DPI, consistent typography, non-misleading axes.
- Saves figures separately from raw data (results/figures/).
- Generates figure manifest with full scientific captions.
"""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from phd_ludo_drl.visualization.final_research_visualizer import FinalResearchVisualizer


def main():
    parser = argparse.ArgumentParser(
        description="Phase 13 — Final Research Visualization Pipeline"
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default="results/figures",
        help="Directory to save figures (default: results/figures)",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        choices=[
            "training_curves",
            "model_win_rate_comparison",
            "random_opponent_comparison",
            "expert_opponent_comparison",
            "reward_comparison",
            "game_length_comparison",
            "confidence_intervals",
            "seed_variability",
            "ablation_results",
            "robustness_results",
        ],
        help="Optional: Run only a specific category. If omitted, runs all 10.",
    )

    args = parser.parse_args()

    visualizer = FinalResearchVisualizer(
        project_root=PROJECT_ROOT,
        figures_dir=args.figures_dir,
    )

    if args.category:
        method_name = f"plot_{args.category}"
        if hasattr(visualizer, method_name):
            print(f"Running individual visualization: {args.category}...")
            out_path = getattr(visualizer, method_name)()
            print(f"Saved: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
        else:
            print(f"Error: Method {method_name} not found.")
            sys.exit(1)
    else:
        output = visualizer.run_all()
        print("\n" + "=" * 90)
        print("PHASE 13 PUBLICATION DELIVERABLES REGISTRY")
        print("=" * 90)
        for name, path in output["figures"].items():
            print(f"  - {name:<35}: {path.name:<32} ({path.stat().st_size / 1024:.1f} KB, 300 DPI)")
        print(f"  - Manifest: {output['manifest'].name}")
        print("=" * 90)


if __name__ == "__main__":
    main()
