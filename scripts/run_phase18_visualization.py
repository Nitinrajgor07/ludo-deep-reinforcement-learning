"""PHASE 18 — RESEARCH VISUALIZATION RUNNER

Executes the Phase 18 visualization pipeline:
- Generates publication-quality figures and tables ONLY from verified experimental data.
- Never hardcodes numerical values into plotting scripts.
- Reads data automatically from disk.
- Saves figures to results/figures/
- Saves tables to results/tables/
"""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from phd_ludo_drl.visualization.research_visualizer import ResearchVisualizer


def main():
    parser = argparse.ArgumentParser(description="Phase 18 — Research Visualization Pipeline")
    parser.add_argument(
        "--figures-dir",
        type=str,
        default="results/figures",
        help="Directory to save figures (default: results/figures)",
    )
    parser.add_argument(
        "--tables-dir",
        type=str,
        default="results/tables",
        help="Directory to save publication tables (default: results/tables)",
    )

    args = parser.parse_args()

    visualizer = ResearchVisualizer(
        project_root=PROJECT_ROOT,
        figures_dir=args.figures_dir,
        tables_dir=args.tables_dir,
    )

    output = visualizer.run_all()

    print("\n" + "=" * 90)
    print("PHASE 18 PUBLICATION DELIVERABLES REGISTRY")
    print("=" * 90)
    print("[Publication Figures (300 DPI)]")
    for name, path in output["figures"].items():
        print(f"  - {name:<25}: {path.name} ({path.stat().st_size / 1024:.1f} KB)")

    print("\n[Publication Tables (CSV, Markdown, LaTeX)]")
    for key, path in output["tables"].items():
        if key.endswith("_md"):
            print(f"  - {path.stem:<35}: {path.name}")

    print("=" * 90)


if __name__ == "__main__":
    main()
