"""Unit tests for Phase 13 Final Research Visualizer."""

import json
from pathlib import Path
import pytest
from PIL import Image

from phd_ludo_drl.visualization.final_research_visualizer import (
    FinalResearchVisualizer,
    PALETTE,
    ALGO_SHORT_LABELS,
)


@pytest.fixture
def visualizer(tmp_path):
    project_root = Path(__file__).resolve().parent.parent
    figures_dir = tmp_path / "figures"
    return FinalResearchVisualizer(
        project_root=project_root,
        figures_dir=figures_dir,
    )


def test_visualizer_initialization_and_palette(visualizer):
    """Verify initialization, default paths, and aesthetic color tokens."""
    assert visualizer.figures_dir.exists()
    assert "dqn" in PALETTE
    assert "ppo" in PALETTE
    assert "model3" in PALETTE
    assert "model4" in PALETTE
    assert len(ALGO_SHORT_LABELS) == 4


def test_plot_training_curves(visualizer):
    """Verify Category 1: Training curves generation."""
    path = visualizer.plot_training_curves()
    assert path.exists()
    assert path.stat().st_size > 10000
    with Image.open(path) as img:
        assert img.format == "PNG"
        assert img.width > 1000


def test_plot_model_win_rate_comparison(visualizer):
    """Verify Category 2: Model win-rate comparison generation."""
    path = visualizer.plot_model_win_rate_comparison()
    assert path.exists()
    assert path.stat().st_size > 10000
    with Image.open(path) as img:
        assert img.format == "PNG"


def test_plot_random_opponent_comparison(visualizer):
    """Verify Category 3: Random-opponent comparison generation."""
    path = visualizer.plot_random_opponent_comparison()
    assert path.exists()
    assert path.stat().st_size > 10000
    with Image.open(path) as img:
        assert img.format == "PNG"


def test_plot_expert_opponent_comparison(visualizer):
    """Verify Category 4: Expert-opponent comparison generation."""
    path = visualizer.plot_expert_opponent_comparison()
    assert path.exists()
    assert path.stat().st_size > 10000
    with Image.open(path) as img:
        assert img.format == "PNG"


def test_plot_reward_comparison(visualizer):
    """Verify Category 5: Reward comparison generation."""
    path = visualizer.plot_reward_comparison()
    assert path.exists()
    assert path.stat().st_size > 10000
    with Image.open(path) as img:
        assert img.format == "PNG"


def test_plot_game_length_comparison(visualizer):
    """Verify Category 6: Game-length comparison generation."""
    path = visualizer.plot_game_length_comparison()
    assert path.exists()
    assert path.stat().st_size > 10000
    with Image.open(path) as img:
        assert img.format == "PNG"


def test_plot_confidence_intervals(visualizer):
    """Verify Category 7: Confidence intervals forest plot generation."""
    path = visualizer.plot_confidence_intervals()
    assert path.exists()
    assert path.stat().st_size > 10000
    with Image.open(path) as img:
        assert img.format == "PNG"


def test_plot_seed_variability(visualizer):
    """Verify Category 8: Seed variability profile generation."""
    path = visualizer.plot_seed_variability()
    assert path.exists()
    assert path.stat().st_size > 10000
    with Image.open(path) as img:
        assert img.format == "PNG"


def test_plot_ablation_results(visualizer):
    """Verify Category 9: Ablation results multi-panel generation."""
    path = visualizer.plot_ablation_results()
    assert path.exists()
    assert path.stat().st_size > 10000
    with Image.open(path) as img:
        assert img.format == "PNG"


def test_plot_robustness_results(visualizer):
    """Verify Category 10: Robustness results multi-panel generation."""
    path = visualizer.plot_robustness_results()
    assert path.exists()
    assert path.stat().st_size > 10000
    with Image.open(path) as img:
        assert img.format == "PNG"


def test_run_all_and_manifest_completeness(visualizer):
    """Verify full pipeline execution and manifest completeness."""
    out = visualizer.run_all()
    assert len(out["figures"]) == 10
    assert out["manifest"].exists()

    with open(out["manifest"], "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert len(manifest) == 10
    expected_categories = [
        "Training Curves",
        "Model Win-Rate Comparison",
        "Random-Opponent Comparison",
        "Expert-Opponent Comparison",
        "Reward Comparison",
        "Game-Length Comparison",
        "Confidence Intervals",
        "Seed Variability",
        "Ablation Results",
        "Robustness Results",
    ]
    manifest_categories = [v["category"] for v in manifest.values()]
    for cat in expected_categories:
        assert cat in manifest_categories
