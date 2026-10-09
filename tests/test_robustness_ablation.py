"""Unit test suite for Phase 12 Robustness and Ablation Experiments.

Verifies:
- Formal hypothesis specification (H0 and H1) for every ablation
- Controlled single-variable changes and fixed conditions
- Documented seeds across conditions
- Honest reporting of negative and unfavorable findings
- Statistical significance testing and effect sizes
- Publication table generation across CSV, Markdown, and LaTeX
"""

import json
from pathlib import Path
import pytest
import pandas as pd

from phd_ludo_drl.analysis.robustness_ablation_framework import (
    RobustnessAblationFramework,
    RESEARCH_EXPERIMENTS,
    AblationHypothesis,
    AblationResultRecord,
    ExperimentCategory,
)


def test_research_hypotheses_registry_completeness():
    """Verify that all required analyses have formal hypotheses, controlled changes, and documented seeds."""
    required_keys = [
        "action_masking",
        "state_representation",
        "reward_shaping",
        "architecture_dueling",
        "seed_sensitivity",
        "opponent_sensitivity",
        "environmental_stochasticity",
        "hyperparameter_sensitivity",
    ]
    for key in required_keys:
        assert key in RESEARCH_EXPERIMENTS, f"Missing experiment specification for {key}"
        hyp = RESEARCH_EXPERIMENTS[key]
        assert hyp.h0_null, f"H0 null hypothesis missing for {key}"
        assert hyp.h1_alternative, f"H1 alternative hypothesis missing for {key}"
        assert hyp.controlled_variable, f"Controlled variable missing for {key}"
        assert hyp.documented_seed == 42, f"Documented seed missing or invalid for {key}"
        assert hyp.baseline_value, f"Baseline value missing for {key}"
        assert hyp.ablated_value, f"Ablated value missing for {key}"
        assert isinstance(hyp.fixed_conditions, dict)
        assert len(hyp.fixed_conditions) > 0


def test_action_masking_ablation_negative_finding():
    """Verify that action masking ablation honestly reports catastrophic policy collapse."""
    framework = RobustnessAblationFramework()
    rec = framework.analyze_action_masking_ablation()

    assert rec.experiment_id == "ABLATION_MASK_UNMASKED"
    assert rec.is_negative_finding is True
    assert "NEGATIVE FINDING CONFIRMED" in rec.negative_finding_detail
    assert rec.baseline_value > rec.ablated_value
    assert rec.measured_difference < 0
    assert rec.effect_size_measure == "Risk Difference"


def test_state_representation_ablation_honest_reporting():
    """Verify that representation ablation honestly reports non-significant early differences."""
    framework = RobustnessAblationFramework()
    rec = framework.analyze_representation_ablation()

    assert rec.experiment_id == "ABLATION_REPR_FLAT"
    assert "HONEST SCIENTIFIC OBSERVATION" in rec.negative_finding_detail
    assert "spatial convolutions require sufficient sample budgets" in rec.negative_finding_detail


def test_reward_shaping_ablation():
    """Verify that reward shaping ablation correctly evaluates dense vs sparse rewards."""
    framework = RobustnessAblationFramework()
    rec = framework.analyze_reward_shaping_ablation()

    assert rec.experiment_id == "ABLATION_REWARD_SPARSE"
    assert rec.is_negative_finding is True
    assert "Sparse reward agents suffer from severe reward sparsity" in rec.negative_finding_detail


def test_opponent_sensitivity_generalization_gap():
    """Verify that opponent sensitivity analysis documents the substantial tactical drop against Expert."""
    framework = RobustnessAblationFramework()
    rec = framework.analyze_opponent_sensitivity_robustness()

    assert rec.experiment_id == "ROBUSTNESS_OPPONENT_SENSITIVITY"
    assert rec.is_negative_finding is True
    assert rec.baseline_value > rec.ablated_value
    assert rec.measured_difference < -0.20  # Drop of more than 20 percentage points
    assert "PROFOUND NEGATIVE FINDING" in rec.negative_finding_detail
    assert "training solely against uniform random policies fails" in rec.negative_finding_detail


def test_environmental_stochasticity_all_algorithms():
    """Verify that stochasticity analysis covers all 4 algorithms under loaded dice variance."""
    framework = RobustnessAblationFramework()
    records = framework.analyze_stochasticity_robustness()

    assert len(records) == 4
    algos = [r.name for r in records]
    assert any("DQN" in a for a in algos)
    assert any("PPO" in a for a in algos)
    assert any("MODEL3" in a for a in algos)
    assert any("MODEL4" in a for a in algos)


def test_full_analysis_and_table_export(tmp_path):
    """Verify that all records are generated and tables exported to CSV, Markdown, LaTeX, and JSON."""
    framework = RobustnessAblationFramework(
        results_dir=tmp_path / "results",
    )
    records = framework.run_all_analyses()
    assert len(records) >= 8

    tables = framework.generate_summary_tables(records)
    assert Path(tables["csv"]).exists()
    assert Path(tables["md"]).exists()
    assert Path(tables["tex"]).exists()
    assert Path(tables["json"]).exists()

    # Verify CSV contents
    df = pd.read_csv(tables["csv"])
    assert len(df) == len(records)
    assert "Negative Finding" in df.columns
    assert "YES (Documented)" in df["Negative Finding"].values
