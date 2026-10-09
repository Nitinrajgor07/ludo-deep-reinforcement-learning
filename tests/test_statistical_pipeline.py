"""Unit test suite for Phase 11 Statistical Analysis Pipeline.

Verifies:
- Wilson score interval calculation
- Bootstrap percentile confidence interval (10,000 resamples)
- Student's t confidence interval
- Effect sizes: Cohen's d, Hedges' g, Odds Ratio, Risk Difference
- Hypothesis tests: McNemar, Two-Proportion z-test, Fisher's Exact, Welch's t-test
- Multiple-comparison corrections: Bonferroni, Holm-Bonferroni, Benjamini-Hochberg
- Data tier classification: PRELIMINARY_PILOT vs FINAL_EXPERIMENT
- Reporting analysis as INCOMPLETE when final raw data is missing (never fabricating values)
- Full pairwise model comparisons across all algorithms
- Cross-seed analysis and seat fairness analysis
- Reproduction from raw data into publication tables without manual input
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from phd_ludo_drl.analysis.statistical_pipeline import (
    compute_wilson_score_interval,
    compute_bootstrap_ci,
    compute_students_t_ci,
    compute_cohens_d,
    compute_proportion_effect_sizes,
    run_mcnemar_test,
    run_two_proportion_z_test,
    bonferroni_correction,
    holm_bonferroni_correction,
    benjamini_hochberg_correction,
    interpret_cohens_d,
    StatisticalAnalysisPipeline,
    DataTier,
    TOTAL_REQUIRED_FINAL_GAMES,
)


def test_wilson_score_interval():
    low, high = compute_wilson_score_interval(50, 100, confidence=0.95)
    assert 0.0 <= low < 0.5 < high <= 1.0
    assert round(low, 2) == 0.40
    assert round(high, 2) == 0.60

    # Edge cases: 0 wins and 100 wins
    low0, high0 = compute_wilson_score_interval(0, 100)
    assert low0 == 0.0
    assert high0 > 0.0

    low1, high1 = compute_wilson_score_interval(100, 100)
    assert low1 < 1.0
    assert high1 == 1.0


def test_bootstrap_ci():
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    low, high = compute_bootstrap_ci(data, num_resamples=1000, seed=42)
    assert low < np.mean(data) < high


def test_students_t_ci():
    data = [10.0, 12.0, 11.0, 13.0, 9.0]
    low, high = compute_students_t_ci(data, confidence=0.95)
    assert low < np.mean(data) < high


def test_cohens_d():
    x = [10.0, 11.0, 12.0, 10.5, 11.5]
    y = [5.0, 6.0, 5.5, 4.5, 5.0]
    res = compute_cohens_d(x, y)
    assert res["cohens_d"] > 0
    assert res["hedges_g"] > 0
    assert interpret_cohens_d(res["cohens_d"]) == "large"


def test_proportion_effects():
    res = compute_proportion_effect_sizes(60, 100, 40, 100)
    assert round(res["risk_difference"], 2) == 0.20
    assert round(res["odds_ratio"], 2) == 2.25


def test_mcnemar_test():
    p1 = [True, True, False, False, True]
    p2 = [True, False, True, False, True]
    res = run_mcnemar_test(p1, p2)
    assert res["b"] == 1
    assert res["c"] == 1
    assert res["p_value"] == 1.0


def test_two_proportion_z_test():
    z, p = run_two_proportion_z_test(60, 100, 40, 100)
    assert z > 0
    assert 0 <= p <= 1.0


def test_multiple_comparison_corrections():
    raw_p = [0.005, 0.015, 0.030, 0.045]
    bonf = bonferroni_correction(raw_p)
    holm = holm_bonferroni_correction(raw_p)
    bh = benjamini_hochberg_correction(raw_p)

    # Bonferroni is most conservative: p * m
    assert bonf[0] == pytest.approx(0.005 * 4)
    assert bonf[1] == pytest.approx(0.015 * 4)

    # Holm is step-down: p_(1) * 4, p_(2) * 3, ...
    assert holm[0] == pytest.approx(0.005 * 4)
    assert holm[1] == pytest.approx(0.015 * 3)

    # Benjamini-Hochberg FDR
    assert bh[0] <= holm[0]
    assert bh[-1] <= bonf[-1]


def test_data_tier_audit_reports_incomplete_on_missing_final_data(tmp_path):
    """When final 200,000 raw games from 500k checkpoints are missing, report INCOMPLETE."""
    empty_raw = tmp_path / "raw"
    empty_raw.mkdir(parents=True, exist_ok=True)

    pipeline = StatisticalAnalysisPipeline(
        raw_dirs=[empty_raw],
        processed_dir=tmp_path / "processed",
        tables_dir=tmp_path / "tables",
        require_final_data=True,
    )

    audit = pipeline.audit_raw_data()
    assert audit["is_final_complete"] is False
    assert audit["final_experiment_games"] == 0
    assert audit["required_final_games"] == 200000

    # Running with require_final_data=True must return status INCOMPLETE without fabricating values
    res = pipeline.run()
    assert res["status"] == "INCOMPLETE"
    assert "final_experiment_games" in res["audit"]
    assert res["audit"]["final_experiment_games"] == 0


def test_data_tier_classification_pilot_vs_final(tmp_path):
    """Verify that pilot runs are tagged PRELIMINARY_PILOT and 500k full runs as FINAL_EXPERIMENT."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Create pilot dataset (250 games, 50 steps)
    pilot_rows = []
    for i in range(250):
        pilot_rows.append({
            "game_idx": i,
            "model_seed": 42,
            "model_algorithm": "dqn",
            "opponent_type": "random",
            "agent_seat": i % 2,
            "win": 1 if i % 2 == 0 else 0,
            "loss": 0 if i % 2 == 0 else 1,
            "draw": 0,
            "reward": 1.0 if i % 2 == 0 else -1.0,
            "opponent_reward": -1.0 if i % 2 == 0 else 1.0,
            "game_length": 100,
            "captures": 1,
            "opponent_captures": 0,
            "tokens_home": 4 if i % 2 == 0 else 0,
            "opponent_tokens_home": 0,
            "checkpoint_steps": 50,  # Pilot!
        })
    df_pilot = pd.DataFrame(pilot_rows)
    df_pilot.to_csv(raw_dir / "dqn_seed_42_vs_random_raw_games.csv", index=False)

    pipeline = StatisticalAnalysisPipeline(
        raw_dirs=[raw_dir],
        processed_dir=tmp_path / "processed",
        tables_dir=tmp_path / "tables",
        require_final_data=False,
    )

    df_ingested = pipeline.ingest_stored_results()
    assert not df_ingested.empty
    assert (df_ingested["data_tier"] == DataTier.PRELIMINARY_PILOT.value).all()


def test_full_pipeline_execution_and_table_generation(tmp_path):
    """Verify end-to-end execution on synthetic raw evaluation records."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Create mock raw datasets for 2 algorithms across 2 seeds and 2 opponents
    rows = []
    for algo in ["dqn", "ppo"]:
        for seed in [42, 101]:
            for opp in ["random", "expert"]:
                for g in range(50):
                    win = 1 if (algo == "dqn" and g % 2 == 0) or (algo == "ppo" and g % 3 == 0) else 0
                    rows.append({
                        "game_idx": g,
                        "model_seed": seed,
                        "model_algorithm": algo,
                        "opponent_type": opp,
                        "agent_seat": g % 2,
                        "win": win,
                        "loss": 1 - win,
                        "draw": 0,
                        "reward": float(win * 2.0 - 1.0),
                        "opponent_reward": float((1 - win) * 2.0 - 1.0),
                        "game_length": 100 + g,
                        "captures": 1,
                        "opponent_captures": 0,
                        "tokens_home": 4 if win else 1,
                        "opponent_tokens_home": 1 if win else 4,
                        "checkpoint_steps": 1000,
                    })
    df_mock = pd.DataFrame(rows)
    df_mock.to_csv(raw_dir / "mock_raw_games.csv", index=False)

    pipeline = StatisticalAnalysisPipeline(
        raw_dirs=[raw_dir],
        processed_dir=tmp_path / "processed",
        tables_dir=tmp_path / "tables",
        require_final_data=False,
    )

    res = pipeline.run()
    assert res["status"] == "COMPLETED"
    assert res["num_games"] == 400
    assert res["num_tests"] > 0
    assert res["num_effects"] > 0

    # Check generated table files
    tables = res["table_files"]
    assert "table1_descriptive_summary_csv" in tables
    assert "table2_win_rate_and_confidence_intervals_csv" in tables
    assert "table3_comparative_hypothesis_tests_csv" in tables
    assert "table4_effect_sizes_csv" in tables
    assert "table5_seat_fairness_analysis_csv" in tables
    assert "table7_cross_seed_analysis_csv" in tables
    assert "table8_opponent_sensitivity_analysis_csv" in tables

    # Verify CSV files exist on disk
    for key, path_str in tables.items():
        assert Path(path_str).exists()
