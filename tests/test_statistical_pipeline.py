"""Unit tests for Phase 17 Statistical Analysis Pipeline."""

import numpy as np
import pytest

from phd_ludo_drl.analysis.statistical_pipeline import (
    compute_wilson_score_interval,
    compute_bootstrap_ci,
    compute_students_t_ci,
    compute_cohens_d,
    compute_proportion_effect_sizes,
    run_mcnemar_test,
    run_two_proportion_z_test,
    interpret_cohens_d,
    StatisticalAnalysisPipeline,
)


def test_wilson_score_interval():
    low, high = compute_wilson_score_interval(50, 100, confidence=0.95)
    assert 0.0 <= low < 0.5 < high <= 1.0
    assert round(low, 2) == 0.40
    assert round(high, 2) == 0.60

    # Edge cases
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


def test_pipeline_on_raw_results(tmp_path):
    # Verify pipeline instantiates and runs cleanly
    pipeline = StatisticalAnalysisPipeline(
        processed_dir=tmp_path / "processed",
        tables_dir=tmp_path / "tables",
    )
    res = pipeline.run()
    assert res["num_games"] >= 1000
    assert res["num_tests"] > 0
    assert res["num_effects"] > 0
