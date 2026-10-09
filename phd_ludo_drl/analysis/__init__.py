"""Analysis package for PhD Ludo DRL.

Statistical analysis pipeline, ablations, stochasticity, and robustness framework.
"""

from phd_ludo_drl.analysis.statistical_pipeline import (
    StatisticalAnalysisPipeline,
    DataTier,
    StatisticalAnalysisError,
    IncompleteRawDataError,
    compute_wilson_score_interval,
    compute_bootstrap_ci,
    compute_students_t_ci,
    compute_cohens_d,
    compute_proportion_effect_sizes,
    compute_wilcoxon_r,
    run_mcnemar_test,
    run_two_proportion_z_test,
    bonferroni_correction,
    holm_bonferroni_correction,
    benjamini_hochberg_correction,
    interpret_cohens_d,
    TOTAL_REQUIRED_FINAL_GAMES,
    REQUIRED_FINAL_STEPS,
    FROZEN_PROTOCOL_VERSION,
)
from phd_ludo_drl.analysis.robustness_ablation_framework import (
    RobustnessAblationFramework,
    AblationHypothesis,
    AblationResultRecord,
    ExperimentCategory,
    RESEARCH_EXPERIMENTS,
)

__all__ = [
    "StatisticalAnalysisPipeline",
    "DataTier",
    "StatisticalAnalysisError",
    "IncompleteRawDataError",
    "compute_wilson_score_interval",
    "compute_bootstrap_ci",
    "compute_students_t_ci",
    "compute_cohens_d",
    "compute_proportion_effect_sizes",
    "compute_wilcoxon_r",
    "run_mcnemar_test",
    "run_two_proportion_z_test",
    "bonferroni_correction",
    "holm_bonferroni_correction",
    "benjamini_hochberg_correction",
    "interpret_cohens_d",
    "TOTAL_REQUIRED_FINAL_GAMES",
    "REQUIRED_FINAL_STEPS",
    "FROZEN_PROTOCOL_VERSION",
    "RobustnessAblationFramework",
    "AblationHypothesis",
    "AblationResultRecord",
    "ExperimentCategory",
    "RESEARCH_EXPERIMENTS",
]
