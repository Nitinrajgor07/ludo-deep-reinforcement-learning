"""Evaluation package for PhD Ludo DRL.

Standardized 5,000-game evaluation benchmark and 200,000-game final evaluation pipeline.
"""

from phd_ludo_drl.evaluation.evaluator import (
    StandardizedEvaluator,
    MatchupGameRecord,
    calculate_wilson_ci,
    calculate_bootstrap_ci,
)
from phd_ludo_drl.evaluation.final_evaluation_pipeline import (
    FinalEvaluationPipeline,
    FinalEvaluationConfig,
    CheckpointValidator,
    RawEvaluationStorage,
    MatchupEvaluationRunner,
    MissingCheckpointError,
    RawDataProtectedError,
    ModelMetadata,
    CheckpointMetadata,
    OpponentMetadata,
    RawGameRecord,
    CheckpointInspectionResult,
    FROZEN_PROTOCOL_VERSION,
    BENCHMARK_MODELS,
    MASTER_SEEDS,
    EVALUATION_OPPONENTS,
    GAMES_PER_OPPONENT,
    REQUIRED_FINAL_STEPS,
    TOTAL_PLANNED_GAMES,
)

__all__ = [
    "StandardizedEvaluator",
    "MatchupGameRecord",
    "calculate_wilson_ci",
    "calculate_bootstrap_ci",
    "FinalEvaluationPipeline",
    "FinalEvaluationConfig",
    "CheckpointValidator",
    "RawEvaluationStorage",
    "MatchupEvaluationRunner",
    "MissingCheckpointError",
    "RawDataProtectedError",
    "ModelMetadata",
    "CheckpointMetadata",
    "OpponentMetadata",
    "RawGameRecord",
    "CheckpointInspectionResult",
    "FROZEN_PROTOCOL_VERSION",
    "BENCHMARK_MODELS",
    "MASTER_SEEDS",
    "EVALUATION_OPPONENTS",
    "GAMES_PER_OPPONENT",
    "REQUIRED_FINAL_STEPS",
    "TOTAL_PLANNED_GAMES",
]
