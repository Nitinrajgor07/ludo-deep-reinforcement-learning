"""Unified Training Framework for PhD Ludo DRL."""

from phd_ludo_drl.training.factory import (
    SUPPORTED_ALGORITHMS,
    normalize_algorithm_name,
    create_agent,
    create_opponent,
    create_trainer,
)
from phd_ludo_drl.training.unified_trainer import UnifiedTrainingRunner
from phd_ludo_drl.training.run_type import (
    RunType,
    PipelineProtocolViolationError,
    normalize_run_type,
    validate_run_classification,
)
from phd_ludo_drl.training.production_pipeline import (
    ProductionTrainingPipeline,
    RunArtifactPaths,
    MASTER_SEEDS,
    FROZEN_PROTOCOL_VERSION,
)

__all__ = [
    "SUPPORTED_ALGORITHMS",
    "normalize_algorithm_name",
    "create_agent",
    "create_opponent",
    "create_trainer",
    "UnifiedTrainingRunner",
    "ProductionTrainingPipeline",
    "RunType",
    "PipelineProtocolViolationError",
    "normalize_run_type",
    "validate_run_classification",
    "RunArtifactPaths",
    "MASTER_SEEDS",
    "FROZEN_PROTOCOL_VERSION",
]
