"""Run classification and protocol validation for training pipeline runs."""

from enum import Enum
from typing import Any, Dict, Optional


class RunType(str, Enum):
    """Categorical classification of training runs in the PhD pipeline."""
    SMOKE_TEST = "smoke_test"
    PILOT_RUN = "pilot_run"
    FINAL_EXPERIMENT = "final_experiment"


class PipelineProtocolViolationError(ValueError):
    """Raised when an experiment execution violates PhD protocol governance rules."""
    pass


# Strict prefix conventions
RUN_TYPE_PREFIXES = {
    RunType.SMOKE_TEST: "SMOKE_",
    RunType.PILOT_RUN: "PILOT_",
    RunType.FINAL_EXPERIMENT: "EXP_FINAL_",
}

# Subfolder conventions under experiments/
RUN_TYPE_SUBDIRS = {
    RunType.SMOKE_TEST: "smoke_tests",
    RunType.PILOT_RUN: "pilot_runs",
    RunType.FINAL_EXPERIMENT: "final_experiments",
}

# Step thresholds for integrity protection
MAX_SMOKE_TEST_STEPS = 5_000
MAX_PILOT_RUN_STEPS = 100_000
FINAL_EXPERIMENT_STEPS = 500_000


def normalize_run_type(run_type: Any) -> RunType:
    """Normalize and validate a run type input."""
    if isinstance(run_type, RunType):
        return run_type
    if isinstance(run_type, str):
        cleaned = run_type.strip().lower()
        for member in RunType:
            if member.value == cleaned:
                return member
    raise PipelineProtocolViolationError(
        f"Invalid run_type '{run_type}'. Must be one of: {[m.value for m in RunType]}"
    )


def validate_run_classification(
    run_type: RunType,
    experiment_id: str,
    total_steps: int,
    is_final_experiment: bool,
    description: str = "",
) -> None:
    """Enforce strict governance rules preventing protocol confusion or mislabeling.
    
    CRITICAL RULE: A smoke test must NEVER be labelled as a final experiment.
    """
    prefix = RUN_TYPE_PREFIXES[run_type]

    if run_type == RunType.SMOKE_TEST:
        if is_final_experiment:
            raise PipelineProtocolViolationError(
                "CRITICAL INVARIANT VIOLATION: A smoke test must NEVER be labelled as a final experiment! "
                f"Got is_final_experiment=True for smoke test '{experiment_id}'."
            )
        if not experiment_id.startswith(prefix):
            raise PipelineProtocolViolationError(
                f"Smoke test ID must begin with prefix '{prefix}', got '{experiment_id}'."
            )
        if "EXP_FINAL" in experiment_id.upper() or "FINAL_EXPERIMENT" in experiment_id.upper():
            raise PipelineProtocolViolationError(
                f"Smoke test ID '{experiment_id}' contains 'FINAL' labels. Smoke tests must never be labelled as final experiments."
            )
        if "final experiment" in description.lower() and "not a final experiment" not in description.lower():
            raise PipelineProtocolViolationError(
                f"Smoke test description '{description}' claims to be a final experiment. Disallowed by protocol."
            )
        if total_steps > MAX_SMOKE_TEST_STEPS:
            raise PipelineProtocolViolationError(
                f"Smoke test step budget {total_steps} exceeds maximum allowable smoke threshold ({MAX_SMOKE_TEST_STEPS})."
            )

    elif run_type == RunType.PILOT_RUN:
        if is_final_experiment:
            raise PipelineProtocolViolationError(
                f"Pilot run '{experiment_id}' cannot have is_final_experiment=True."
            )
        if not experiment_id.startswith(prefix):
            raise PipelineProtocolViolationError(
                f"Pilot run ID must begin with prefix '{prefix}', got '{experiment_id}'."
            )
        if total_steps > MAX_PILOT_RUN_STEPS:
            raise PipelineProtocolViolationError(
                f"Pilot run step budget {total_steps} exceeds maximum allowable pilot threshold ({MAX_PILOT_RUN_STEPS})."
            )

    elif run_type == RunType.FINAL_EXPERIMENT:
        if not is_final_experiment:
            raise PipelineProtocolViolationError(
                f"Final experiment '{experiment_id}' must have is_final_experiment=True."
            )
        if not experiment_id.startswith(prefix):
            raise PipelineProtocolViolationError(
                f"Final experiment ID must begin with prefix '{prefix}', got '{experiment_id}'."
            )
        if total_steps != FINAL_EXPERIMENT_STEPS:
            raise PipelineProtocolViolationError(
                f"Final experiment step budget must strictly equal {FINAL_EXPERIMENT_STEPS:,} steps under Protocol v1.0.0 (got {total_steps:,})."
            )
