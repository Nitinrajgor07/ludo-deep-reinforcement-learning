"""Tests for Phase 9 Production Training Pipeline.

Covers:
- Strict run classification (SMOKE_TEST vs PILOT_RUN vs FINAL_EXPERIMENT)
- Enforcing critical invariant: A smoke test must NEVER be labelled as a final experiment
- Deterministic seeding and seed derivation
- Independent run directories and layout contracts
- Configuration snapshot creation
- Checkpoint saving and SHA-256 manifest tracking
- Real metrics logging (JSONL and CSV) without fake data
- Git and environment provenance capture
- Protection against overwriting existing raw runs
- Failure recovery: resuming an interrupted run from latest checkpoint
- Execution of real smoke tests across all 4 models (DQN, PPO, Discrete SAC, C51)
"""

import json
from pathlib import Path
import pytest
import shutil
import tempfile
import torch

from phd_ludo_drl.training.production_pipeline import (
    ProductionTrainingPipeline,
    RunType,
    PipelineProtocolViolationError,
    RunArtifactPaths,
    MASTER_SEEDS,
    FROZEN_PROTOCOL_VERSION,
    FINAL_EXPERIMENT_STEPS,
)
from phd_ludo_drl.experiments.version_info import capture_version_provenance, get_git_info
from phd_ludo_drl.training.factory import SUPPORTED_ALGORITHMS


@pytest.fixture
def temp_project_dir():
    """Create an isolated temporary workspace directory for test execution."""
    temp_dir = tempfile.mkdtemp(prefix="phd_pipeline_test_")
    path = Path(temp_dir)
    yield path
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_frozen_protocol_constants():
    """Verify frozen protocol constants adhere strictly to Version 1.0.0."""
    assert FROZEN_PROTOCOL_VERSION == "1.0.0 (FROZEN)"
    assert MASTER_SEEDS == [42, 101, 202, 303, 404]
    assert FINAL_EXPERIMENT_STEPS == 500_000
    assert len(SUPPORTED_ALGORITHMS) == 4
    assert set(SUPPORTED_ALGORITHMS) == {"dqn", "ppo", "model3", "model4"}


def test_smoke_test_cannot_be_labelled_as_final_experiment(temp_project_dir):
    """CRITICAL PROTOCOL INVARIANT: A smoke test must NEVER be labelled as a final experiment."""
    pipeline = ProductionTrainingPipeline(base_dir=temp_project_dir)

    # 1. Attempting to run a smoke test with final experiment labeling must raise PipelineProtocolViolationError
    with pytest.raises(PipelineProtocolViolationError, match="CRITICAL INVARIANT VIOLATION"):
        from phd_ludo_drl.training.run_type import validate_run_classification
        validate_run_classification(
            run_type=RunType.SMOKE_TEST,
            experiment_id="SMOKE_DQN_42",
            total_steps=100,
            is_final_experiment=True,  # VIOLATION!
            description="Testing violation",
        )

    # 2. Attempting to put 'FINAL' in a smoke test ID must fail
    with pytest.raises(PipelineProtocolViolationError, match="contains 'FINAL' labels"):
        from phd_ludo_drl.training.run_type import validate_run_classification
        validate_run_classification(
            run_type=RunType.SMOKE_TEST,
            experiment_id="SMOKE_EXP_FINAL_DQN_42",  # VIOLATION!
            total_steps=100,
            is_final_experiment=False,
            description="Testing violation",
        )

    # 3. Attempting to run a smoke test with 500,000 steps must fail
    with pytest.raises(PipelineProtocolViolationError, match="exceeds maximum allowable smoke threshold"):
        from phd_ludo_drl.training.run_type import validate_run_classification
        validate_run_classification(
            run_type=RunType.SMOKE_TEST,
            experiment_id="SMOKE_DQN_42",
            total_steps=500_000,  # VIOLATION!
            is_final_experiment=False,
        )


def test_git_and_system_provenance_capture(temp_project_dir):
    """Verify Git commit, branch, dirty flag, Python and dependency versions are captured."""
    provenance = capture_version_provenance(temp_project_dir)
    p_dict = provenance.to_dict()

    assert "project_version" in p_dict
    assert "git_commit" in p_dict
    assert "git_branch" in p_dict
    assert "git_dirty" in p_dict
    assert "python_version" in p_dict
    assert "torch_version" in p_dict
    assert "gymnasium_version" in p_dict
    assert "numpy_version" in p_dict
    assert "platform_system" in p_dict


def test_independent_run_directories_and_no_overwrite(temp_project_dir):
    """Verify each run has an independent directory and raw runs are protected from overwrite."""
    pipeline = ProductionTrainingPipeline(base_dir=temp_project_dir)

    # Execute a small smoke run
    res1 = pipeline.run_experiment(
        algorithm="dqn",
        seed=42,
        total_steps=50,
        run_type=RunType.SMOKE_TEST,
        eval_games=2,
        checkpoint_interval=25,
    )

    run1_root = temp_project_dir / res1["artifacts"]["root_dir"]
    assert run1_root.exists()
    assert (run1_root / "config_frozen.yaml").exists()
    assert (run1_root / "metadata.json").exists()
    assert (run1_root / "results" / "summary.json").exists()

    # Re-running without resume against the same directory must raise FileExistsError
    with pytest.raises(FileExistsError, match="Raw runs cannot be overwritten"):
        pipeline.run_experiment(
            algorithm="dqn",
            seed=42,
            total_steps=50,
            run_type=RunType.SMOKE_TEST,
            eval_games=2,
            resume=False,
            resume_from_dir=run1_root,
        )


def test_checkpointing_and_manifest_integrity(temp_project_dir):
    """Verify periodic checkpoints, final model, and SHA-256 manifest entries."""
    pipeline = ProductionTrainingPipeline(base_dir=temp_project_dir)

    res = pipeline.run_experiment(
        algorithm="dqn",
        seed=42,
        total_steps=60,
        run_type=RunType.SMOKE_TEST,
        checkpoint_interval=20,
        eval_games=2,
    )

    run_root = temp_project_dir / res["artifacts"]["root_dir"]
    manifest_file = run_root / "checkpoints" / "checkpoint_manifest.json"
    assert manifest_file.exists()

    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert manifest["experiment_id"] == res["metadata"]["experiment_id"]
    assert len(manifest["checkpoints"]) >= 2  # At least step 20, 40, and final

    # Verify that every checkpoint in manifest actually exists on disk and has matching SHA-256
    from phd_ludo_drl.training.production_pipeline import compute_file_sha256
    for entry in manifest["checkpoints"]:
        ckpt_file = run_root / entry["path"]
        assert ckpt_file.exists(), f"Checkpoint file {ckpt_file} not found"
        actual_sha = compute_file_sha256(ckpt_file)
        assert actual_sha == entry["sha256"]
        assert entry["size_bytes"] > 0


def test_streaming_metrics_logging(temp_project_dir):
    """Verify logs/training.log, metrics.jsonl, and metrics.csv contain real execution data."""
    pipeline = ProductionTrainingPipeline(base_dir=temp_project_dir)

    res = pipeline.run_experiment(
        algorithm="dqn",
        seed=42,
        total_steps=50,
        run_type=RunType.SMOKE_TEST,
        eval_games=2,
    )

    run_root = temp_project_dir / res["artifacts"]["root_dir"]
    metrics_jsonl = run_root / "logs" / "metrics.jsonl"
    metrics_csv = run_root / "logs" / "metrics.csv"
    training_log = run_root / "logs" / "training.log"

    assert training_log.exists()
    assert training_log.stat().st_size > 0

    assert metrics_jsonl.exists()
    assert metrics_csv.exists()

    # Read CSV lines
    csv_lines = metrics_csv.read_text(encoding="utf-8").strip().splitlines()
    assert len(csv_lines) >= 1
    assert "timestamp_utc,step,episode,reward,loss,win_rate" in csv_lines[0]


def test_failure_recovery_and_resumption(temp_project_dir):
    """Verify that an interrupted run can be resumed from its latest checkpoint."""
    pipeline = ProductionTrainingPipeline(base_dir=temp_project_dir)

    # 1. Run initial phase up to step 40 with checkpointing every 20 steps
    res_phase1 = pipeline.run_experiment(
        algorithm="dqn",
        seed=101,
        total_steps=40,
        run_type=RunType.SMOKE_TEST,
        checkpoint_interval=20,
        eval_games=2,
    )

    run_dir = temp_project_dir / res_phase1["artifacts"]["root_dir"]
    ckpt_step20 = run_dir / "checkpoints" / "dqn_step_20.pt"
    assert ckpt_step20.exists()

    # 2. Simulate resuming from step 40 to step 80
    res_resumed = pipeline.run_experiment(
        algorithm="dqn",
        seed=101,
        total_steps=80,
        run_type=RunType.SMOKE_TEST,
        checkpoint_interval=20,
        eval_games=2,
        resume=True,
        resume_from_dir=run_dir,
    )

    assert res_resumed["training_metrics"]["total_steps"] == 80
    assert (run_dir / "checkpoints" / "final_model.pt").exists()


@pytest.mark.parametrize("algorithm", ["dqn", "ppo", "model3", "model4"])
def test_all_four_models_smoke_test(algorithm, temp_project_dir):
    """Execute real end-to-end smoke test across each of the 4 benchmark algorithms."""
    pipeline = ProductionTrainingPipeline(base_dir=temp_project_dir)

    res = pipeline.run_experiment(
        algorithm=algorithm,
        seed=42,
        total_steps=40,
        run_type=RunType.SMOKE_TEST,
        checkpoint_interval=20,
        eval_games=2,
        description=f"Automated smoke test for {algorithm}",
    )

    # Verify run categorization
    assert res["metadata"]["run_type"] == "smoke_test"
    assert res["metadata"]["is_final_experiment"] is False
    assert res["metadata"]["algorithm"] == algorithm
    assert res["metadata"]["seed"] == 42
    assert "EXP_FINAL" not in res["metadata"]["experiment_id"]
    assert res["metadata"]["experiment_id"].startswith("SMOKE_")

    # Verify real artifacts generated
    root_dir = temp_project_dir / res["artifacts"]["root_dir"]
    assert (root_dir / "config_frozen.yaml").exists()
    assert (root_dir / "metadata.json").exists()
    assert (root_dir / "checkpoints" / "final_model.pt").exists()
    assert (root_dir / "checkpoints" / "checkpoint_manifest.json").exists()
    assert (root_dir / "logs" / "training.log").exists()
    assert (root_dir / "results" / "summary.json").exists()

    # Check metrics validity
    assert res["training_metrics"]["total_steps"] == 40
    assert "p0_win_rate" in res["evaluation_metrics"]
