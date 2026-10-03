"""Tests for experiment identifier generation and directory resolution."""

from pathlib import Path
from phd_ludo_drl.experiments.experiment_id import ExperimentIdentifier, ExperimentMetadata


def test_experiment_identifier_creation():
    """Verify experiment metadata creation with valid naming conventions."""
    meta = ExperimentIdentifier.create(
        algorithm="dqn",
        seed=42,
        encoding="structured_cnn",
        description="Baseline DQN test run",
        config_dict={"lr": 0.0001, "gamma": 0.99},
    )

    assert isinstance(meta, ExperimentMetadata)
    assert meta.algorithm == "dqn"
    assert meta.seed == 42
    assert meta.encoding == "structured_cnn"
    assert "EXP_DQN_structured_cnn_42_" in meta.experiment_id
    assert len(meta.config_hash) == 8


def test_experiment_directory_resolution(tmp_path: Path):
    """Verify standardized subdirectories are created on resolution."""
    exp_id = "EXP_TEST_123"
    paths = ExperimentIdentifier.resolve_paths(tmp_path, exp_id)

    assert paths["root"].exists()
    assert paths["checkpoints"].exists()
    assert paths["logs"].exists()
    assert paths["results"].exists()
