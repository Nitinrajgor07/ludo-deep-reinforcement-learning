"""Unit test suite for Phase 10 Final Evaluation Pipeline.

Verifies:
- 200,000 total games configuration contract (4 models x 5 seeds x 2 opponents x 5,000 games)
- Strict detection and reporting of missing 500k checkpoints
- Refusal to substitute pilot checkpoints as final
- Strict "Never overwrite raw evaluation data" enforcement
- Balanced seating (50% Player 0, 50% Player 1 alternating)
- Deterministic action selection and independent game seeds
- Raw game-level record completeness
"""

import json
from pathlib import Path
import tempfile
import pytest
import torch

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
    calculate_wilson_ci,
    calculate_bootstrap_ci,
    BENCHMARK_MODELS,
    MASTER_SEEDS,
    EVALUATION_OPPONENTS,
    GAMES_PER_OPPONENT,
    REQUIRED_FINAL_STEPS,
    TOTAL_PLANNED_GAMES,
)
from phd_ludo_drl.config.config_manager import ConfigManager
from phd_ludo_drl.training.factory import create_agent


def test_protocol_constants_and_configuration():
    """Verify Phase 10 mathematical and structural contracts."""
    cfg = FinalEvaluationConfig()
    assert cfg.models == ["dqn", "ppo", "model3", "model4"]
    assert cfg.seeds == [42, 101, 202, 303, 404]
    assert cfg.opponents == ["random", "expert"]
    assert cfg.games_per_opponent == 5000
    assert cfg.required_min_steps == 500000
    assert cfg.allow_overwrite is False
    assert cfg.total_planned_games == 200000


def test_checkpoint_validator_detects_missing(tmp_path):
    """Validator must detect missing checkpoints and report them accurately."""
    validator = CheckpointValidator(base_dir=tmp_path, min_required_steps=500000)
    audit = validator.validate_all(models=["dqn"], seeds=[42])
    assert audit["all_ready"] is False
    assert audit["missing_count"] == 1
    assert audit["ready_count"] == 0

    res = audit["inspections"]["dqn_seed_42"]
    assert res.status_code == "MISSING"
    assert res.is_500k_valid is False
    assert res.is_pilot is False


def test_checkpoint_validator_rejects_pilot_checkpoint(tmp_path):
    """Validator must flag a checkpoint with <500,000 steps as PILOT_INSUFFICIENT."""
    # Create fake pilot checkpoint with 500 steps
    pilot_dir = tmp_path / "experiments" / "EXP_DQN_structured_cnn_42" / "checkpoints"
    pilot_dir.mkdir(parents=True, exist_ok=True)
    pilot_ckpt = pilot_dir / "final_model.pt"

    torch.save({
        "total_steps": 500,
        "online_net": {},
        "target_net": {},
    }, pilot_ckpt)

    validator = CheckpointValidator(base_dir=tmp_path, min_required_steps=500000)
    res = validator.inspect_single(model="dqn", seed=42)

    assert res.exists is True
    assert res.steps == 500
    assert res.is_500k_valid is False
    assert res.is_pilot is True
    assert res.status_code == "PILOT_INSUFFICIENT"
    assert "CANNOT substitute pilot checkpoint" in res.status_message


def test_validator_accepts_valid_500k_checkpoint(tmp_path):
    """Validator must accept checkpoints with >= 500,000 steps."""
    valid_dir = tmp_path / "experiments" / "EXP_DQN_structured_cnn_42" / "checkpoints"
    valid_dir.mkdir(parents=True, exist_ok=True)
    valid_ckpt = valid_dir / "final_model.pt"

    torch.save({
        "total_steps": 500000,
        "online_net": {},
        "target_net": {},
    }, valid_ckpt)

    validator = CheckpointValidator(base_dir=tmp_path, min_required_steps=500000)
    res = validator.inspect_single(model="dqn", seed=42)

    assert res.exists is True
    assert res.steps == 500000
    assert res.is_500k_valid is True
    assert res.is_pilot is False
    assert res.status_code == "READY_500K"


def test_pipeline_halts_and_raises_on_missing_checkpoints(tmp_path):
    """Pipeline must STOP and raise MissingCheckpointError when 500k checkpoints are missing."""
    cfg = FinalEvaluationConfig(
        models=["dqn"],
        seeds=[42],
        required_min_steps=500000,
    )
    pipeline = FinalEvaluationPipeline(base_dir=tmp_path, config=cfg)

    with pytest.raises(MissingCheckpointError) as exc_info:
        pipeline.run_pipeline(dry_run=False)

    assert "Protocol Invariant Violation" in str(exc_info.value)
    assert "Do not substitute pilot checkpoints" in str(exc_info.value)


def test_raw_data_storage_never_overwrites(tmp_path):
    """Storage engine must raise RawDataProtectedError when target raw files already exist."""
    raw_dir = tmp_path / "raw"
    storage = RawEvaluationStorage(raw_dir=raw_dir)

    model_meta = ModelMetadata(
        algorithm_key="dqn",
        algorithm_name="Double Dueling DQN",
        agent_class="DQNAgent",
        encoder_architecture="StructuredCNNEncoder",
        action_space_size=4,
        evaluation_policy="argmax",
        action_masking="masked",
    )
    ckpt_meta = CheckpointMetadata(
        checkpoint_path="fake.pt",
        training_steps=500000,
        training_seed=42,
        sha256_hash="abc",
        file_size_bytes=100,
        last_modified_utc="2026-10-06T00:00:00Z",
        is_500k_final=True,
    )
    opp_meta = OpponentMetadata(
        opponent_type="random",
        opponent_class="RandomOpponent",
        strategy_description="Uniform",
        is_deterministic=False,
    )

    rec = RawGameRecord(
        game_idx=0,
        game_seed=1042,
        model_algorithm="dqn",
        model_seed=42,
        opponent_type="random",
        agent_seat=0,
        win=1,
        loss=0,
        draw=0,
        reward=1.0,
        opponent_reward=-1.0,
        game_length=120,
        captures=2,
        opponent_captures=0,
        tokens_home=4,
        opponent_tokens_home=1,
        terminated=True,
        truncated=False,
    )

    # First write should succeed
    csv_p, jsonl_p, meta_p = storage.write_matchup_data(
        records=[rec],
        model_meta=model_meta,
        ckpt_meta=ckpt_meta,
        opp_meta=opp_meta,
        allow_overwrite=False,
    )
    assert csv_p.exists()
    assert jsonl_p.exists()

    # Second write with allow_overwrite=False must fail
    with pytest.raises(RawDataProtectedError) as exc_info:
        storage.write_matchup_data(
            records=[rec],
            model_meta=model_meta,
            ckpt_meta=ckpt_meta,
            opp_meta=opp_meta,
            allow_overwrite=False,
        )

    assert "Protocol Invariant: NEVER overwrite raw evaluation data" in str(exc_info.value)


def test_matchup_runner_balanced_seating_and_metrics(tmp_path):
    """Verify seat alternation, independent seeds, and metric computation on small game count."""
    # Create a small valid test checkpoint for DQN
    cfg = ConfigManager.load_config()
    agent = create_agent("dqn", cfg)
    ckpt_path = tmp_path / "test_dqn.pt"
    agent.save_checkpoint(ckpt_path)

    runner = MatchupEvaluationRunner(max_turns=200)
    # Run 4 games: 2 as P0, 2 as P1
    records, summary = runner.run_matchup(
        algorithm="dqn",
        checkpoint_path=ckpt_path,
        seed=42,
        opponent_type="random",
        num_games=4,
        base_seed=1000,
    )

    assert len(records) == 4
    # Check balanced seating: indices 0, 2 -> P0; 1, 3 -> P1
    assert records[0].agent_seat == 0
    assert records[1].agent_seat == 1
    assert records[2].agent_seat == 0
    assert records[3].agent_seat == 1

    # Check independent unique seeds
    seeds = [r.game_seed for r in records]
    assert len(set(seeds)) == 4

    # Check record fields
    for r in records:
        assert r.win in (0, 1)
        assert r.loss in (0, 1)
        assert r.draw in (0, 1)
        assert r.win + r.loss + r.draw == 1
        assert r.game_length > 0
        assert r.illegal_actions_attempted == 0

    assert summary["p0_games"] == 2
    assert summary["p1_games"] == 2
    assert summary["total_games"] == 4


def test_statistical_ci_calculations():
    """Verify Wilson score and bootstrap CI bounds."""
    # 50 wins out of 100
    w_low, w_high = calculate_wilson_ci(50, 100, confidence=0.95)
    assert 0.40 <= w_low <= 0.41
    assert 0.59 <= w_high <= 0.61

    # Bootstrap CI
    data = [1.0, 1.0, 1.0, 0.0, 0.0]
    b_low, b_high = calculate_bootstrap_ci(data, num_resamples=1000, seed=42)
    assert 0.2 <= b_low <= 0.6
    assert 0.6 <= b_high <= 1.0
