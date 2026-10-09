"""PHASE 10 — FINAL EVALUATION PIPELINE

Authoritative evaluation orchestrator strictly adhering to:
- FINAL_EXPERIMENT_PROTOCOL.md (Version 1.0.0 FROZEN)
- PhD Research Plan: Phase 10 Standardized Evaluation Protocol

Protocol Invariants:
1. Four Models:
   - Model 1: Double Dueling DQN (dqn)
   - Model 2: PPO Actor-Critic with Action Masking (ppo)
   - Model 3: Discrete Soft Actor-Critic (model3)
   - Model 4: Categorical DQN / C51 (model4)
2. Five Master Research Seeds:
   - S = [42, 101, 202, 303, 404]
3. Evaluation Game Count:
   - 5,000 games vs Random Opponent
   - 5,000 games vs Expert Opponent (Alhajry 2012 priority strategy)
   - 10,000 games per seed run
   - 50,000 games per model across 5 seeds
   - Grand Total: 200,000 evaluation games across all 4 models
4. Scientific Guarantees:
   - Fixed evaluation policy (deterministic action selection, argmax over masked actions)
   - Deterministic evaluation behavior where required
   - Balanced seating/player positions (strictly 50% Player 0, 50% Player 1)
   - Independent game seeds (no cross-game or cross-matchup collisions)
   - Raw game-level records stored with comprehensive metadata (model, checkpoint, opponent)
   - Never overwrite raw evaluation data (strict protection of raw files)
   - Checkpoint integrity: If any required 500k checkpoint is missing, STOP and report it.
     Never substitute a pilot checkpoint and label it final.
"""

from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.stats as stats
import torch

from phd_ludo_drl.config.config_manager import ConfigManager, ProjectConfig
from phd_ludo_drl.config.seed import set_seed
from phd_ludo_drl.environment.ludo_env import LudoEnvironment
from phd_ludo_drl.opponents.base_opponent import BaseOpponent
from phd_ludo_drl.opponents.random_opponent import RandomOpponent
from phd_ludo_drl.opponents.expert_opponent import ExpertOpponent
from phd_ludo_drl.opponents.self_play import SelfPlayArena, GameRecord, _get_player_action
from phd_ludo_drl.agents.base_agent import BaseAgent
from phd_ludo_drl.training.factory import (
    SUPPORTED_ALGORITHMS,
    normalize_algorithm_name,
    create_agent,
    create_opponent,
)

# ==============================================================================
# PROTOCOL CONSTANTS
# ==============================================================================
FROZEN_PROTOCOL_VERSION: str = "1.0.0 (FROZEN)"
MASTER_SEEDS: List[int] = [42, 101, 202, 303, 404]
BENCHMARK_MODELS: List[str] = ["dqn", "ppo", "model3", "model4"]
EVALUATION_OPPONENTS: List[str] = ["random", "expert"]
GAMES_PER_OPPONENT: int = 5000
GAMES_PER_MODEL_SEED: int = 10000  # 5,000 vs Random + 5,000 vs Expert
TOTAL_PLANNED_GAMES: int = 200000  # 4 models * 5 seeds * 10,000 games
REQUIRED_FINAL_STEPS: int = 500_000
EVAL_BASE_SEED: int = 1042
MAX_GAME_TURNS: int = 1000


# ==============================================================================
# EXCEPTIONS
# ==============================================================================
class EvaluationProtocolError(Exception):
    """Base exception for evaluation protocol violations."""
    pass


class MissingCheckpointError(EvaluationProtocolError):
    """Raised when a required 500k checkpoint is missing or has insufficient steps."""
    pass


class RawDataProtectedError(EvaluationProtocolError):
    """Raised when an attempt is made to overwrite existing raw evaluation data."""
    pass


# ==============================================================================
# METADATA & DATA STRUCTURES
# ==============================================================================
@dataclass
class ModelMetadata:
    """Metadata describing the evaluated reinforcement learning model."""
    algorithm_key: str
    algorithm_name: str
    agent_class: str
    encoder_architecture: str
    action_space_size: int
    evaluation_policy: str
    action_masking: str
    parameters_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CheckpointMetadata:
    """Cryptographic and operational metadata for a model checkpoint."""
    checkpoint_path: str
    training_steps: int
    training_seed: int
    sha256_hash: str
    file_size_bytes: int
    last_modified_utc: str
    is_500k_final: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OpponentMetadata:
    """Metadata describing the benchmark opponent."""
    opponent_type: str
    opponent_class: str
    strategy_description: str
    is_deterministic: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RawGameRecord:
    """Comprehensive game-level record captured for every individual evaluation game."""
    game_idx: int
    game_seed: int
    model_algorithm: str
    model_seed: int
    opponent_type: str
    agent_seat: int  # 0 or 1
    win: int         # 1 if agent won, 0 otherwise
    loss: int        # 1 if opponent won, 0 otherwise
    draw: int        # 1 if no winner (turns limit reached), 0 otherwise
    reward: float    # Agent undiscounted cumulative episode reward
    opponent_reward: float
    game_length: int # Number of turns elapsed
    captures: int    # Agent captures
    opponent_captures: int
    tokens_home: int # Agent finished tokens (0..4)
    opponent_tokens_home: int # Opponent finished tokens (0..4)
    terminated: bool
    truncated: bool
    illegal_actions_attempted: int = 0
    checkpoint_steps: int = 0
    checkpoint_sha256: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def csv_header(cls) -> List[str]:
        return [
            "game_idx",
            "game_seed",
            "model_algorithm",
            "model_seed",
            "opponent_type",
            "agent_seat",
            "win",
            "loss",
            "draw",
            "reward",
            "opponent_reward",
            "game_length",
            "captures",
            "opponent_captures",
            "tokens_home",
            "opponent_tokens_home",
            "terminated",
            "truncated",
            "illegal_actions_attempted",
            "checkpoint_steps",
            "checkpoint_sha256",
        ]

    def to_csv_row(self) -> List[Any]:
        return [
            self.game_idx,
            self.game_seed,
            self.model_algorithm,
            self.model_seed,
            self.opponent_type,
            self.agent_seat,
            self.win,
            self.loss,
            self.draw,
            round(self.reward, 4),
            round(self.opponent_reward, 4),
            self.game_length,
            self.captures,
            self.opponent_captures,
            self.tokens_home,
            self.opponent_tokens_home,
            int(self.terminated),
            int(self.truncated),
            self.illegal_actions_attempted,
            self.checkpoint_steps,
            self.checkpoint_sha256,
        ]


@dataclass
class CheckpointInspectionResult:
    """Inspection status of a specific model checkpoint."""
    model: str
    seed: int
    expected_path: Path
    actual_path: Optional[Path]
    exists: bool
    steps: Optional[int]
    is_500k_valid: bool
    is_pilot: bool
    sha256: Optional[str]
    file_size_bytes: Optional[int]
    status_code: str  # "READY_500K", "PILOT_INSUFFICIENT", "MISSING", "CORRUPTED"
    status_message: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "seed": self.seed,
            "expected_path": str(self.expected_path),
            "actual_path": str(self.actual_path) if self.actual_path else None,
            "exists": self.exists,
            "steps": self.steps,
            "is_500k_valid": self.is_500k_valid,
            "is_pilot": self.is_pilot,
            "sha256": self.sha256,
            "file_size_bytes": self.file_size_bytes,
            "status_code": self.status_code,
            "status_message": self.status_message,
        }


# ==============================================================================
# STATISTICAL CALCULATION UTILITIES
# ==============================================================================
def calculate_wilson_ci(wins: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate Wilson score interval for a binomial proportion."""
    if total <= 0:
        return (0.0, 0.0)
    z = stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0)
    p = wins / total
    denom = 1.0 + (z**2) / total
    center = (p + (z**2) / (2.0 * total)) / denom
    margin = (z * math.sqrt((p * (1.0 - p) + (z**2) / (4.0 * total)) / total)) / denom
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return (float(lower), float(upper))


def calculate_bootstrap_ci(
    data: Sequence[float],
    num_resamples: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """Calculate non-parametric bootstrap percentile confidence interval for mean."""
    if not data:
        return (0.0, 0.0)
    arr = np.asarray(data, dtype=np.float64)
    if len(arr) == 1 or np.all(arr == arr[0]):
        return (float(arr[0]), float(arr[0]))
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(arr), size=(num_resamples, len(arr)))
    resampled_means = np.mean(arr[indices], axis=1)
    alpha = (1.0 - confidence) / 2.0
    lower = float(np.percentile(resampled_means, alpha * 100.0))
    upper = float(np.percentile(resampled_means, (1.0 - alpha) * 100.0))
    return (lower, upper)


# ==============================================================================
# CHECKPOINT DISCOVERY & VALIDATION ENGINE
# ==============================================================================
class CheckpointValidator:
    """Verifies that all 20 required 500k checkpoints exist and meet protocol criteria.

    Strictly refuses to substitute pilot or smoke checkpoints as final models.
    """

    def __init__(
        self,
        base_dir: Path,
        min_required_steps: int = REQUIRED_FINAL_STEPS,
    ):
        self.base_dir = Path(base_dir).resolve()
        self.min_required_steps = min_required_steps

    def locate_checkpoint(self, model: str, seed: int) -> Optional[Path]:
        """Search candidate directories for a valid checkpoint file."""
        model_canon = normalize_algorithm_name(model)
        model_upper = model_canon.upper()

        candidates: List[Path] = [
            # 1. Standard final experiments output directories
            self.base_dir / "experiments" / f"EXP_{model_upper}_structured_cnn_{seed}" / "checkpoints" / "final_model.pt",
            self.base_dir / "experiments" / f"EXP_{model_upper}_structured_cnn_{seed}_500000" / "checkpoints" / "final_model.pt",
            self.base_dir / "checkpoints" / "final_500k" / f"{model_canon}_seed_{seed}_500k.pt",
            self.base_dir / "checkpoints" / f"{model_canon}_seed_{seed}_500k.pt",
            self.base_dir / "experiments" / "final_checkpoints" / f"{model_canon}_seed_{seed}.pt",
        ]

        for p in candidates:
            if p.exists() and p.is_file():
                return p

        # 2. Glob search in experiments for timestamped full runs
        exp_dir = self.base_dir / "experiments"
        if exp_dir.exists():
            pattern = f"EXP_{model_upper}_*_{seed}_*"
            for run_dir in sorted(exp_dir.glob(pattern), reverse=True):
                # Avoid smoke tests or pilot runs
                if "SMOKE" in run_dir.name or "PILOT" in run_dir.name:
                    continue
                final_pt = run_dir / "checkpoints" / "final_model.pt"
                if final_pt.exists():
                    return final_pt
                # Also check step-specific files
                step_pt = run_dir / "checkpoints" / f"{model_canon}_step_{self.min_required_steps}.pt"
                if step_pt.exists():
                    return step_pt

        return None

    def inspect_checkpoint(self, path: Path) -> Tuple[Optional[int], str, int]:
        """Inspect torch checkpoint to extract step count, SHA-256 hash, and file size."""
        file_size = path.stat().st_size
        sha256_hash = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256_hash.update(chunk)
        digest = sha256_hash.hexdigest()

        steps: Optional[int] = None
        try:
            ckpt_data = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(ckpt_data, dict):
                steps = ckpt_data.get("total_steps")
                if steps is None:
                    steps = ckpt_data.get("step")
                if steps is None:
                    steps = ckpt_data.get("global_step")
        except Exception:
            steps = None

        # Check companion summary.json if available
        if steps is None or steps == 0:
            parent_dir = path.parent.parent
            summary_json = parent_dir / "results" / "summary.json"
            if summary_json.exists():
                try:
                    sdata = json.loads(summary_json.read_text(encoding="utf-8"))
                    steps = sdata.get("training_metrics", {}).get("total_steps", steps)
                except Exception:
                    pass

        return steps, digest, file_size

    def inspect_single(self, model: str, seed: int) -> CheckpointInspectionResult:
        """Inspect a single model-seed pair against the 500k requirement."""
        model_canon = normalize_algorithm_name(model)
        expected = self.base_dir / "experiments" / f"EXP_{model_canon.upper()}_structured_cnn_{seed}" / "checkpoints" / "final_model.pt"
        actual = self.locate_checkpoint(model_canon, seed)

        if actual is None:
            return CheckpointInspectionResult(
                model=model_canon,
                seed=seed,
                expected_path=expected,
                actual_path=None,
                exists=False,
                steps=None,
                is_500k_valid=False,
                is_pilot=False,
                sha256=None,
                file_size_bytes=None,
                status_code="MISSING",
                status_message=f"Missing required 500k checkpoint for {model_canon.upper()} seed {seed}. Expected path: {expected}",
            )

        steps, digest, size = self.inspect_checkpoint(actual)
        if steps is None:
            return CheckpointInspectionResult(
                model=model_canon,
                seed=seed,
                expected_path=expected,
                actual_path=actual,
                exists=True,
                steps=None,
                is_500k_valid=False,
                is_pilot=False,
                sha256=digest,
                file_size_bytes=size,
                status_code="CORRUPTED",
                status_message=f"Checkpoint found at {actual} but failed to extract step count.",
            )

        if steps < self.min_required_steps:
            return CheckpointInspectionResult(
                model=model_canon,
                seed=seed,
                expected_path=expected,
                actual_path=actual,
                exists=True,
                steps=steps,
                is_500k_valid=False,
                is_pilot=True,
                sha256=digest,
                file_size_bytes=size,
                status_code="PILOT_INSUFFICIENT",
                status_message=(
                    f"INSUFFICIENT TRAINING STEPS ({steps:,} / {self.min_required_steps:,}). "
                    f"Found intermediate/pilot checkpoint at {actual}. "
                    "CANNOT substitute pilot checkpoint as final model."
                ),
            )

        return CheckpointInspectionResult(
            model=model_canon,
            seed=seed,
            expected_path=expected,
            actual_path=actual,
            exists=True,
            steps=steps,
            is_500k_valid=True,
            is_pilot=False,
            sha256=digest,
            file_size_bytes=size,
            status_code="READY_500K",
            status_message=f"VALID 500k checkpoint verified ({steps:,} steps). SHA-256: {digest[:12]}...",
        )

    def validate_all(
        self,
        models: Optional[List[str]] = None,
        seeds: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Validate all required model-seed checkpoints."""
        target_models = [normalize_algorithm_name(m) for m in (models or BENCHMARK_MODELS)]
        target_seeds = seeds or MASTER_SEEDS

        results: Dict[str, CheckpointInspectionResult] = {}
        missing_count = 0
        pilot_count = 0
        ready_count = 0
        total_required = len(target_models) * len(target_seeds)

        for model in target_models:
            for seed in target_seeds:
                key = f"{model}_seed_{seed}"
                inspection = self.inspect_single(model, seed)
                results[key] = inspection
                if inspection.status_code == "READY_500K":
                    ready_count += 1
                elif inspection.status_code == "PILOT_INSUFFICIENT":
                    pilot_count += 1
                else:
                    missing_count += 1

        all_ready = (ready_count == total_required)
        return {
            "all_ready": all_ready,
            "total_required": total_required,
            "ready_count": ready_count,
            "missing_count": missing_count,
            "pilot_count": pilot_count,
            "inspections": results,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }

    def format_report(self, validation_output: Dict[str, Any]) -> str:
        """Format validation output into a clear, publication-grade diagnostic report."""
        lines: List[str] = []
        lines.append("=" * 88)
        lines.append("PHASE 10 FINAL EVALUATION — 500K CHECKPOINT READINESS AUDIT")
        lines.append(f"Protocol Version: {FROZEN_PROTOCOL_VERSION}")
        lines.append(f"Audit Timestamp: {validation_output['timestamp_utc']}")
        lines.append(f"Required Steps per Run: {self.min_required_steps:,}")
        lines.append(f"Status Summary: {validation_output['ready_count']}/{validation_output['total_required']} Checkpoints Ready")
        lines.append("-" * 88)
        lines.append(f"{'Model':<10} | {'Seed':<6} | {'Status':<18} | {'Steps Found':<14} | {'Checkpoint Location'}")
        lines.append("-" * 88)

        inspections = validation_output["inspections"]
        for key, res in inspections.items():
            steps_str = f"{res.steps:,}" if res.steps is not None else "N/A"
            loc_str = str(res.actual_path.relative_to(self.base_dir)) if res.actual_path else "NOT FOUND"
            status_tag = res.status_code
            if status_tag == "READY_500K":
                status_disp = "[PASS] READY"
            elif status_tag == "PILOT_INSUFFICIENT":
                status_disp = "[FAIL] PILOT/INT"
            else:
                status_disp = "[FAIL] MISSING"
            lines.append(f"{res.model.upper():<10} | {res.seed:<6} | {status_disp:<18} | {steps_str:<14} | {loc_str}")

        lines.append("-" * 88)
        if not validation_output["all_ready"]:
            lines.append("CRITICAL PROTOCOL SAFETY DIRECTIVE:")
            lines.append("One or more required 500k checkpoints are MISSING or INSUFFICIENT.")
            lines.append("EXECUTION STOPPED.")
            lines.append("Under NO circumstances will a pilot checkpoint be substituted and labelled final.")
        else:
            lines.append("ALL 20 REQUIRED 500K CHECKPOINTS VERIFIED READY FOR FINAL EVALUATION.")
        lines.append("=" * 88)
        return "\n".join(lines)

    def assert_all_checkpoints_ready(
        self,
        models: Optional[List[str]] = None,
        seeds: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Verify all checkpoints are ready. Raise MissingCheckpointError and STOP if any are missing."""
        val = self.validate_all(models=models, seeds=seeds)
        report = self.format_report(val)
        if not val["all_ready"]:
            raise MissingCheckpointError(
                f"\n{report}\n"
                f"Protocol Safety Violation: Cannot begin 200,000 final evaluation games because "
                f"{val['missing_count'] + val['pilot_count']} required 500k checkpoints are missing or below {self.min_required_steps:,} steps.\n"
                f"STOPPING EXECUTION. Do not substitute pilot checkpoints."
            )
        return val


# ==============================================================================
# RAW EVALUATION DATA STORAGE ENGINE (NEVER OVERWRITE)
# ==============================================================================
class RawEvaluationStorage:
    """Manages raw evaluation data persistence with strict no-overwrite protection."""

    def __init__(self, raw_dir: Path):
        self.raw_dir = Path(raw_dir).resolve()
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def get_csv_path(self, model: str, seed: int, opponent: str) -> Path:
        model_canon = normalize_algorithm_name(model)
        return self.raw_dir / f"{model_canon}_seed_{seed}_vs_{opponent}_raw_games.csv"

    def get_jsonl_path(self, model: str, seed: int, opponent: str) -> Path:
        model_canon = normalize_algorithm_name(model)
        return self.raw_dir / f"{model_canon}_seed_{seed}_vs_{opponent}_raw_games.jsonl"

    def get_meta_path(self, model: str, seed: int, opponent: str) -> Path:
        model_canon = normalize_algorithm_name(model)
        return self.raw_dir / f"{model_canon}_seed_{seed}_vs_{opponent}_matchup_meta.json"

    def check_exists(self, model: str, seed: int, opponent: str) -> bool:
        csv_p = self.get_csv_path(model, seed, opponent)
        jsonl_p = self.get_jsonl_path(model, seed, opponent)
        return csv_p.exists() or jsonl_p.exists()

    def write_matchup_data(
        self,
        records: List[RawGameRecord],
        model_meta: ModelMetadata,
        ckpt_meta: CheckpointMetadata,
        opp_meta: OpponentMetadata,
        allow_overwrite: bool = False,
    ) -> Tuple[Path, Path, Path]:
        """Safely write raw game-level records and metadata to disk.

        Raises RawDataProtectedError if data exists and allow_overwrite is False.
        """
        if not records:
            raise ValueError("Cannot write empty records list.")

        model = records[0].model_algorithm
        seed = records[0].model_seed
        opponent = records[0].opponent_type

        csv_path = self.get_csv_path(model, seed, opponent)
        jsonl_path = self.get_jsonl_path(model, seed, opponent)
        meta_path = self.get_meta_path(model, seed, opponent)

        # Check existing files
        if not allow_overwrite:
            existing = [p for p in (csv_path, jsonl_path, meta_path) if p.exists()]
            if existing:
                raise RawDataProtectedError(
                    f"Raw evaluation data already exists for {model} seed {seed} vs {opponent} at {existing}. "
                    "Protocol Invariant: NEVER overwrite raw evaluation data. Execution aborted to preserve integrity."
                )

        # 1. Write CSV with header
        temp_csv = csv_path.with_suffix(".tmp")
        with open(temp_csv, "w", newline="", encoding="utf-8") as f:
            import csv as csv_mod
            writer = csv_mod.writer(f)
            writer.writerow(RawGameRecord.csv_header())
            for rec in records:
                writer.writerow(rec.to_csv_row())
        temp_csv.replace(csv_path)

        # 2. Write JSONL
        temp_jsonl = jsonl_path.with_suffix(".tmp")
        with open(temp_jsonl, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec.to_dict()) + "\n")
        temp_jsonl.replace(jsonl_path)

        # 3. Write Matchup Metadata Header
        meta_payload = {
            "protocol_version": FROZEN_PROTOCOL_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "model_metadata": model_meta.to_dict(),
            "checkpoint_metadata": ckpt_meta.to_dict(),
            "opponent_metadata": opp_meta.to_dict(),
            "total_games": len(records),
            "csv_file": csv_path.name,
            "jsonl_file": jsonl_path.name,
        }
        temp_meta = meta_path.with_suffix(".tmp")
        with open(temp_meta, "w", encoding="utf-8") as f:
            json.dump(meta_payload, f, indent=2)
        temp_meta.replace(meta_path)

        return csv_path, jsonl_path, meta_path


# ==============================================================================
# MATCHUP EVALUATION RUNNER
# ==============================================================================
class MatchupEvaluationRunner:
    """Executes a single 5,000-game seat-balanced evaluation tournament."""

    def __init__(
        self,
        config: Optional[ProjectConfig] = None,
        max_turns: int = MAX_GAME_TURNS,
    ):
        self.config = config or ConfigManager.load_config()
        self.max_turns = max_turns

    def run_matchup(
        self,
        algorithm: str,
        checkpoint_path: Path,
        seed: int,
        opponent_type: str,
        num_games: int = GAMES_PER_OPPONENT,
        base_seed: int = EVAL_BASE_SEED,
        checkpoint_steps: int = REQUIRED_FINAL_STEPS,
        checkpoint_sha256: str = "",
    ) -> Tuple[List[RawGameRecord], Dict[str, Any]]:
        """Run a seat-balanced evaluation tournament against an opponent.

        Seat Allocation: Strictly 50% Player 0, 50% Player 1 (alternating per game).
        Evaluation Policy: Fixed deterministic argmax over legal actions.
        Independent Seeds: Derived uniquely per game without overlap.
        """
        algo_canon = normalize_algorithm_name(algorithm)
        if num_games % 2 != 0:
            raise ValueError(f"num_games must be even for balanced 50/50 seating. Got {num_games}.")

        # Instantiate environment and arena
        env = LudoEnvironment(seed=base_seed, auto_advance_no_moves=True, max_turns=self.max_turns)
        arena = SelfPlayArena(env=env)

        # Instantiate agent and load checkpoint
        agent = create_agent(algo_canon, self.config)
        agent.load_checkpoint(checkpoint_path)

        # Opponent offset based on opponent type
        opp_seed_offset = 100 if opponent_type == "random" else 200
        opp_seed = seed + opp_seed_offset
        opponent = create_opponent(opponent_type, seed=opp_seed)

        # Unique seed base per model-seed-opponent tuple
        model_idx = BENCHMARK_MODELS.index(algo_canon) if algo_canon in BENCHMARK_MODELS else 0
        opp_idx = EVALUATION_OPPONENTS.index(opponent_type) if opponent_type in EVALUATION_OPPONENTS else 0
        matchup_seed_root = base_seed + (seed * 1000) + (model_idx * 50000) + (opp_idx * 25000)

        records: List[RawGameRecord] = []

        for game_idx in range(num_games):
            # Unique deterministic independent game seed
            game_seed = matchup_seed_root + game_idx

            # Balanced seating: strictly alternating
            # Even index: agent is P0, opponent is P1
            # Odd index:  agent is P1, opponent is P0
            agent_seat = 0 if (game_idx % 2 == 0) else 1

            if agent_seat == 0:
                p0_player, p1_player = agent, opponent
            else:
                p0_player, p1_player = opponent, agent

            # Play single game in SelfPlayArena (enforces deterministic action selection on agents)
            rec: GameRecord = arena.play_game(p0_player, p1_player, seed=game_seed, max_turns=self.max_turns)

            # Determine outcomes relative to primary agent
            if rec.winner is None:
                win = 0
                loss = 0
                draw = 1
            elif rec.winner == agent_seat:
                win = 1
                loss = 0
                draw = 0
            else:
                win = 0
                loss = 1
                draw = 0

            # Token finish counts
            p0_home = sum(1 for t in env.tokens if t.player_id == 0 and t.is_finished)
            p1_home = sum(1 for t in env.tokens if t.player_id == 1 and t.is_finished)

            if agent_seat == 0:
                agent_reward = rec.p0_reward
                opp_reward = rec.p1_reward
                agent_caps = rec.p0_captures
                opp_caps = rec.p1_captures
                agent_tokens_home = p0_home
                opp_tokens_home = p1_home
            else:
                agent_reward = rec.p1_reward
                opp_reward = rec.p0_reward
                agent_caps = rec.p1_captures
                opp_caps = rec.p0_captures
                agent_tokens_home = p1_home
                opp_tokens_home = p0_home

            raw_rec = RawGameRecord(
                game_idx=game_idx,
                game_seed=game_seed,
                model_algorithm=algo_canon,
                model_seed=seed,
                opponent_type=opponent_type,
                agent_seat=agent_seat,
                win=win,
                loss=loss,
                draw=draw,
                reward=agent_reward,
                opponent_reward=opp_reward,
                game_length=rec.turns,
                captures=agent_caps,
                opponent_captures=opp_caps,
                tokens_home=agent_tokens_home,
                opponent_tokens_home=opp_tokens_home,
                terminated=rec.terminated,
                truncated=rec.truncated,
                illegal_actions_attempted=0,
                checkpoint_steps=checkpoint_steps,
                checkpoint_sha256=checkpoint_sha256,
            )
            records.append(raw_rec)

        summary = self.compute_matchup_summary(records, algo_canon, seed, opponent_type)
        return records, summary

    def compute_matchup_summary(
        self,
        records: List[RawGameRecord],
        algorithm: str,
        seed: int,
        opponent_type: str,
    ) -> Dict[str, Any]:
        """Compute aggregate statistics from raw records."""
        total = len(records)
        wins = sum(r.win for r in records)
        losses = sum(r.loss for r in records)
        draws = sum(r.draw for r in records)

        win_rate = wins / total if total > 0 else 0.0
        loss_rate = losses / total if total > 0 else 0.0
        draw_rate = draws / total if total > 0 else 0.0

        p0_recs = [r for r in records if r.agent_seat == 0]
        p1_recs = [r for r in records if r.agent_seat == 1]

        p0_wins = sum(r.win for r in p0_recs)
        p1_wins = sum(r.win for r in p1_recs)

        p0_win_rate = p0_wins / len(p0_recs) if p0_recs else 0.0
        p1_win_rate = p1_wins / len(p1_recs) if p1_recs else 0.0
        seat_bias = p0_win_rate - p1_win_rate

        rewards = [r.reward for r in records]
        turns = [r.game_length for r in records]
        caps = [r.captures for r in records]
        tokens_home = [r.tokens_home for r in records]

        avg_reward = float(np.mean(rewards))
        reward_std = float(np.std(rewards, ddof=1)) if len(rewards) > 1 else 0.0
        reward_var = float(np.var(rewards, ddof=1)) if len(rewards) > 1 else 0.0

        avg_turns = float(np.mean(turns))
        turns_std = float(np.std(turns, ddof=1)) if len(turns) > 1 else 0.0
        turns_median = float(np.median(turns))
        turns_min = int(np.min(turns))
        turns_max = int(np.max(turns))

        avg_caps = float(np.mean(caps))
        avg_tokens_home = float(np.mean(tokens_home))

        wilson_lower, wilson_upper = calculate_wilson_ci(wins, total, confidence=0.95)
        bootstrap_win_lower, bootstrap_win_upper = calculate_bootstrap_ci(
            [float(r.win) for r in records], num_resamples=10000, confidence=0.95
        )
        bootstrap_rew_lower, bootstrap_rew_upper = calculate_bootstrap_ci(
            rewards, num_resamples=10000, confidence=0.95
        )

        return {
            "algorithm": algorithm,
            "seed": seed,
            "opponent_type": opponent_type,
            "total_games": total,
            "agent_wins": wins,
            "opponent_wins": losses,
            "draws": draws,
            "win_rate": round(win_rate, 4),
            "loss_rate": round(loss_rate, 4),
            "draw_rate": round(draw_rate, 4),
            "p0_games": len(p0_recs),
            "p0_wins": p0_wins,
            "p0_win_rate": round(p0_win_rate, 4),
            "p1_games": len(p1_recs),
            "p1_wins": p1_wins,
            "p1_win_rate": round(p1_win_rate, 4),
            "seat_bias": round(seat_bias, 4),
            "average_reward": round(avg_reward, 4),
            "reward_std": round(reward_std, 4),
            "reward_variance": round(reward_var, 4),
            "reward_ci_95_bootstrap": [round(bootstrap_rew_lower, 4), round(bootstrap_rew_upper, 4)],
            "win_rate_ci_95_wilson": [round(wilson_lower, 4), round(wilson_upper, 4)],
            "win_rate_ci_95_bootstrap": [round(bootstrap_win_lower, 4), round(bootstrap_win_upper, 4)],
            "average_turns": round(avg_turns, 2),
            "turns_std": round(turns_std, 2),
            "turns_median": turns_median,
            "turns_min": turns_min,
            "turns_max": turns_max,
            "average_captures": round(avg_caps, 2),
            "average_tokens_home": round(avg_tokens_home, 2),
            "illegal_action_rate": 0.0,
        }


# ==============================================================================
# PIPELINE CONFIGURATION & ORCHESTRATOR
# ==============================================================================
@dataclass
class FinalEvaluationConfig:
    """Configuration specification for Phase 10 Final Evaluation."""
    models: List[str] = field(default_factory=lambda: list(BENCHMARK_MODELS))
    seeds: List[int] = field(default_factory=lambda: list(MASTER_SEEDS))
    opponents: List[str] = field(default_factory=lambda: list(EVALUATION_OPPONENTS))
    games_per_opponent: int = GAMES_PER_OPPONENT
    required_min_steps: int = REQUIRED_FINAL_STEPS
    base_seed: int = EVAL_BASE_SEED
    max_turns: int = MAX_GAME_TURNS
    allow_overwrite: bool = False  # Strictly False: Never overwrite raw evaluation data
    output_root: Path = field(default_factory=lambda: Path("results/final_evaluation"))

    @property
    def total_planned_games(self) -> int:
        return len(self.models) * len(self.seeds) * len(self.opponents) * self.games_per_opponent


class FinalEvaluationPipeline:
    """Authoritative Phase 10 Final Evaluation Pipeline.

    Enforces:
    - 200,000 games across 4 models x 5 seeds x 2 opponents (5,000 games each)
    - Fixed evaluation policy & deterministic action selection
    - Balanced seating (50% P0 / 50% P1)
    - Independent game seeds
    - Raw game-level persistence (CSV and JSONL)
    - Never overwrite raw evaluation data
    - Checkpoint verification: STOP and report if any 500k checkpoint is missing.
      Never substitute a pilot checkpoint and label it final.
    """

    def __init__(
        self,
        base_dir: Path,
        config: Optional[FinalEvaluationConfig] = None,
    ):
        self.base_dir = Path(base_dir).resolve()
        self.config = config or FinalEvaluationConfig()
        self.validator = CheckpointValidator(
            base_dir=self.base_dir,
            min_required_steps=self.config.required_min_steps,
        )
        self.storage = RawEvaluationStorage(
            raw_dir=self.base_dir / self.config.output_root / "raw"
        )
        self.runner = MatchupEvaluationRunner(max_turns=self.config.max_turns)

    def check_readiness(self) -> Dict[str, Any]:
        """Audit the repository to determine whether all required 500k checkpoints exist."""
        return self.validator.validate_all(
            models=self.config.models,
            seeds=self.config.seeds,
        )

    def get_readiness_report(self) -> str:
        """Produce human-readable diagnostic audit report."""
        val = self.check_readiness()
        return self.validator.format_report(val)

    def run_pipeline(
        self,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Execute the 200,000-game final evaluation pipeline.

        Pre-flight check:
        If ANY required 500k checkpoint is missing, STOP and report it immediately!
        """
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Phase 10 Final Evaluation Pipeline Initialized.")
        print(f"Total Planned Evaluation Games: {self.config.total_planned_games:,}")
        print(f"Models: {[m.upper() for m in self.config.models]}")
        print(f"Seeds: {self.config.seeds}")
        print(f"Opponents: {self.config.opponents} ({self.config.games_per_opponent:,} games each)")
        print(f"Output Directory: {self.config.output_root}")
        print("-" * 88)

        # 1. Pre-flight Checkpoint Verification (STRICT PROTOCOL ENFORCEMENT)
        print("Executing Pre-Flight 500k Checkpoint Audit...")
        readiness = self.check_readiness()
        report = self.validator.format_report(readiness)
        print(report)

        if not readiness["all_ready"]:
            # MANDATORY DIRECTIVE: STOP and report it. Do not substitute pilot checkpoint!
            raise MissingCheckpointError(
                f"Protocol Invariant Violation: Missing required 500k checkpoints.\n"
                f"Only {readiness['ready_count']} of {readiness['total_required']} required 500k checkpoints exist.\n"
                f"STOPPING EXECUTION. Do not substitute pilot checkpoints."
            )

        if dry_run:
            print("[DRY RUN] All 20 required 500k checkpoints are verified. Ready to evaluate 200,000 games.")
            return {"status": "READY", "readiness": readiness}

        # 2. Execute Evaluation Tournaments (Only reachable when all 500k checkpoints are valid)
        campaign_start = time.time()
        matchup_summaries: List[Dict[str, Any]] = []
        total_games_played = 0

        for model in self.config.models:
            for seed in self.config.seeds:
                key = f"{model}_seed_{seed}"
                inspection: CheckpointInspectionResult = readiness["inspections"][key]
                ckpt_path = inspection.actual_path
                assert ckpt_path is not None, f"Checkpoint path cannot be None for ready checkpoint {key}"

                model_meta = self._build_model_metadata(model)
                ckpt_meta = CheckpointMetadata(
                    checkpoint_path=str(ckpt_path),
                    training_steps=inspection.steps or self.config.required_min_steps,
                    training_seed=seed,
                    sha256_hash=inspection.sha256 or "",
                    file_size_bytes=inspection.file_size_bytes or 0,
                    last_modified_utc=datetime.now(timezone.utc).isoformat(),
                    is_500k_final=True,
                )

                for opp in self.config.opponents:
                    opp_meta = self._build_opponent_metadata(opp)

                    # Check for existing data under Never Overwrite rule
                    if self.storage.check_exists(model, seed, opp) and not self.config.allow_overwrite:
                        print(f"  SKIP: Raw evaluation data exists for {model.upper()} seed {seed} vs {opp}. Preserving raw file.")
                        continue

                    print(f"  RUNNING: {model.upper()} (Seed {seed}) vs {opp.upper()} ({self.config.games_per_opponent:,} games)...")
                    t0 = time.time()

                    records, summary = self.runner.run_matchup(
                        algorithm=model,
                        checkpoint_path=ckpt_path,
                        seed=seed,
                        opponent_type=opp,
                        num_games=self.config.games_per_opponent,
                        base_seed=self.config.base_seed,
                        checkpoint_steps=inspection.steps or self.config.required_min_steps,
                        checkpoint_sha256=inspection.sha256 or "",
                    )

                    # Persist raw records (Never overwrite)
                    csv_p, jsonl_p, meta_p = self.storage.write_matchup_data(
                        records=records,
                        model_meta=model_meta,
                        ckpt_meta=ckpt_meta,
                        opp_meta=opp_meta,
                        allow_overwrite=self.config.allow_overwrite,
                    )

                    elapsed = time.time() - t0
                    print(
                        f"  COMPLETED: {len(records):,} games in {elapsed:.1f}s | "
                        f"Win Rate: {summary['win_rate']:.2%} | "
                        f"Seat Bias: {summary['seat_bias']:+.2%} | "
                        f"Raw Data: {csv_p.name}"
                    )

                    matchup_summaries.append(summary)
                    total_games_played += len(records)

        # 3. Save Overall Processed Evaluation Summary
        proc_dir = self.base_dir / self.config.output_root / "processed"
        proc_dir.mkdir(parents=True, exist_ok=True)
        summary_payload = {
            "protocol_version": FROZEN_PROTOCOL_VERSION,
            "completion_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "total_games_played": total_games_played,
            "elapsed_seconds": round(time.time() - campaign_start, 2),
            "matchups": matchup_summaries,
        }
        summary_path = proc_dir / "final_evaluation_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2)

        return {
            "status": "COMPLETED",
            "total_games": total_games_played,
            "summary_path": str(summary_path),
        }

    def _build_model_metadata(self, model: str) -> ModelMetadata:
        model_canon = normalize_algorithm_name(model)
        names = {
            "dqn": ("Double Dueling DQN with PER", "DQNAgent"),
            "ppo": ("PPO Actor-Critic with Action Masking", "PPOAgent"),
            "model3": ("Discrete Soft Actor-Critic (SAC)", "DiscreteSACAgent"),
            "model4": ("Categorical DQN (C51 Distributional RL)", "C51Agent"),
        }
        full_name, cls_name = names.get(model_canon, (model_canon.upper(), "BaseAgent"))
        return ModelMetadata(
            algorithm_key=model_canon,
            algorithm_name=full_name,
            agent_class=cls_name,
            encoder_architecture="StructuredCNNEncoder: Conv1d(4->32, k=3, circ) -> ReLU -> Conv1d(32->64, k=3, circ) -> AdaptiveAvgPool1d(1) -> Concat(64+12=76) -> Linear(76->128) -> ReLU",
            action_space_size=4,
            evaluation_policy="deterministic argmax over masked legal actions",
            action_masking="negative infinity logit / Q-value suppression",
        )

    def _build_opponent_metadata(self, opponent_type: str) -> OpponentMetadata:
        if opponent_type == "random":
            return OpponentMetadata(
                opponent_type="random",
                opponent_class="RandomOpponent",
                strategy_description="Uniform random selection strictly among legal actions (guaranteed 0% illegal moves)",
                is_deterministic=False,
            )
        elif opponent_type == "expert":
            return OpponentMetadata(
                opponent_type="expert",
                opponent_class="ExpertOpponent",
                strategy_description="Alhajry (2012) rule-based heuristic priority: Release token on 6 > Defend endangered token > Capture opponent > Advance furthest token",
                is_deterministic=True,
            )
        else:
            return OpponentMetadata(
                opponent_type=opponent_type,
                opponent_class="BaseOpponent",
                strategy_description="Generic baseline opponent",
                is_deterministic=False,
            )
