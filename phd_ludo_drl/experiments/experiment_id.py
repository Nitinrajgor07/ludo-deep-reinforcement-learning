"""Experiment identification, metadata hashing, and directory management."""

from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class ExperimentMetadata:
    experiment_id: str
    algorithm: str
    encoding: str
    seed: int
    timestamp: str
    description: str
    config_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ExperimentIdentifier:
    """Generates standardized experiment IDs and directory structures."""

    @staticmethod
    def generate_hash(config_dict: Dict[str, Any]) -> str:
        """Create a deterministic SHA-256 short hash of the experiment configuration."""
        serialized = json.dumps(config_dict, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:8]

    @classmethod
    def create(
        cls,
        algorithm: str,
        seed: int,
        encoding: str = "structured_cnn",
        description: str = "",
        config_dict: Optional[Dict[str, Any]] = None,
    ) -> ExperimentMetadata:
        """Create a new standardized experiment metadata record."""
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_hash = cls.generate_hash(config_dict) if config_dict else "default"
        exp_id = f"EXP_{algorithm.upper()}_{encoding}_{seed}_{now_str}"

        return ExperimentMetadata(
            experiment_id=exp_id,
            algorithm=algorithm.lower(),
            encoding=encoding.lower(),
            seed=seed,
            timestamp=now_str,
            description=description,
            config_hash=config_hash,
        )

    @staticmethod
    def resolve_paths(base_dir: Path, exp_id: str) -> Dict[str, Path]:
        """Resolve and create standardized artifact paths for a given experiment ID."""
        exp_root = base_dir / "experiments" / exp_id
        checkpoint_dir = exp_root / "checkpoints"
        log_dir = exp_root / "logs"
        results_dir = exp_root / "results"

        for p in [checkpoint_dir, log_dir, results_dir]:
            p.mkdir(parents=True, exist_ok=True)

        return {
            "root": exp_root,
            "checkpoints": checkpoint_dir,
            "logs": log_dir,
            "results": results_dir,
        }
