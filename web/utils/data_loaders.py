"""Data loading utilities for Ludo DRL demonstration web UI.

Clean pure-Python implementation without Streamlit dependency.
Delegates to web.services.data_service.DataService.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from web.services.data_service import (
    CheckpointInfo,
    data_service,
    get_project_root,
)

__all__ = [
    "get_project_root",
    "scan_available_checkpoints",
    "load_pilot_summary",
    "load_table_csv",
    "load_robustness_summary",
    "load_experiment_list",
    "load_experiments_table",
    "CheckpointInfo",
]


def scan_available_checkpoints() -> List[CheckpointInfo]:
    """Scan the repository for all valid .pt model checkpoints."""
    return data_service.scan_checkpoints()


def load_pilot_summary() -> Optional[pd.DataFrame]:
    """Load pilot experiment summary metrics table if available."""
    df = data_service.load_table_dataframe("pilot_summary")
    return df if not df.empty else None


def load_table_csv(table_name: str) -> Optional[pd.DataFrame]:
    """Load arbitrary CSV table from results/tables or docs/tables."""
    df = data_service.load_table_dataframe(table_name)
    return df if not df.empty else None


def load_robustness_summary() -> Optional[pd.DataFrame]:
    """Load Phase 12 robustness evaluation summary."""
    df = data_service.load_table_dataframe("robustness_summary")
    return df if not df.empty else None


def load_experiment_list() -> List[str]:
    """Return sorted list of valid experiment subdirectories."""
    root = get_project_root()
    exp_dir = root / "experiments"
    if not exp_dir.exists():
        return []
    return sorted([d.name for d in exp_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])


def load_experiments_table() -> pd.DataFrame:
    """Build unified summary DataFrame across all experiments."""
    return data_service.load_table_dataframe("pilot_summary")
