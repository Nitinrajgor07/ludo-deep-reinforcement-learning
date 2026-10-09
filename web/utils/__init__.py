"""Utility functions for web demonstration."""

from web.utils.data_loaders import (
    get_project_root,
    scan_available_checkpoints,
    load_pilot_summary,
    load_table_csv,
    load_robustness_summary,
    load_experiment_list,
    load_experiments_table,
    CheckpointInfo,
)
from web.utils.auth import (
    is_authenticated,
    login,
    logout,
    render_login_page,
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
    "is_authenticated",
    "login",
    "logout",
    "render_login_page",
]
