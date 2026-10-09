"""Project versioning, Git provenance, and system environment capture."""

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Dict, Optional


@dataclass
class VersionProvenance:
    """Comprehensive provenance record capturing project, Git, and system state."""
    project_version: str
    git_commit: str
    git_commit_short: str
    git_branch: str
    git_dirty: bool
    git_commit_date: str
    python_version: str
    platform_system: str
    platform_release: str
    platform_machine: str
    processor: str
    torch_version: str
    cuda_available: bool
    gymnasium_version: str
    numpy_version: str
    captured_at_utc: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def get_git_info(repo_root: Optional[Path] = None) -> Dict[str, Any]:
    """Extract Git repository status and commit information.
    
    Returns fallback dictionary if Git is unavailable or directory is not a repository.
    """
    root = repo_root or Path.cwd()
    git_info: Dict[str, Any] = {
        "commit": "unknown",
        "commit_short": "unknown",
        "branch": "unknown",
        "dirty": False,
        "commit_date": "unknown",
        "commit_message": "unknown",
    }

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            stderr=subprocess.DEVNULL,
        ).decode("utf-8").strip()
        git_info["commit"] = commit
        git_info["commit_short"] = commit[:8] if len(commit) >= 8 else commit
    except Exception:
        pass

    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(root),
            stderr=subprocess.DEVNULL,
        ).decode("utf-8").strip()
        git_info["branch"] = branch
    except Exception:
        pass

    try:
        status_output = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(root),
            stderr=subprocess.DEVNULL,
        ).decode("utf-8").strip()
        git_info["dirty"] = len(status_output) > 0
    except Exception:
        pass

    try:
        commit_date = subprocess.check_output(
            ["git", "log", "-1", "--format=%cI"],
            cwd=str(root),
            stderr=subprocess.DEVNULL,
        ).decode("utf-8").strip()
        git_info["commit_date"] = commit_date
    except Exception:
        pass

    try:
        commit_msg = subprocess.check_output(
            ["git", "log", "-1", "--format=%s"],
            cwd=str(root),
            stderr=subprocess.DEVNULL,
        ).decode("utf-8").strip()
        git_info["commit_message"] = commit_msg
    except Exception:
        pass

    return git_info


def get_project_version(repo_root: Optional[Path] = None) -> str:
    """Read project version from pyproject.toml or return fallback."""
    root = repo_root or Path.cwd()
    pyproject_path = root / "pyproject.toml"
    if pyproject_path.exists():
        try:
            content = pyproject_path.read_text(encoding="utf-8")
            for line in content.splitlines():
                if line.strip().startswith("version ="):
                    return line.split("=")[1].strip().strip('"').strip("'")
        except Exception:
            pass
    return "1.0.0"


def capture_version_provenance(repo_root: Optional[Path] = None) -> VersionProvenance:
    """Capture a complete snapshot of version, git status, and hardware/software environment."""
    root = repo_root or Path.cwd()
    git_info = get_git_info(root)
    proj_version = get_project_version(root)

    # Dependency versions
    try:
        import torch
        torch_ver = torch.__version__
        cuda_avail = bool(torch.cuda.is_available())
    except ImportError:
        torch_ver = "unavailable"
        cuda_avail = False

    try:
        import gymnasium
        gym_ver = gymnasium.__version__
    except ImportError:
        gym_ver = "unavailable"

    try:
        import numpy as np
        np_ver = np.__version__
    except ImportError:
        np_ver = "unavailable"

    now_utc = datetime.now(timezone.utc).isoformat()

    return VersionProvenance(
        project_version=proj_version,
        git_commit=git_info["commit"],
        git_commit_short=git_info["commit_short"],
        git_branch=git_info["branch"],
        git_dirty=git_info["dirty"],
        git_commit_date=git_info["commit_date"],
        python_version=sys.version.split()[0],
        platform_system=platform.system(),
        platform_release=platform.release(),
        platform_machine=platform.machine(),
        processor=platform.processor(),
        torch_version=torch_ver,
        cuda_available=cuda_avail,
        gymnasium_version=gym_ver,
        numpy_version=np_ver,
        captured_at_utc=now_utc,
    )
