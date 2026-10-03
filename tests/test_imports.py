"""Basic project import tests ensuring package integrity and dependency resolution."""

import importlib
import pytest


def test_core_dependencies():
    """Verify that core PhD research dependencies are present."""
    for pkg in ["torch", "numpy", "matplotlib", "gymnasium", "yaml", "pytest"]:
        mod = importlib.import_module(pkg)
        assert mod is not None, f"Dependency {pkg} failed to import"


def test_package_imports():
    """Verify that all internal submodules import cleanly without cyclic dependencies."""
    modules = [
        "phd_ludo_drl",
        "phd_ludo_drl.config",
        "phd_ludo_drl.logs",
        "phd_ludo_drl.experiments",
        "phd_ludo_drl.environment",
        "phd_ludo_drl.encoding",
        "phd_ludo_drl.agents",
        "phd_ludo_drl.agents.dqn",
        "phd_ludo_drl.agents.ppo",
        "phd_ludo_drl.agents.model3",
        "phd_ludo_drl.agents.model4",
        "phd_ludo_drl.opponents",
        "phd_ludo_drl.training",
        "phd_ludo_drl.evaluation",
        "phd_ludo_drl.analysis",
        "phd_ludo_drl.visualization",
    ]
    for mod_name in modules:
        mod = importlib.import_module(mod_name)
        assert mod is not None, f"Submodule {mod_name} failed to import"


def test_base_interfaces():
    """Verify that core abstract base classes are defined and importable."""
    from phd_ludo_drl.agents.base_agent import BaseAgent
    from phd_ludo_drl.opponents.base_opponent import BaseOpponent

    assert BaseAgent is not None
    assert BaseOpponent is not None
