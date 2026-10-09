"""Game controller managing environment interaction and AI model inference.

Clean pure-Python implementation without any Streamlit dependency.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from phd_ludo_drl.environment.ludo_env import LudoEnvironment
from phd_ludo_drl.training.factory import create_agent, create_opponent
from phd_ludo_drl.agents.base_agent import BaseAgent
from phd_ludo_drl.opponents.base_opponent import BaseOpponent
from web.services.game_service import GameSession, get_cached_agent

__all__ = ["GameSession", "load_cached_agent"]


def load_cached_agent(algorithm: str, checkpoint_path_str: Optional[str] = None) -> BaseAgent:
    """Retrieve or instantiate a cached agent without Streamlit."""
    return get_cached_agent(algorithm, checkpoint_path_str)
