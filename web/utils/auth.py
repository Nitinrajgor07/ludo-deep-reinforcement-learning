"""Local session utilities for PhD Ludo DRL Research Demonstration.

Zero Streamlit dependency. Local development is unauthenticated for seamless research inspection.
"""

from typing import Tuple


def get_configured_credentials() -> Tuple[str, str]:
    """Return local development role."""
    return ("researcher", "")


def is_authenticated() -> bool:
    """Always True for local research dashboard access."""
    return True


def login(input_user: str = "", input_pwd: str = "") -> bool:
    """No-op login helper for local research dashboard."""
    return True


def logout() -> None:
    """No-op logout helper for local research dashboard."""
    pass


def render_login_page() -> bool:
    """No-op login renderer."""
    return True
