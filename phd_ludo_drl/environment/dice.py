"""Stochastic dice simulation with deterministic override support for testing."""

from typing import Optional
import numpy as np


class Dice:
    """Simulates a fair 6-sided die with reproducible seeding and test overrides."""

    def __init__(self, seed: Optional[int] = None):
        self._rng = np.random.default_rng(seed)
        self._current_value: int = 1
        self._override_value: Optional[int] = None

    def seed(self, seed: Optional[int] = None) -> None:
        """Seed or reseed the random number generator."""
        self._rng = np.random.default_rng(seed)

    def roll(self) -> int:
        """Roll the die and return an integer in [1, 6]."""
        if self._override_value is not None:
            val = self._override_value
            self._override_value = None
            self._current_value = val
            return val

        self._current_value = int(self._rng.integers(1, 7))
        return self._current_value

    @property
    def value(self) -> int:
        """Current dice value."""
        return self._current_value

    def set_override(self, value: int) -> None:
        """Set a one-time deterministic value for the next roll (for tests)."""
        if not (1 <= value <= 6):
            raise ValueError(f"Dice value must be between 1 and 6, got {value}")
        self._override_value = value

    def clear_override(self) -> None:
        """Clear any pending roll override."""
        self._override_value = None
