"""Stochastic dice simulation with deterministic override support for testing."""

from typing import Optional
import numpy as np


class Dice:
    """Simulates a fair 6-sided die with reproducible seeding and test overrides."""

    def __init__(self, seed: Optional[int] = None, probabilities: Optional[np.ndarray] = None):
        self._rng = np.random.default_rng(seed)
        self._current_value: int = 1
        self._override_value: Optional[int] = None
        self._probabilities: Optional[np.ndarray] = None
        if probabilities is not None:
            self.set_distribution(probabilities)

    def set_distribution(self, probabilities: Optional[np.ndarray]) -> None:
        """Set or clear probability distribution over rolls [1..6]."""
        if probabilities is None:
            self._probabilities = None
            return
        p = np.asarray(probabilities, dtype=np.float64)
        if len(p) != 6:
            raise ValueError(f"Probabilities must have length 6, got {len(p)}")
        if np.any(p < 0):
            raise ValueError(f"Probabilities must be non-negative, got {p}")
        total = np.sum(p)
        if not np.isclose(total, 1.0, atol=1e-5):
            raise ValueError(f"Probabilities must sum to 1.0, got {total}")
        self._probabilities = p / total

    @property
    def probabilities(self) -> Optional[np.ndarray]:
        """Current probability distribution, or None if uniform fair die."""
        return self._probabilities

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

        if self._probabilities is not None:
            self._current_value = int(self._rng.choice([1, 2, 3, 4, 5, 6], p=self._probabilities))
        else:
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
