"""Tests for random seed management and reproducibility."""

import random
import numpy as np
import pytest
import torch
from phd_ludo_drl.config.seed import set_seed, get_derived_seed


def test_seed_determinism():
    """Verify that setting the same seed yields identical random sequences."""
    seed = 42

    # Run 1
    set_seed(seed)
    py_val_1 = random.random()
    np_val_1 = np.random.rand(5)
    th_val_1 = torch.rand(5)

    # Run 2
    set_seed(seed)
    py_val_2 = random.random()
    np_val_2 = np.random.rand(5)
    th_val_2 = torch.rand(5)

    assert py_val_1 == py_val_2
    assert np.allclose(np_val_1, np_val_2)
    assert torch.allclose(th_val_1, th_val_2)


def test_derived_seeds_are_distinct_and_deterministic():
    """Verify derived seeds for workers/parallel envs are unique and reproducible."""
    base_seed = 100
    derived_0 = get_derived_seed(base_seed, 0)
    derived_1 = get_derived_seed(base_seed, 1)
    derived_2 = get_derived_seed(base_seed, 2)

    assert derived_0 != derived_1 != derived_2
    assert derived_0 == get_derived_seed(base_seed, 0)


def test_invalid_seed_rejection():
    """Verify negative seed values are rejected."""
    with pytest.raises(ValueError):
        set_seed(-1)
