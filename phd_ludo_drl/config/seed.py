"""Seed management and deterministic execution controls for reproducibility."""

import os
import random
from typing import Optional
import numpy as np
import torch


def set_seed(seed: int, deterministic_torch: bool = True) -> int:
    """Set random seed across Python random, NumPy, and PyTorch.

    Args:
        seed: The integer seed value.
        deterministic_torch: Whether to enforce deterministic operations in PyTorch.

    Returns:
        The seed integer that was set.
    """
    if not isinstance(seed, int) or seed < 0:
        raise ValueError(f"Seed must be a non-negative integer, got {seed}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic_torch:
        torch.use_deterministic_algorithms(False)  # keep False for ops unsupported on CPU if needed
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    return seed


def get_derived_seed(base_seed: int, index: int) -> int:
    """Derive an independent deterministic seed for parallel workers or environments.

    Args:
        base_seed: Primary experiment seed.
        index: Worker/environment index.

    Returns:
        Derived integer seed.
    """
    return (base_seed * 10007 + index * 1009) % (2**31 - 1)
