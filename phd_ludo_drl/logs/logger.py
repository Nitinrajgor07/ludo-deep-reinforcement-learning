"""Structured logging infrastructure for PhD research experiments."""

import logging
import os
from pathlib import Path
import sys
from typing import Optional


def setup_logger(
    name: str = "phd_ludo_drl",
    log_dir: Optional[Path] = None,
    level: int = logging.INFO,
    log_to_console: bool = True,
) -> logging.Logger:
    """Configure and return a structured logger with console and file handlers.

    Args:
        name: Logger identifier.
        log_dir: Directory where log file will be saved.
        level: Logging level.
        log_to_console: Whether to attach a console stream handler.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Avoid duplicate handlers on re-initialization
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path / f"{name}.log", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
