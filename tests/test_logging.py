"""Tests for structured logging infrastructure."""

import logging
from pathlib import Path
from phd_ludo_drl.logs.logger import setup_logger


def test_logger_setup(tmp_path: Path):
    """Verify logger creation, file writing, and message formatting."""
    log_name = "test_experiment"
    logger = setup_logger(name=log_name, log_dir=tmp_path, level=logging.DEBUG, log_to_console=False)

    assert logger.name == log_name
    assert logger.level == logging.DEBUG

    test_message = "Test message for PhD research logging verification"
    logger.info(test_message)

    log_file = tmp_path / f"{log_name}.log"
    assert log_file.exists()

    content = log_file.read_text(encoding="utf-8")
    assert test_message in content
    assert "[INFO]" in content
