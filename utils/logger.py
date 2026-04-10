"""Logging setup for training runs."""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "rl-panda-grasp",
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure and return a logger with console and optional file output.

    Args:
        name: Logger name.
        log_dir: Directory for log files. If None, only console output.
        level: Logging level.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path / "training.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
