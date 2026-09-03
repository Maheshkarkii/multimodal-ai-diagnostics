"""
Structured logging configuration.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "field_engineer",
    level: str = "INFO",
    log_file: Optional[Path] = None
) -> logging.Logger:
    """
    Set up standardized console and optional file logging.

    Args:
        name: Logger namespace.
        level: Logging level string (e.g. 'DEBUG', 'INFO', 'WARNING').
        log_file: Optional path to output log file.

    Returns:
        Configured logging.Logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Prevent duplicate handlers if re-initialized
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
