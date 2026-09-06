"""
Logging Utility with Rich Console Formatting and File Output Support.
"""

from pathlib import Path
from typing import Optional
import logging
import sys


def get_logger(
    name: str = "TextDetection",
    log_file: Optional[str] = None,
    log_level: int = logging.INFO,
) -> logging.Logger:
    """Initialize and retrieve a structured logger."""
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Check handlers
    if logger.handlers:
        if log_file:
            has_file = any(isinstance(h, logging.FileHandler) for h in logger.handlers)
            if not has_file:
                log_path = Path(log_file)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
                file_handler.setLevel(log_level)
                file_fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
                file_handler.setFormatter(file_fmt)
                logger.addHandler(file_handler)
        return logger

    # Try to use RichHandler for beautiful terminal output if available
    try:
        from rich.logging import RichHandler
        console_handler = RichHandler(
            rich_tracebacks=True,
            show_time=True,
            show_path=False,
            markup=True,
        )
    except ImportError:
        console_handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        console_handler.setFormatter(fmt)

    console_handler.setLevel(log_level)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
        file_handler.setLevel(log_level)
        file_fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(file_fmt)
        logger.addHandler(file_handler)

    return logger
