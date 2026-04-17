from __future__ import annotations

import logging
import sys
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler

_console = Console(stderr=True)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = RichHandler(
            console=_console,
            show_time=True,
            show_path=False,
            rich_tracebacks=True,
        )
        handler.setLevel(level)
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


def set_global_level(level: int) -> None:
    """Adjust level on all qstatic_diag loggers."""
    for name, logger in logging.Logger.manager.loggerDict.items():
        if name.startswith("qstatic_diag") and isinstance(logger, logging.Logger):
            logger.setLevel(level)
