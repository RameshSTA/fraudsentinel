from __future__ import annotations

import logging
from typing import Optional


def setup_logger(name: str = "scam_pipeline", level: int = logging.INFO) -> logging.Logger:
    """
    Create a console logger with a consistent, professional format.

    Parameters
    ----------
    name:
        Logger name.
    level:
        Logging level, e.g. logging.INFO.

    Returns
    -------
    logging.Logger
        Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.propagate = False
    return logger