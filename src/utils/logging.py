"""Run logging: console + per-run log.txt."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"


def get_run_logger(name: str, log_file: Path | None = None) -> logging.Logger:
    """Logger writing to stderr and, if given, to ``log_file``."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    # Reset handlers so repeated calls (tests, batches) do not duplicate lines.
    for h in list(logger.handlers):
        h.close()
        logger.removeHandler(h)
    fmt = logging.Formatter(_FORMAT)
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def close_run_logger(logger: logging.Logger) -> None:
    for h in list(logger.handlers):
        h.close()
        logger.removeHandler(h)
