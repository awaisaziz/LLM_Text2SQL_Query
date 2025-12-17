"""Logging helpers for the Text2SQL pipeline."""
from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(output_root: Path) -> Path:
    """Configure logging to file and console.

    The log file is stored under ``output_root / "log" / "run.log"``. The parent
    directories are created if they do not exist.
    """

    log_dir = output_root / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "run.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s:%(name)s:%(message)s",
        filename=log_file,
        filemode="w",
    )

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    logging.getLogger().addHandler(console)

    logging.info("Logging initialised")
    return log_file
