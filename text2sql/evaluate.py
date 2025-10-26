"""Wrapper around the official Spider evaluation script."""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate predictions using Spider metrics")
    parser.add_argument("predictions", type=Path, help="Path to predictions JSONL file")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config.json",
        help="Configuration JSON with Spider paths",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    return json.loads(config_path.read_text())


def predictions_to_sql_file(predictions_path: Path, output_path: Path) -> None:
    """Extract ``pred_sql`` from a JSONL file and write a flat SQL file."""

    with predictions_path.open("r", encoding="utf-8") as src, output_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            if not line.strip():
                continue
            record = json.loads(line)
            dst.write((record.get("pred_sql") or "").strip() + "\n")


def run_spider_evaluation(gold_sql_path: Path, pred_sql_path: Path, database_dir: Path) -> Tuple[str, str]:
    """Execute the official Spider evaluation script."""

    cmd = [
        "python",
        str(gold_sql_path.parent / "evaluate.py"),
        "--gold", str(gold_sql_path),
        "--pred", str(pred_sql_path),
        "--db", str(database_dir),
    ]
    LOGGER.info("Running Spider evaluation: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout, result.stderr


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    config = load_config(args.config)
    spider_root = Path(config["spider_path"])
    gold_sql_path = spider_root / config["gold_sql_filename"]
    database_dir = spider_root / config["database_dir"]

    with tempfile.TemporaryDirectory() as tmpdir:
        pred_sql_path = Path(tmpdir) / "predicted.sql"
        predictions_to_sql_file(args.predictions, pred_sql_path)

        stdout, stderr = run_spider_evaluation(gold_sql_path, pred_sql_path, database_dir)
        print(stdout)
        if stderr:
            LOGGER.warning(stderr)


if __name__ == "__main__":  # pragma: no cover
    main()
