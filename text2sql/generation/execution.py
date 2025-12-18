"""SQL execution and voting utilities for the Text-to-SQL pipeline."""
from __future__ import annotations

import json
import logging
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

LOGGER = logging.getLogger(__name__)


@dataclass
class CandidateSQL:
    sql: str
    execution_result: list[Any] | None
    error: str | None


def resolve_db_path(db_root: Path, db_id: str) -> Path:
    """Locate the SQLite database path for ``db_id`` under ``db_root``."""

    candidate_paths = [db_root / db_id / f"{db_id}.sqlite", db_root / db_id / f"{db_id}.db"]
    for path in candidate_paths:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not locate SQLite database for {db_id} under {db_root}")


def execute_sql(sql: str, db_path: Path) -> CandidateSQL:
    """Execute SQL against the specified SQLite database and capture results."""

    cleaned_sql = _first_statement(sql)
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(cleaned_sql)
            rows = cursor.fetchall()
            result = [tuple(row) for row in rows]
            return CandidateSQL(sql=cleaned_sql, execution_result=result, error=None)
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("Execution failed for candidate: %s", exc)
        return CandidateSQL(sql=cleaned_sql or sql, execution_result=None, error=str(exc))


def majority_vote(candidates: Iterable[CandidateSQL]) -> tuple[str, list[CandidateSQL]]:
    """Select the SQL query with the highest execution-based vote."""

    candidate_list = list(candidates)
    if not candidate_list:
        return "", candidate_list

    vote_counts: dict[str, int] = {}
    best_sql = candidate_list[0].sql
    best_count = 0

    for candidate in candidate_list:
        key = (
            json.dumps(candidate.execution_result, sort_keys=True, default=str)
            if candidate.error is None
            else f"error:{candidate.error}"
        )
        vote_counts[key] = vote_counts.get(key, 0) + 1
        if vote_counts[key] > best_count:
            best_sql = candidate.sql
            best_count = vote_counts[key]

    return best_sql, candidate_list


def _first_statement(sql: str) -> str:
    """Return only the first SQL statement to avoid multi-query execution errors."""

    stripped = sql.strip()
    if not stripped:
        return ""

    parts = [part.strip() for part in re.split(r";\s*|\n\s*\n+", stripped) if part.strip()]
    return parts[0] if parts else ""
