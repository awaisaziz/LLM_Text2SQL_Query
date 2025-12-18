"""SQL execution and voting utilities for the Text-to-SQL pipeline."""
from __future__ import annotations

import json
import logging
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

_STATEMENT_SPLIT_PATTERN = re.compile(
    r";\s*|\n+(?=\s*(?:SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|WITH|PRAGMA)\b)", re.IGNORECASE
)


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


def majority_vote_sql(sql_candidates: list[str], db_path: str | Path) -> tuple[str, list[CandidateSQL]]:
    """
    Execute SQL candidates and select the query with the strongest execution agreement.

    Returns the best SQL (fallback to the first candidate on ties or all failures)
    and metadata for each candidate execution.
    """

    if not sql_candidates:
        return "", []

    database_path = Path(db_path)
    executed_candidates: list[CandidateSQL] = [execute_sql(sql, database_path) for sql in sql_candidates]

    all_failed = all(candidate.execution_result is None for candidate in executed_candidates)
    if all_failed:
        return normalize_sql_query(executed_candidates[0].sql), executed_candidates

    vote_counts: dict[str, int] = {}
    best_sql = executed_candidates[0].sql
    best_count = 0

    for candidate in executed_candidates:
        key = (
            json.dumps(candidate.execution_result, sort_keys=True, default=str)
            if candidate.execution_result is not None
            else f"error:{candidate.error or 'unknown'}"
        )
        count = vote_counts.get(key, 0) + 1
        vote_counts[key] = count
        if count > best_count:
            best_sql = candidate.sql
            best_count = count

    return normalize_sql_query(best_sql), executed_candidates


def _first_statement(sql: str) -> str:
    """Return only the first SQL statement to avoid multi-query execution errors."""

    stripped = sql.strip()
    if not stripped:
        return ""

    first_statement = _STATEMENT_SPLIT_PATTERN.split(stripped, maxsplit=1)[0].strip()
    return first_statement


def normalize_sql_query(sql: str) -> str:
    """Normalize SQL by returning only the first statement and trimming whitespace/semicolons."""

    normalized = _first_statement(sql).strip()
    return normalized[:-1].strip() if normalized.endswith(";") else normalized
