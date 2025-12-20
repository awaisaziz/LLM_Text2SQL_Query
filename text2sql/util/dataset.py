"""Utility helpers for loading and working with the Spider dataset."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

LOGGER = logging.getLogger(__name__)


@dataclass
class SpiderExample:
    """Container for a single dataset example."""

    question: str
    gold_sql: str
    db_id: str


class SpiderDataset:
    """Reader for Text-to-SQL development sets (Spider/BIRD)."""

    def __init__(
        self,
        root: os.PathLike[str] | str,
        dev_filename: str = "dev.json",
        tables_filename: str = "tables.json",
        sql_field: str = "query",
        dataset_name: str = "spider",
    ) -> None:
        self.root = Path(root)
        self.dataset_name = dataset_name
        self.dev_path = self.root / dev_filename
        self.tables_path = self.root / tables_filename

        if not self.dev_path.exists():
            raise FileNotFoundError(f"Could not find Spider dev file: {self.dev_path}")
        if not self.tables_path.exists():
            raise FileNotFoundError(
                f"Could not find Spider schema file: {self.tables_path}"
            )

        LOGGER.debug("Loading Spider dev set from %s", self.dev_path)
        raw_examples = json.loads(self.dev_path.read_text())
        self._examples: List[SpiderExample] = []
        for item in raw_examples:
            sql_value = item.get(sql_field) or item.get("query") or item.get("SQL")
            if sql_value is None:
                raise KeyError(f"Could not find SQL field '{sql_field}' (or fallback) in dev file.")
            self._examples.append(
                SpiderExample(
                    question=item["question"],
                    gold_sql=sql_value,
                    db_id=item["db_id"],
                )
            )
        LOGGER.debug("Loaded %d %s examples", len(self._examples), dataset_name.upper())

        LOGGER.debug("Loading schema metadata from %s", self.tables_path)
        self._schemas: Dict[str, dict] = {
            item["db_id"]: item for item in json.loads(self.tables_path.read_text())
        }

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._examples)

    def __iter__(self) -> Iterable[SpiderExample]:  # pragma: no cover - trivial
        yield from self._examples

    def get(self, index: int) -> SpiderExample:
        return self._examples[index]

    def iter_examples(self, limit: Optional[int] = None) -> Iterable[SpiderExample]:
        """Iterate over Spider examples with an optional limit."""

        if limit is None:
            yield from self._examples
            return

        for example in self._examples[:limit]:
            yield example

    # ------------------------------------------------------------------
    # Schema helpers
    # ------------------------------------------------------------------
    def get_schema(self, db_id: str) -> str:
        """Return a human-readable schema description for ``db_id``."""

        schema = self._schemas.get(db_id)
        if schema is None:
            raise KeyError(f"Unknown Spider database id: {db_id}")

        lines: List[str] = []
        for table_name, column_names in self._iter_tables(schema):
            friendly_columns = ", ".join(column_names)
            lines.append(f"Table: {table_name}({friendly_columns})")

        schema_str = "\n".join(lines)
        LOGGER.debug("Schema for %s:\n%s", db_id, schema_str)
        return schema_str

    @staticmethod
    def _iter_tables(schema: Dict[str, object]) -> Iterable[tuple[str, List[str]]]:
        tables = schema.get("table_names_original", [])
        columns: List[List[object]] = schema.get("column_names_original", [])

        table_to_columns: Dict[int, List[str]] = {i: [] for i in range(len(tables))}
        for table_idx, column_name in columns:
            if table_idx == -1:
                # Skip pseudo column for *
                continue
            table_to_columns.setdefault(table_idx, []).append(column_name)

        for idx, table_name in enumerate(tables):
            yield table_name, table_to_columns.get(idx, [])


def load_dataset(
    dataset_path: os.PathLike[str] | str,
    dev_filename: str = "dev.json",
    tables_filename: str = "tables.json",
    sql_field: str = "query",
    dataset_name: str = "spider",
) -> SpiderDataset:
    """Instantiate :class:`SpiderDataset` for the given dataset directory."""

    return SpiderDataset(
        dataset_path,
        dev_filename=dev_filename,
        tables_filename=tables_filename,
        sql_field=sql_field,
        dataset_name=dataset_name,
    )
