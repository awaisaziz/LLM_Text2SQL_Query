"""Utility helpers for loading and working with the Spider dataset."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import re

LOGGER = logging.getLogger(__name__)


@dataclass
class SpiderExample:
    """Container for a single Spider example."""

    question: str
    gold_sql: str
    db_id: str


class SpiderDataset:
    """Reader for the Spider development set.

    Parameters
    ----------
    root : str or Path
        Path to the Spider dataset directory containing ``dev.json`` and
        ``tables.json``.
    dev_filename : str
        Name of the JSON file with Spider examples.
    tables_filename : str
        Name of the JSON file describing schema metadata.
    """

    def __init__(
        self,
        root: os.PathLike[str] | str,
        dev_filename: str = "dev.json",
        tables_filename: str = "tables.json",
    ) -> None:
        self.root = Path(root)
        self.dev_path = self.root / dev_filename
        self.tables_path = self.root / tables_filename

        if not self.dev_path.exists():
            raise FileNotFoundError(f"Could not find Spider dev file: {self.dev_path}")
        if not self.tables_path.exists():
            raise FileNotFoundError(
                f"Could not find Spider schema file: {self.tables_path}"
            )

        LOGGER.debug("Loading Spider dev set from %s", self.dev_path)
        self._examples: List[SpiderExample] = [
            SpiderExample(
                question=item["question"],
                gold_sql=item["query"],
                db_id=item["db_id"],
            )
            for item in json.loads(self.dev_path.read_text())
        ]
        LOGGER.debug("Loaded %d Spider examples", len(self._examples))

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


def load_dataset_from_config(config_path: os.PathLike[str] | str) -> SpiderDataset:
    """Instantiate :class:`SpiderDataset` from a ``config.json`` file."""

    config_path = Path(config_path)
    config = json.loads(config_path.read_text())
    spider_path = config.get("spider_path", "spider_data")
    dev_filename = config.get("dev_filename", "dev.json")
    tables_filename = config.get("tables_filename", "tables.json")

    return SpiderDataset(spider_path, dev_filename=dev_filename, tables_filename=tables_filename)

# ------------------------------------------------------------------
# Extract the SQL from the query and clean the response
# ------------------------------------------------------------------
def extract_sql_query(response: str) -> str:
    """
    Extracts the SQL query from a model response and removes any markdown/code fences or prefixes.
    Works with formats like:
    - ```sql\nSELECT * FROM ...\n```
    - SQL Query: SELECT * FROM ...
    - ``` SELECT * FROM ... ```
    - plain SELECT * FROM ...
    """
    if not response:
        return ""

    text = response.strip()

    # 1. Remove ```sql or ``` fences if they exist
    text = re.sub(r"^```(?:sql)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    # 2. Remove common prefixes like "SQL Query:" or "The SQL is:"
    text = re.sub(r"(?i)^sql\s*query:\s*", "", text)
    text = re.sub(r"(?i)^the\s*sql\s*(query|statement)\s*(is)?:\s*", "", text)

    # 3. Sometimes LLMs return explanation + query. Extract first SELECT or WITH onwards.
    match = re.search(r"(?i)(SELECT|WITH)\s", text)
    if match:
        text = text[match.start():]

    # 4. Strip trailing spaces, newlines, or semicolons (keep one if needed)
    text = text.strip()

    return text

