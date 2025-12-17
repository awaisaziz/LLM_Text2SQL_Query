"""Zero-shot prompt template for Text-to-SQL."""
from __future__ import annotations

from textwrap import dedent

ZERO_SHOT_TEMPLATE = dedent(
    """
    You are an expert SQL query developer.
    Given the following database schema:
    {schema}
    Write a correct SQL query to answer this question:
    Q: {question}
    Only output the SQL query.
    """
).strip()


def build_prompt(question: str, schema: str, db_id: str | None = None) -> str:
    """Return the zero-shot prompt for ``question`` and ``schema``."""

    del db_id  # db_id is unused for now, but kept for compatibility
    return ZERO_SHOT_TEMPLATE.format(question=question.strip(), schema=schema.strip())
