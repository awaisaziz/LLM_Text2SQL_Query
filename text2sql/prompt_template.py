"""Prompt templates for Text-to-SQL models."""
from __future__ import annotations

from textwrap import dedent


ZERO_SHOT_TEMPLATE = dedent(
    """
    You are an expert SQL developer.
    Given the following database schema:
    {schema}
    Write a correct SQL query to answer this question:
    Q: {question}
    Only output the SQL query.
    """
).strip()


def build_prompt(question: str, schema: str, db_id: str | None = None) -> str:
    """Return the zero-shot prompt for ``question`` and ``schema``.

    The ``db_id`` is included to make it easy to extend the prompt in the
    future, but it is not currently used in the template.
    """

    del db_id  # db_id is unused for now, but kept for compatibility
    return ZERO_SHOT_TEMPLATE.format(question=question.strip(), schema=schema.strip())
