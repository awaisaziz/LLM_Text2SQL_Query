"""Prompt construction utilities for Text-to-SQL generation."""
from __future__ import annotations

from textwrap import dedent
from typing import Iterable, Mapping, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from text2sql.generation.rag_pipeline import Example

ZERO_SHOT_TEMPLATE = dedent(
    """
    Given the following database schema:
    {schema}
    Write a correct SQL query to answer this question:
    Q: {question}
    """
).strip()


def build_zero_shot_prompt(question: str, schema: str, db_id: str | None = None) -> Mapping[str, str]:
    """Return the zero-shot prompt for ``question`` and ``schema`` as chat messages."""

    del db_id  # db_id is unused for now, but kept for compatibility
    system_prompt = (
        "You are an expert Text-to-SQL system that maps natural language questions to SQL queries. "
        "Use the database schema to craft a correct SQL answer. Return only the SQL query."
    )

    user_prompt = f"""
    Given the following database schema:
    {schema}
    Write a correct SQL query to answer this question:
    Q: {question}
    """.strip()

    return {"system": system_prompt, "user": user_prompt}


def build_cot_prompt(
    schema: str,
    user_question: str,
    retrieved_examples: Iterable["Example"],
    mode: str = "cot",
) -> Mapping[str, str]:
    """Construct a chain-of-thought prompt using retrieved examples.

    Returns a mapping with ``system`` and ``user`` keys to allow callers to
    pass structured chat prompts to LLM providers.
    """

    examples_block = "\n\n".join(
        f"Question: {ex.question}\nSQL: {ex.sql}" for ex in retrieved_examples
    )
    reasoning_prefix = (
        "Follow chain-of-thought reasoning before writing the final SQL."
        if mode.lower() == "cot"
        else ""
    )

    system_prompt = (
        "You are an expert Text-to-SQL system that maps natural language questions to SQL queries. "
        "Use the database schema and similar examples to craft the answer. "
        "Think step-by-step and generate SQL. Return only the SQL query."
    )

    user_prompt = (
        f"Database schema:\n{schema}\n\n"
        f"Retrieved examples:\n{examples_block}\n\n"
        f"User question: {user_question}\n"
        f"{reasoning_prefix}\n"
    ).strip()

    return {"system": system_prompt, "user": user_prompt}
