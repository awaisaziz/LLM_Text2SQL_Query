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
    
    system_prompt = """You are an expert SQL query generator. You generate multiple semantically correct SQL query variations for the same question.

    IMPORTANT RULES YOU MUST FOLLOW:
    1. Think step-by-step: Analyze the question, schema, and examples
    2. ALWAYS think step-by-step before writing SQL
    3. SQL query must use a DISTINCT approach (different tables, joins, subqueries, aggregation methods)
    4. Only use "AS" for table aliases when joining tables or when column names would be ambiguous
    - ALLOWED: "SELECT T2.Year ,  T1.Official_Name FROM city AS T1 JOIN farm_competition AS T2 ON T1.City_ID  =  T2.Host_city_ID"
    - ALLOWED: "SELECT avg(T1.product_price) FROM Products AS T1 JOIN Order_items AS T2 ON T1.product_id  =  T2.product_id"
    - PROHIBITED: "SELECT avg(product_price) AS average_price FROM Products"
    - PROHIBITED: "SELECT count(*) AS club_count FROM club"
    5. NEVER add column aliases - preserve original column names from the schema
    6. Study the provided similar examples to understand patterns
    7. Generate ONLY the SQL query - no explanations, no markdown, no additional text
    
    Output format: A single SQL query with proper formatting.
    """.strip()


    examples_block = "\n\n".join(
        f"Question: {ex.question}\nSQL: {ex.sql}" for ex in retrieved_examples
    )
    
    user_prompt = f"""
    DATABASE SCHEMA:
    {schema}

    SIMILAR EXAMPLES:
    {examples_block}

    QUESTION: {user_question}

    THINKING PROCESS (REQUIRED):
    1. What is the question asking for? Identify required tables and columns.
    2. Which similar examples show relevant patterns?
    3. What JOINs, WHERE conditions, or aggregations are needed?
    4. How should the query be structured based on the schema?
    5. Double-check that all column/table names match the schema exactly.
    6. Follow chain-of-thought reasoning before writing the final SQL.

    FINAL SQL QUERY (ONLY THE SQL):""".strip()

    return {"system": system_prompt, "user": user_prompt}
