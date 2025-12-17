"""Helpers for extracting and normalising SQL output."""
from __future__ import annotations

import re


def extract_sql_query(response: str) -> str:
    """Extract the SQL query from a model response and remove formatting."""

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
    text = text.strip().replace("\n", " ")

    return text
