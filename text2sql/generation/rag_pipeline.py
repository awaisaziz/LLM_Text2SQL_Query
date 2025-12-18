"""Retrieval and generation helpers for RAG-based Text-to-SQL."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Sequence

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from text2sql.models.router import OpenAIChatLLM, Prompt

LOGGER = logging.getLogger(__name__)


@dataclass
class Example:
    question: str
    sql: str
    db_id: str


def format_schema(db_id: str, tables_metadata: dict[str, Any]) -> str:
    """Return a human-readable schema description for ``db_id``."""

    schema = tables_metadata.get(db_id)
    if schema is None:
        raise KeyError(f"Schema for db_id '{db_id}' not found in tables.json")

    lines: List[str] = []
    for table_idx, table_name in enumerate(schema.get("table_names_original", [])):
        columns = [col_name for idx, col_name in schema.get("column_names_original", []) if idx == table_idx]
        lines.append(f"Table: {table_name}({', '.join(columns)})")
    return "\n".join(lines)


def retrieve_similar_examples(
    question: str,
    examples: Sequence[Example],
    k: int,
    embedding_model_name: str,
    embedder: SentenceTransformer | None = None,
    example_embeddings: list | None = None,
) -> list[Example]:
    """Retrieve the top-k similar examples based on cosine similarity."""

    embedder = embedder or SentenceTransformer(embedding_model_name)
    example_embeddings = example_embeddings or embedder.encode([ex.question for ex in examples])
    query_embedding = embedder.encode([question])

    scores = cosine_similarity(query_embedding, example_embeddings)[0]
    top_indices = scores.argsort()[::-1][:k]
    retrieved = [examples[idx] for idx in top_indices]
    LOGGER.info("Retrieved top-%d similar examples for question: %s", len(retrieved), question)
    return retrieved


def generate_sql_candidates(
    prompt: Prompt,
    n: int,
    provider: str,
    model: str,
) -> list[str]:
    """Generate ``n`` SQL candidates using the configured provider."""

    candidates: list[str] = []
    router_client: OpenAIChatLLM | None = None
    for _ in range(n):
        router_client = router_client or OpenAIChatLLM(router=provider)
        sql = router_client.generate(prompt=prompt, model=model).sql

        candidates.append(sql)
    return candidates
