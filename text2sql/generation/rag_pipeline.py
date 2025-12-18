"""Retrieval and generation helpers for RAG-based Text-to-SQL."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, List, Sequence

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from text2sql.models.router import OpenAIChatLLM

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
    candidate_questions = [ex.question for ex in examples]
    example_embeddings = example_embeddings or embedder.encode(candidate_questions)
    query_embedding = embedder.encode([question])

    scores = cosine_similarity(query_embedding, example_embeddings)[0]
    top_indices = scores.argsort()[::-1][:k]
    retrieved = [examples[idx] for idx in top_indices]
    LOGGER.info("Retrieved top-%d similar examples", len(retrieved))
    return retrieved


def _generate_with_transformers(prompt: str, model: str, temperature: float, generator=None) -> tuple[str, Any]:
    from transformers import pipeline

    gen = generator or pipeline("text-generation", model=model, device_map="auto")
    outputs = gen(prompt, max_new_tokens=256, temperature=temperature, do_sample=temperature > 0)
    text = outputs[0]["generated_text"].replace(prompt, "", 1).strip()
    return text, gen


def generate_sql_candidates(
    prompt: str,
    n: int,
    provider: str,
    model: str,
    temperature: float,
) -> list[str]:
    """Generate ``n`` SQL candidates using the configured provider."""

    candidates: list[str] = []
    generator_cache = None
    router_client: OpenAIChatLLM | None = None
    for _ in range(n):
        if provider == "transformers":
            sql, generator_cache = _generate_with_transformers(prompt, model, temperature, generator_cache)
        else:
            router_client = router_client or OpenAIChatLLM(router=provider)
            sql = router_client.generate(prompt=prompt, model=model).sql

        candidates.append(sql)
    return candidates
