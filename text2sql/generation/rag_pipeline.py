"""Retrieval-augmented Text-to-SQL inference pipeline.

This module implements a self-consistent Text-to-SQL workflow that combines
retrieval, in-context prompting, and execution-based majority voting. It is
designed for inference only and uses open-source components for embeddings and
SQL generation.
"""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from text2sql.config.rag_config import DEFAULT_CONFIG_PATH, RAGConfig, load_rag_config

LOGGER = logging.getLogger(__name__)


@dataclass
class Example:
    question: str
    sql: str
    db_id: str


@dataclass
class CandidateSQL:
    sql: str
    execution_result: list[Any] | None
    error: str | None


def load_examples(dataset_path: Path, num_retrieve: int, filename: str = "dev.json") -> list[Example]:
    """Load Text-to-SQL examples for retrieval from a dataset file.

    Defaults to ``dev.json`` so the pipeline can iterate over development
    questions and use the same split for retrieval unless overridden.
    """

    data_path = dataset_path / filename
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find {filename} at {data_path}")

    raw_items = json.loads(data_path.read_text())[:num_retrieve]
    examples: list[Example] = []
    for item in raw_items:
        sql_value = item.get("sql") or item.get("query") or ""
        examples.append(Example(question=item["question"], sql=sql_value, db_id=item["db_id"]))
    LOGGER.debug("Loaded %d retrieval examples from %s", len(examples), filename)
    return examples


def _load_tables_metadata(dataset_path: Path) -> Dict[str, Any]:
    tables_path = dataset_path / "tables.json"
    if not tables_path.exists():
        raise FileNotFoundError(f"Could not find tables.json at {tables_path}")
    return {item["db_id"]: item for item in json.loads(tables_path.read_text())}


def _format_schema(db_id: str, tables_metadata: Dict[str, Any]) -> str:
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


def build_prompt(
    schema: str,
    user_question: str,
    retrieved_examples: Iterable[Example],
    prompt_technique: str = "cot",
) -> str:
    """Construct the in-context prompt for SQL generation."""

    examples_block = "\n\n".join(
        f"Question: {ex.question}\nSQL: {ex.sql}" for ex in retrieved_examples
    )
    reasoning_prefix = (
        "Follow chain-of-thought reasoning before writing the final SQL." if prompt_technique.lower() == "cot" else ""
    )

    prompt = (
        "You are an expert Text-to-SQL system that maps natural language questions to SQL queries. "
        "Use the database schema and similar examples to craft the answer.\n\n"
        f"Database schema:\n{schema}\n\n"
        f"Retrieved examples:\n{examples_block}\n\n"
        f"User question: {user_question}\n"
        f"{reasoning_prefix}\n"
        "Think step-by-step and generate SQL. Return only the SQL query."
    )
    return prompt


def _generate_with_ollama(prompt: str, model: str, temperature: float) -> str:
    from ollama import chat

    response = chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": temperature},
    )
    return response["message"]["content"].strip()


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
    for _ in range(n):
        if provider == "ollama":
            sql = _generate_with_ollama(prompt, model, temperature)
        elif provider == "transformers":
            sql, generator_cache = _generate_with_transformers(prompt, model, temperature, generator_cache)
        else:
            raise ValueError(f"Unsupported provider '{provider}'. Use 'ollama' or 'transformers'.")

        candidates.append(sql)
    return candidates


def _resolve_db_path(db_root: Path, db_id: str) -> Path:
    candidate_paths = [db_root / db_id / f"{db_id}.sqlite", db_root / db_id / f"{db_id}.db"]
    for path in candidate_paths:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not locate SQLite database for {db_id} under {db_root}")


def execute_sql(sql: str, db_path: Path) -> CandidateSQL:
    """Execute SQL against the specified SQLite database and capture results."""

    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(sql)
            rows = cursor.fetchall()
            result = [tuple(row) for row in rows]
            return CandidateSQL(sql=sql, execution_result=result, error=None)
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("Execution failed for candidate: %s", exc)
        return CandidateSQL(sql=sql, execution_result=None, error=str(exc))


def majority_vote(candidates: Iterable[CandidateSQL]) -> tuple[str, list[CandidateSQL]]:
    """Select the SQL query with the highest execution-based vote."""

    vote_counts: dict[str, int] = {}
    best_sql = ""
    candidate_list = list(candidates)
    for candidate in candidate_list:
        key = json.dumps(candidate.execution_result, sort_keys=True, default=str) if candidate.error is None else f"error:{candidate.error}"
        vote_counts[key] = vote_counts.get(key, 0) + 1

    if not vote_counts:
        return best_sql, candidate_list

    winning_key = max(vote_counts, key=vote_counts.get)
    for candidate in candidate_list:
        key = json.dumps(candidate.execution_result, sort_keys=True, default=str) if candidate.error is None else f"error:{candidate.error}"
        if key == winning_key:
            best_sql = candidate.sql
            break

    return best_sql, candidate_list


def run_pipeline(
    user_question: str,
    db_id: str,
    config: RAGConfig,
    examples: list[Example] | None = None,
    tables_metadata: Dict[str, Any] | None = None,
    embedder: SentenceTransformer | None = None,
    example_embeddings: list | None = None,
) -> tuple[str, list[CandidateSQL]]:
    tables_metadata = tables_metadata or _load_tables_metadata(config.dataset_path)
    examples = examples or load_examples(config.dataset_path, config.num_retrieve)

    retrieved = retrieve_similar_examples(
        user_question,
        examples,
        k=config.k,
        embedding_model_name=config.embedding_model_name,
        embedder=embedder,
        example_embeddings=example_embeddings,
    )

    schema = _format_schema(db_id, tables_metadata)
    prompt = build_prompt(schema, user_question, retrieved, prompt_technique=config.prompt_technique)
    sql_candidates = generate_sql_candidates(
        prompt,
        n=config.n,
        provider=config.llm_provider,
        model=config.llm_model,
        temperature=config.temperature,
    )

    db_path = _resolve_db_path(config.db_root, db_id)
    executed_candidates = [execute_sql(sql, db_path) for sql in sql_candidates]
    final_sql, candidates_with_votes = majority_vote(executed_candidates)
    return final_sql, candidates_with_votes


def generate_dataset_rag_predictions(
    dataset: Iterable[Example],
    config: RAGConfig,
    output_path: Path,
    num_samples: int | None = None,
) -> list[str]:
    """Run the RAG pipeline across a dataset and save predictions."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tables_metadata = _load_tables_metadata(config.dataset_path)
    retrieval_examples = load_examples(config.dataset_path, config.num_retrieve)
    embedder = SentenceTransformer(config.embedding_model_name)
    example_embeddings = embedder.encode([ex.question for ex in retrieval_examples])

    predictions: list[str] = []
    for idx, example in enumerate(dataset):
        if num_samples is not None and idx >= num_samples:
            break

        final_sql, _ = run_pipeline(
            example.question,
            example.db_id,
            config,
            examples=retrieval_examples,
            tables_metadata=tables_metadata,
            embedder=embedder,
            example_embeddings=example_embeddings,
        )
        predictions.append(final_sql)

    output_path.write_text("\n".join(predictions) + "\n", encoding="utf-8")
    LOGGER.info("Saved %d RAG predictions to %s", len(predictions), output_path)
    return predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG-based Text-to-SQL inference")
    parser.add_argument("--question", required=True, help="User question to translate into SQL.")
    parser.add_argument("--db_id", required=True, help="Database id corresponding to the question.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to JSON config file (defaults to config.json).",
    )
    parser.add_argument("--provider", choices=["ollama", "transformers"], help="LLM provider override.")
    parser.add_argument("--model", help="Model name override for the provider.")
    parser.add_argument("--prompt_technique", help="Prompting technique label (e.g., cot, direct).")
    parser.add_argument("--temperature", type=float, help="Sampling temperature override.")
    parser.add_argument("--n", type=int, help="Number of SQL generations for self-consistency.")
    parser.add_argument("--k", type=int, help="Number of retrieved examples.")
    parser.add_argument("--num_retrieve", type=int, help="Number of candidate examples to search.")
    parser.add_argument("--embedding_model", help="Sentence transformer model to use for retrieval.")
    return parser.parse_args()


def main() -> None:  # pragma: no cover - CLI glue
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    config = load_rag_config(args.config)

    if args.provider:
        config.llm_provider = args.provider
    if args.model:
        config.llm_model = args.model
    if args.prompt_technique:
        config.prompt_technique = args.prompt_technique
    if args.temperature is not None:
        config.temperature = args.temperature
    if args.n is not None:
        config.n = args.n
    if args.k is not None:
        config.k = args.k
    if args.num_retrieve is not None:
        config.num_retrieve = args.num_retrieve
    if args.embedding_model:
        config.embedding_model_name = args.embedding_model

    final_sql, candidates = run_pipeline(args.question, args.db_id, config)
    print("Final SQL:", final_sql)
    print("All candidates and execution results:")
    for idx, candidate in enumerate(candidates, 1):
        print(f"Candidate {idx}: {candidate.sql}")
        if candidate.error:
            print(f"  Error: {candidate.error}")
        else:
            print(f"  Result: {candidate.execution_result}")


if __name__ == "__main__":
    main()
