"""SQL generation pipeline built around a configured LLM provider."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Union

from sentence_transformers import SentenceTransformer

from text2sql.generation.execution import CandidateSQL, execute_sql, majority_vote, resolve_db_path
from text2sql.generation.rag_pipeline import Example, format_schema, generate_sql_candidates, retrieve_similar_examples
from text2sql.models.router import OpenAIChatLLM
from text2sql.prompt.chat_prompt import ChatPrompt
from text2sql.prompt.prompt_builder import build_cot_prompt
from text2sql.util.dataset import SpiderDataset
from text2sql.util.sql_cleaner import extract_sql_query

LOGGER = logging.getLogger(__name__)


def load_retrieval_examples(dataset_path: Path, num_retrieve: int, filename: str = "dev.json") -> list[Example]:
    """Load Text-to-SQL examples for retrieval from a dataset file."""

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


def load_tables_metadata(dataset_path: Path, tables_filename: str = "tables.json") -> dict[str, object]:
    """Read the schema metadata file (tables.json) for later formatting."""

    tables_path = dataset_path / tables_filename
    if not tables_path.exists():
        raise FileNotFoundError(f"Could not find tables.json at {tables_path}")
    return {item["db_id"]: item for item in json.loads(tables_path.read_text())}


def generate_cot_dataset_predictions(
    dataset: SpiderDataset,
    config: Mapping[str, Any],
    output_path: Path,
    num_samples: Optional[int] = None,
) -> list[str]:
    """Run retrieval-augmented generation with execution voting."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(config["dataset_path"])
    rag_config: Mapping[str, Any] = config.get("rag", {}) or {}
    provider = config.get("default_provider")
    model_name = config.get("default_model")
    if not provider or not model_name:
        raise ValueError("Both default_provider and default_model must be set in the config for RAG mode.")

    tables_metadata = load_tables_metadata(dataset_path, config.get("tables_filename", "tables.json"))
    retrieval_examples = load_retrieval_examples(
        dataset_path,
        int(rag_config.get("num_retrieve", 200)),
        filename=rag_config.get("retrieval_examples_filename", "test.json"),
    )

    embedding_model = rag_config.get("embedding_model_name", "sentence-transformers/all-MiniLM-L6-v2")
    LOGGER.info("Using embedding model '%s' for retrieval", embedding_model)
    embedder = SentenceTransformer(embedding_model)
    example_embeddings = embedder.encode([ex.question for ex in retrieval_examples])

    predictions: list[str] = []
    for example in dataset.iter_examples(limit=num_samples):
        retrieved = retrieve_similar_examples(
            example.question,
            retrieval_examples,
            k=int(rag_config.get("k", 4)),
            embedding_model_name=embedding_model,
            embedder=embedder,
            example_embeddings=example_embeddings,
        )
        LOGGER.info("Retrieved %d examples for question: %s", len(retrieved), example.question)

        schema = format_schema(example.db_id, tables_metadata)
        prompt = build_cot_prompt(
            schema,
            example.question,
            retrieved,
            prompt_technique=rag_config.get("mode", "cot"),
        )
        sql_candidates = generate_sql_candidates(
            prompt,
            n=int(rag_config.get("n", 1)),
            provider=str(provider),
            model=str(model_name),
            temperature=float(rag_config.get("temperature", 0.0)),
        )
        LOGGER.info("Generated %d SQL candidates for question: %s", len(sql_candidates), example.question)
        LOGGER.info("SQL Candidates: %s", sql_candidates)

        db_path = resolve_db_path(Path(config["db_root"]), example.db_id)
        executed_candidates: list[CandidateSQL] = [execute_sql(sql, db_path) for sql in sql_candidates]
        final_sql, _ = majority_vote(executed_candidates)
        LOGGER.info("Final SQL after execution voting: %s", final_sql)
        predictions.append(final_sql)

    output_path.write_text("\n".join(predictions) + "\n", encoding="utf-8")
    LOGGER.info("Saved %d RAG predictions to %s", len(predictions), output_path)
    return predictions


class SQLGenerator:
    """Generate SQL predictions for a dataset using a chat-completion model."""

    def __init__(
        self,
        client: OpenAIChatLLM,
        model_name: str,
        prompt_builder: Callable[[str, str, Optional[str]], Union[str, ChatPrompt]],
        request_delay: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> None:
        self.client = client
        self.model_name = model_name
        self.prompt_builder = prompt_builder
        self.request_delay = request_delay
        self.max_tokens = max_tokens

    def generate_dataset_predictions(
        self,
        dataset: SpiderDataset,
        output_path: Path,
        num_samples: Optional[int] = None,
    ) -> list[str]:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        predictions: list[str] = []
        iterator: Iterable = dataset.iter_examples(limit=num_samples)
        for example in iterator:
            schema = dataset.get_schema(example.db_id)
            prompt = self.prompt_builder(example.question, schema, db_id=example.db_id)

            try:
                if isinstance(prompt, ChatPrompt):
                    LOGGER.info("System prompt sent to LLM: %s", prompt.system_prompt)
                    LOGGER.info("User prompt sent to LLM: %s", prompt.user_prompt)
                else:
                    LOGGER.info("Prompt sent to LLM: %s", prompt)
                result = self.client.generate(
                    prompt=prompt, model=self.model_name, max_tokens=self.max_tokens
                )
                predicted_sql = extract_sql_query(result.sql)
                LOGGER.info("Predicted SQL Query: %s", predicted_sql)
            except Exception as exc:  # pragma: no cover - network dependent
                LOGGER.error("Failed to generate SQL for %s: %s", example.db_id, exc)
                predicted_sql = ""

            predictions.append(predicted_sql)

            if self.request_delay > 0:
                time.sleep(self.request_delay)

        output_path.write_text("\n".join(predictions) + "\n", encoding="utf-8")
        LOGGER.info("Saved %d predictions to %s", len(predictions), output_path)
        return predictions
