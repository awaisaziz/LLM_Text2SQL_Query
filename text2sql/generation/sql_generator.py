"""SQL generation pipeline built around a configured LLM provider."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional

from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from text2sql.generation.execution import majority_vote_sql, normalize_sql_query, resolve_db_path
from text2sql.generation.rag_pipeline import Example, format_schema, generate_sql_candidates, retrieve_similar_examples
from text2sql.models.router import OpenAIChatLLM, Prompt
from text2sql.prompt.prompt_builder import build_cot_prompt
from text2sql.util.dataset import SpiderDataset
from text2sql.util.sql_cleaner import extract_sql_query

LOGGER = logging.getLogger(__name__)


def _write_plain_sql(predictions: list[tuple[str, str]], output_path: Path) -> None:
    """Persist predicted SQL statements as plain text (Spider)."""

    output_path.write_text("\n".join(sql for sql, _ in predictions) + "\n", encoding="utf-8")


def _write_bird_json(predictions: list[tuple[str, str]], output_path: Path) -> None:
    """Persist predicted SQL statements in the BIRD JSON format."""

    formatted = {str(idx): f"{sql}\t----- bird -----\t{db_id}" for idx, (sql, db_id) in enumerate(predictions)}
    output_path.write_text(json.dumps(formatted, ensure_ascii=False, indent=2), encoding="utf-8")


def load_retrieval_examples(
    dataset_path: Path, num_retrieve: int, filename: str = "dev.json", sql_field: str = "query"
) -> list[Example]:
    """Load Text-to-SQL examples for retrieval from a dataset file."""

    data_path = dataset_path / filename
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find {filename} at {data_path}")

    raw_items = json.loads(data_path.read_text())[:num_retrieve]
    examples: list[Example] = []
    for item in raw_items:
        sql_value = item.get(sql_field) or item.get("query") or item.get("SQL") or item.get("sql")
        examples.append(Example(question=item["question"], sql=sql_value, db_id=item["db_id"]))
    LOGGER.info("Loaded %d retrieval examples from %s", len(examples), filename)
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
    max_tokens: Optional[int] = None,
    dataset_name: str = "spider",
) -> list[str]:
    """Run retrieval-augmented generation with execution voting."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(config["dataset_path"])
    provider = config.get("default_provider")
    model_name = config.get("default_model")
    if not provider or not model_name:
        raise ValueError("Both default_provider and default_model must be set in the config for RAG mode.")

    tables_metadata = load_tables_metadata(dataset_path, config.get("tables_filename", "tables.json"))
    retrieval_examples = load_retrieval_examples(
        dataset_path,
        int(config["rag"].get("num_retrieve", 200)),
        filename=config["rag"].get("retrieval_examples_filename", "test.json"),
        sql_field=config.get("sql_field", "query"),
    )

    embedding_model = config["rag"].get("embedding_model_name", "sentence-transformers/all-MiniLM-L6-v2")
    LOGGER.info("Using embedding model '%s' for retrieval", embedding_model)
    embedder = SentenceTransformer(embedding_model)
    example_embeddings = embedder.encode([ex.question for ex in retrieval_examples])

    predictions: list[tuple[str, str]] = []
    examples_iter = dataset.iter_examples(limit=num_samples)
    total_examples = num_samples if num_samples is not None else len(dataset)
    for example in tqdm(examples_iter, total=total_examples, desc="Generating SQL (RAG)"):
        retrieved = retrieve_similar_examples(
            example.question,
            retrieval_examples,
            k=int(config["rag"].get("k", 4)),
            embedding_model_name=embedding_model,
            embedder=embedder,
            example_embeddings=example_embeddings,
        )
        # LOGGER.info("Retrieved %d examples for question: %s", len(retrieved), example.question)
        LOGGER.info("Retrieved examples: %s", [ex.sql for ex in retrieved])

        schema = format_schema(example.db_id, tables_metadata)
        prompt = build_cot_prompt(
            schema,
            example.question,
            retrieved,
            mode=str(config.get("mode", "cot")),
        )

        LOGGER.info("Messages sent to LLM: %s", prompt.get("user"))
        sql_candidates = generate_sql_candidates(
            prompt,
            n=int(config["rag"].get("n", 5)),
            provider=str(provider),
            model=str(model_name),
            max_tokens=max_tokens,
        )
        LOGGER.info("Generated %d SQL candidates for question: %s", len(sql_candidates), example.question)
        LOGGER.info("SQL Candidates: %s", sql_candidates)

        db_path = resolve_db_path(Path(config["db_root"]), example.db_id)
        final_sql, executed_candidates = majority_vote_sql(sql_candidates, db_path)
        # normalized_final_sql = normalize_sql_query(final_sql)
        # LOGGER.info("Final SQL after execution voting: %s", normalized_final_sql)
        # predictions.append(normalized_final_sql)
        LOGGER.info("Final SQL after execution voting: %s", final_sql)
        predictions.append((final_sql, example.db_id))

    if dataset_name.lower() == "bird":
        _write_bird_json(predictions, output_path)
    else:
        _write_plain_sql(predictions, output_path)
    LOGGER.info("Saved %d RAG predictions to %s", len(predictions), output_path)
    return [sql for sql, _ in predictions]


class SQLGenerator:
    """Generate SQL predictions for a dataset using a chat-completion model."""

    def __init__(
        self,
        client: OpenAIChatLLM,
        model_name: str,
        prompt_builder: Callable[[str, str, Optional[str]], Prompt],
        request_delay: float = 0.0,
        max_tokens: Optional[int] = None,
        dataset_name: str = "spider",
    ) -> None:
        self.client = client
        self.model_name = model_name
        self.prompt_builder = prompt_builder
        self.request_delay = request_delay
        self.max_tokens = max_tokens
        self.dataset_name = dataset_name

    def generate_dataset_predictions(
        self,
        dataset: SpiderDataset,
        output_path: Path,
        num_samples: Optional[int] = None,
    ) -> list[str]:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        predictions: list[tuple[str, str]] = []
        iterator: Iterable = dataset.iter_examples(limit=num_samples)
        total_examples = num_samples if num_samples is not None else len(dataset)
        for example in tqdm(iterator, total=total_examples, desc="Generating SQL"):
            schema = dataset.get_schema(example.db_id)
            prompt = self.prompt_builder(example.question, schema, db_id=example.db_id)

            try:
                if isinstance(prompt, Mapping):
                    LOGGER.info("System prompt sent to LLM: %s", prompt.get("system"))
                    LOGGER.info("User prompt sent to LLM: %s", prompt.get("user"))
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

            predictions.append((predicted_sql, example.db_id))

            if self.request_delay > 0:
                time.sleep(self.request_delay)

        if self.dataset_name.lower() == "bird":
            _write_bird_json(predictions, output_path)
        else:
            _write_plain_sql(predictions, output_path)
        LOGGER.info("Saved %d predictions to %s", len(predictions), output_path)
        return [sql for sql, _ in predictions]
