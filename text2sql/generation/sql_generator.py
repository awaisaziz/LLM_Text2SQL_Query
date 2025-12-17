"""SQL generation pipeline built around a configured LLM provider."""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, Iterable, Optional

from text2sql.models.router import OpenAIChatLLM
from text2sql.util.dataset import SpiderDataset
from text2sql.util.sql_cleaner import extract_sql_query

LOGGER = logging.getLogger(__name__)


class SQLGenerator:
    """Generate SQL predictions for a dataset using a chat-completion model."""

    def __init__(
        self,
        client: OpenAIChatLLM,
        model_name: str,
        prompt_builder: Callable[[str, str, Optional[str]], str],
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
