"""Command-line interface for running the Text-to-SQL baseline."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from time import perf_counter
from typing import List

from dotenv import load_dotenv
from tqdm import tqdm

from . import data_utils, llm, prompt_template

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Spider Text-to-SQL baseline")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config.json",
        help="Path to configuration JSON file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model identifier for the selected router (defaults to config default_model).",
    )
    parser.add_argument(
        "--router",
        type=str,
        default=None,
        choices=sorted(llm.ROUTER_CONFIGS.keys()),
        help="LLM router to target (defaults to config default_router or openrouter).",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of examples to evaluate (default: all).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("predictions.sql"),
        help="Destination JSONL path for predictions.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def main() -> None:
    # Configurable parameters
    args = parse_args()
    
    # Load the api key
    load_dotenv(dotenv_path=args.config.with_name(".env"), override=False)
    load_dotenv(override=False)
    
    # Create log folder if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    # Log file path
    log_file = log_dir / "run.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s:%(name)s:%(message)s",
        filename=log_file,
        filemode="w"   # "w" to overwrite on each run, or "a" to append
    )

    # Also log to console
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    logging.getLogger().addHandler(console)

    logging.info("Logging has started")

    dataset = data_utils.load_dataset_from_config(args.config)
    config = json.loads(Path(args.config).read_text())

    model_name = args.model or config.get("default_model")
    if not model_name:
        raise ValueError("No model specified. Provide --model or default_model in config.json.")

    router_name = args.router or config.get("default_router") or "openrouter"
    if router_name not in llm.ROUTER_CONFIGS:
        raise ValueError(
            f"Unsupported router '{router_name}'. Valid options: {', '.join(sorted(llm.ROUTER_CONFIGS))}."
        )

    LOGGER.info("Using router %s with model %s", router_name, model_name)

    predictions_path = args.out
    predictions_path.parent.mkdir(parents=True, exist_ok=True)

    # Initiate the LLM through the selected router
    client = llm.OpenAIChatLLM(router=router_name)

    pred_rows: List[str] = []
    start_time = perf_counter()
    iterator = dataset.iter_examples(limit=args.num_samples)
    total = args.num_samples or len(dataset)
    for example in tqdm(iterator, total=total, desc="Generating SQL"):
        schema = dataset.get_schema(example.db_id)
        prompt = prompt_template.build_prompt(example.question, schema, db_id=example.db_id)

        try:
            LOGGER.info("Prompt sent to LLM: %s", prompt)
            result = client.generate(prompt=prompt, model=model_name)
            predicted_sql = data_utils.extract_sql_query(result.sql)
            LOGGER.info("Predicted SQL Query: %s", predicted_sql)
        except Exception as exc:  # pragma: no cover - network dependent
            LOGGER.error("Failed to generate SQL for %s: %s", example.db_id, exc)
            predicted_sql = ""

        # Append SQL (or empty line if model failed)
        pred_rows.append(predicted_sql)

    elapsed = perf_counter() - start_time
    predictions_path.write_text("\n".join(pred_rows) + "\n", encoding="utf-8")
    LOGGER.info("Saved %d predictions to %s", len(pred_rows), predictions_path)
    avg_latency = elapsed / len(pred_rows) if pred_rows else 0.0
    LOGGER.info("Total latency: %.2f seconds (avg %.2f s/example)", elapsed, avg_latency)


if __name__ == "__main__":  # pragma: no cover
    main()
