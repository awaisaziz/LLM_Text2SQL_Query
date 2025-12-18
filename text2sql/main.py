"""Command-line interface for running the Text-to-SQL pipeline."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv

from text2sql.config import DEFAULT_CONFIG_PATH, load_config
from text2sql.generation.sql_generator import SQLGenerator
from text2sql.generation.rag_pipeline import generate_dataset_rag_predictions, run_pipeline
from text2sql.models.router import ROUTER_CONFIGS, OpenAIChatLLM
from text2sql.prompt.zero_shot import build_prompt
from text2sql.util.dataset import SpiderDataset, load_dataset
from text2sql.util.logger import setup_logging

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Spider Text-to-SQL pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to configuration JSON file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model identifier for the selected provider (defaults to config default_model).",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        choices=sorted(ROUTER_CONFIGS.keys()),
        help="LLM provider to target (defaults to config default_provider).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        help="Pipeline mode: 'zero_shot' for baseline or 'cot' to enable retrieval-augmented prompting.",
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
        default=None,
        help="Destination path for predictions (defaults to config output_llm).",
    )
    parser.add_argument("--k", type=int, help="Number of retrieved examples for RAG mode.")
    parser.add_argument("--n", type=int, help="Number of SQL generations for self-consistency in RAG mode.")
    parser.add_argument("--num_retrieve", type=int, help="Number of candidate examples to search for retrieval.")
    return parser.parse_args()


def _load_environment(config_path: Path) -> None:
    load_dotenv(dotenv_path=config_path.with_name(".env"), override=False)
    load_dotenv(override=False)


def _resolve_predictions_path(output_path: Path) -> Path:
    output_root = Path("output")
    output_root.mkdir(exist_ok=True)
    return output_path if output_path.is_absolute() else output_root / output_path


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args)

    _load_environment(Path(args.config))
    log_file = setup_logging(Path("output"))
    LOGGER.info("Logs will be written to %s", log_file)

    dataset: SpiderDataset = load_dataset(config["dataset_path"])
    predictions_path = _resolve_predictions_path(Path(config["output_llm"]))
    rag_mode = (config.get("mode") or "").lower() == "cot"

    if rag_mode:
        LOGGER.info(
            "Running retrieval-augmented pipeline with provider %s model %s (mode=%s)",
            config.get("default_provider"),
            config.get("default_model"),
            config.get("mode"),
        )
        num_samples = args.num_samples if args.num_samples is not None else config.get("num_sample")
        start_time = perf_counter()
        generate_dataset_rag_predictions(dataset, config, predictions_path, num_samples=num_samples)
        elapsed = perf_counter() - start_time
    else:
        model_name = config.get("default_model")
        if not model_name:
            raise ValueError("No model specified. Provide --model or default_model in config.json.")

        provider_name = config.get("default_provider")
        if provider_name not in ROUTER_CONFIGS:
            raise ValueError(
                f"Unsupported provider '{provider_name}'. Valid options: {', '.join(sorted(ROUTER_CONFIGS))}."
            )

        LOGGER.info("Using provider %s with model %s", provider_name, model_name)

        client = OpenAIChatLLM(router=provider_name)
        generator = SQLGenerator(
            client=client,
            model_name=model_name,
            prompt_builder=build_prompt,
            request_delay=config.get("request_delay", 0.0),
            max_tokens=config.get("max_tokens", 8000),
        )

        start_time = perf_counter()
        num_samples = args.num_samples if args.num_samples is not None else config.get("num_sample")
        generator.generate_dataset_predictions(dataset, predictions_path, num_samples=num_samples)
        elapsed = perf_counter() - start_time

    total_samples = num_samples or len(dataset)
    avg_latency = elapsed / total_samples if total_samples else 0.0
    LOGGER.info("Total latency: %.2f seconds (avg %.2f s/example)", elapsed, avg_latency)


if __name__ == "__main__":  # pragma: no cover
    main()
