"""Command-line interface for running the Text-to-SQL pipeline."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from time import perf_counter
from typing import Optional

from dotenv import load_dotenv

from text2sql.config.config_loader import DEFAULT_CONFIG_PATH, AppConfig, load_config
from text2sql.generation.sql_generator import SQLGenerator
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
    return parser.parse_args()


def _load_environment(config_path: Path) -> None:
    load_dotenv(dotenv_path=config_path.with_name(".env"), override=False)
    load_dotenv(override=False)


def _resolve_paths(config: AppConfig, override_out: Optional[Path]) -> Path:
    output_root = Path("output")
    output_root.mkdir(exist_ok=True)

    predictions_path = override_out or config.output_llm
    predictions_path = predictions_path if predictions_path.is_absolute() else output_root / predictions_path
    return predictions_path


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    _load_environment(args.config)
    log_file = setup_logging(Path("output"))
    LOGGER.info("Logs will be written to %s", log_file)

    dataset: SpiderDataset = load_dataset(config.dataset_path)
    model_name = args.model or config.default_model
    if not model_name:
        raise ValueError("No model specified. Provide --model or default_model in config.json.")

    provider_name = args.provider or config.default_provider
    if provider_name not in ROUTER_CONFIGS:
        raise ValueError(
            f"Unsupported provider '{provider_name}'. Valid options: {', '.join(sorted(ROUTER_CONFIGS))}."
        )

    predictions_path = _resolve_paths(config, args.out)

    LOGGER.info("Using provider %s with model %s", provider_name, model_name)

    client = OpenAIChatLLM(router=provider_name)
    generator = SQLGenerator(
        client=client,
        model_name=model_name,
        prompt_builder=build_prompt,
        request_delay=config.request_delay,
        max_tokens=config.max_tokens,
    )

    start_time = perf_counter()
    num_samples = args.num_samples if args.num_samples is not None else config.num_sample
    generator.generate_dataset_predictions(dataset, predictions_path, num_samples=num_samples)
    elapsed = perf_counter() - start_time

    total_samples = num_samples or len(dataset)
    avg_latency = elapsed / total_samples if total_samples else 0.0
    LOGGER.info("Total latency: %.2f seconds (avg %.2f s/example)", elapsed, avg_latency)


if __name__ == "__main__":  # pragma: no cover
    main()
