"""Command-line interface for running the Text-to-SQL pipeline."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from time import perf_counter
from typing import Optional

from dotenv import load_dotenv

from text2sql.config.config_loader import DEFAULT_CONFIG_PATH, AppConfig, load_config
from text2sql.config.rag_config import RAGConfig, load_rag_config
from text2sql.generation.sql_generator import SQLGenerator
from text2sql.generation.rag_pipeline import generate_dataset_rag_predictions, run_pipeline
from text2sql.models.router import ROUTER_CONFIGS, OpenAIChatLLM
from text2sql.prompt.zero_shot import build_prompt
from text2sql.util.dataset import SpiderDataset, load_dataset
from text2sql.util.logger import setup_logging

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Spider Text-to-SQL pipeline")
    rag_group = parser.add_argument_group("Retrieval-augmented (single question)")
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
    parser.add_argument("--embedding_model", help="Sentence transformer model to use for retrieval.")
    parser.add_argument("--temperature", type=float, help="Sampling temperature for RAG generation.")
    rag_group.add_argument("--rag_question", help="Run retrieval-augmented pipeline for a single question.")
    rag_group.add_argument("--rag_db_id", help="Database id for the retrieval-augmented pipeline.")
    rag_group.add_argument(
        "--rag_provider",
        choices=["ollama", "transformers"],
        help="LLM provider override for retrieval-augmented mode.",
    )
    rag_group.add_argument("--rag_model", help="Model override for retrieval-augmented mode.")
    rag_group.add_argument("--rag_prompt_technique", help="Prompting technique label (e.g., cot, direct).")
    rag_group.add_argument("--rag_temperature", type=float, help="Sampling temperature override.")
    rag_group.add_argument("--rag_n", type=int, help="Number of SQL generations for self-consistency.")
    rag_group.add_argument("--rag_k", type=int, help="Number of retrieved examples.")
    rag_group.add_argument("--rag_num_retrieve", type=int, help="Number of candidate examples to search.")
    rag_group.add_argument("--rag_embedding_model", help="Sentence transformer model to use for retrieval.")
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

    if args.mode:
        config.mode = args.mode

    _load_environment(args.config)
    log_file = setup_logging(Path("output"))
    LOGGER.info("Logs will be written to %s", log_file)

    rag_requested = args.rag_question or args.rag_db_id
    if rag_requested:
        if not (args.rag_question and args.rag_db_id):
            raise ValueError("--rag_question and --rag_db_id must both be provided for RAG mode.")

        rag_config: RAGConfig = load_rag_config(args.config)
        if args.rag_provider:
            rag_config.llm_provider = args.rag_provider
        if args.rag_model:
            rag_config.llm_model = args.rag_model
        if args.rag_prompt_technique:
            rag_config.prompt_technique = args.rag_prompt_technique
        if args.rag_temperature is not None:
            rag_config.temperature = args.rag_temperature
        if args.rag_n is not None:
            rag_config.n = args.rag_n
        if args.rag_k is not None:
            rag_config.k = args.rag_k
        if args.rag_num_retrieve is not None:
            rag_config.num_retrieve = args.rag_num_retrieve
        if args.rag_embedding_model:
            rag_config.embedding_model_name = args.rag_embedding_model

        final_sql, candidates = run_pipeline(args.rag_question, args.rag_db_id, rag_config)
        print("Final SQL:", final_sql)
        print("All candidates and execution results:")
        for idx, candidate in enumerate(candidates, 1):
            print(f"Candidate {idx}: {candidate.sql}")
            if candidate.error:
                print(f"  Error: {candidate.error}")
            else:
                print(f"  Result: {candidate.execution_result}")
        return

    dataset: SpiderDataset = load_dataset(config.dataset_path)
    predictions_path = _resolve_paths(config, args.out)
    rag_mode = (config.mode or "").lower() != "zero_shot"

    if rag_mode:
        rag_config = load_rag_config(args.config)
        rag_config.prompt_technique = config.mode
        if args.provider:
            rag_config.llm_provider = args.provider
        if args.model:
            rag_config.llm_model = args.model
        if args.k is not None:
            rag_config.k = args.k
        if args.n is not None:
            rag_config.n = args.n
        if args.num_retrieve is not None:
            rag_config.num_retrieve = args.num_retrieve
        if args.embedding_model:
            rag_config.embedding_model_name = args.embedding_model
        if args.temperature is not None:
            rag_config.temperature = args.temperature

        LOGGER.info(
            "Running retrieval-augmented pipeline with provider %s model %s (mode=%s)",
            rag_config.llm_provider,
            rag_config.llm_model,
            rag_config.prompt_technique,
        )
        num_samples = args.num_samples if args.num_samples is not None else config.num_sample
        start_time = perf_counter()
        generate_dataset_rag_predictions(dataset, rag_config, predictions_path, num_samples=num_samples)
        elapsed = perf_counter() - start_time
    else:
        model_name = args.model or config.default_model
        if not model_name:
            raise ValueError("No model specified. Provide --model or default_model in config.json.")

        provider_name = args.provider or config.default_provider
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
