"""Centralized configuration loader for the Text-to-SQL pipeline."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, MutableMapping, Sequence

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.json"

_OVERRIDE_MAP: dict[str, Sequence[str]] = {
    "provider": ("default_provider",),
    "model": ("default_model",),
    "num_samples": ("num_sample",),
    "out": ("output_llm",),
    "mode": ("mode",),
    "k": ("rag", "k"),
    "n": ("rag", "n"),
    "num_retrieve": ("rag", "num_retrieve"),
}


def _resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def _apply_override(config: MutableMapping[str, Any], key_path: Sequence[str], value: Any) -> None:
    target = config
    for key in key_path[:-1]:
        nested = target.get(key)
        if not isinstance(nested, MutableMapping):
            nested = {}
            target[key] = nested
        target = nested  # type: ignore[assignment]
    target[key_path[-1]] = value


def load_config(config_path: str | Path | None = None, cli_args: Any | None = None) -> dict[str, Any]:
    """Load configuration from JSON and apply CLI overrides.

    All dataset, model, and RAG parameters originate from the JSON file. Command-line
    arguments listed in ``_OVERRIDE_MAP`` are applied on top to support run-time tweaks.
    """

    path = Path(config_path or DEFAULT_CONFIG_PATH).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    data = json.loads(path.read_text())
    base_dir = path.parent

    rag_data = data.get("rag", {})

    dataset_path = _resolve_path(data.get("dataset_path", "./spider_data/"), base_dir)
    db_root_default = dataset_path / "database"
    db_root = _resolve_path(data.get("db_root", db_root_default), base_dir)

    config: dict[str, Any] = {
        "dataset_path": dataset_path,
        "tables_filename": data.get("tables_filename", "tables.json"),
        "default_provider": data.get("default_provider", "deepseek"),
        "default_model": data.get("default_model", "deepseek-chat"),
        "num_sample": int(data.get("num_sample", 100)),
        "max_tokens": int(data.get("max_tokens", 8000)),
        "request_delay": float(data.get("request_delay", 0.0)),
        "mode": data.get("mode", "zero_shot"),
        "db_root": db_root,
        "output_llm": Path(data.get("output_llm", "predicted/deepseek_chat_predicted.json")),
        "rag": {
            "num_retrieve": int(rag_data.get("num_retrieve", 200)),
            "k": int(rag_data.get("k", 4)),
            "n": int(rag_data.get("n", 5)),
            "embedding_model_name": rag_data.get(
                "embedding_model_name", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            "retrieval_examples_filename": rag_data.get("retrieval_examples_filename", "test.json"),
        },
    }

    if cli_args is not None:
        for arg_name, key_path in _OVERRIDE_MAP.items():
            override_value = getattr(cli_args, arg_name, None)
            if override_value is None:
                continue

            final_value: Any = override_value
            if key_path[-1] in {"num_sample", "max_tokens", "num_retrieve", "k", "n"}:
                final_value = int(override_value)
            elif key_path[-1] in {"output_llm"}:
                final_value = Path(override_value)
            _apply_override(config, key_path, final_value)

    config["rag"]["prompt_technique"] = (config.get("mode") or "cot").lower()
    return config
