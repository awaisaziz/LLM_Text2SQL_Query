"""Configuration helpers for the retrieval-augmented pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import json


@dataclass
class RAGConfig:
    dataset_path: Path
    db_root: Path
    num_retrieve: int
    k: int
    n: int
    embedding_model_name: str
    prompt_technique: str
    llm_provider: str
    llm_model: str
    temperature: float


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.json"


def load_rag_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> RAGConfig:
    """Load RAG-specific configuration from the shared JSON config."""

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    data: Dict[str, Any] = json.loads(path.read_text())
    rag_data: Dict[str, Any] = data.get("rag", {})

    dataset_path = Path(data.get("dataset_path", "./spider_data/")).expanduser()
    db_root = Path(data.get("db_root", dataset_path / "database")).expanduser()

    default_provider = data.get("default_provider", "deepseek")
    default_model = data.get("default_model", "deepseek-chat")

    return RAGConfig(
        dataset_path=dataset_path if dataset_path.is_absolute() else (Path.cwd() / dataset_path).resolve(),
        db_root=db_root if db_root.is_absolute() else (Path.cwd() / db_root).resolve(),
        num_retrieve=int(rag_data.get("num_retrieve", 200)),
        k=int(rag_data.get("k", 4)),
        n=int(rag_data.get("n", 5)),
        embedding_model_name=rag_data.get(
            "embedding_model_name", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        prompt_technique=data.get("mode", "zero_shot"),
        llm_provider=default_provider,
        llm_model=default_model,
        temperature=0.2,
    )
