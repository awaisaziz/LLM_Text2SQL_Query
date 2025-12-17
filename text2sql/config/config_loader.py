"""Configuration loader for the Text2SQL pipeline."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AppConfig:
    dataset_path: Path
    default_provider: str
    default_model: str
    num_sample: int
    max_tokens: int
    request_delay: float
    mode: str
    db_root: Path
    output_llm: Path


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.json"


def load_config(config_path: Path | str = DEFAULT_CONFIG_PATH) -> AppConfig:
    path = Path(config_path).resolve()
    config_data = json.loads(path.read_text())
    dataset_path = Path(config_data.get("dataset_path", "./spider_data/")).expanduser()
    if not dataset_path.is_absolute():
        dataset_path = (Path.cwd() / dataset_path).resolve()

    db_root = Path(config_data.get("db_root", "spider_data/database")).expanduser()
    if not db_root.is_absolute():
        db_root = (Path.cwd() / db_root).resolve()

    output_path = Path(
        config_data.get("output_llm", "output/predicted/deepseek_chat_predicted.json")
    )

    return AppConfig(
        dataset_path=dataset_path,
        default_provider=config_data.get("default_provider", "deepseek"),
        default_model=config_data.get("default_model", "deepseek-chat"),
        num_sample=int(config_data.get("num_sample", 100)),
        max_tokens=int(config_data.get("max_tokens", 8000)),
        request_delay=float(config_data.get("request_delay", 0.0)),
        mode=config_data.get("mode", "zero_shot"),
        db_root=db_root,
        output_llm=output_path,
    )
