from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import yaml


@dataclass
class DatasetSourceConfig:
    """Configuration for an individual dataset source."""

    name: str
    source_type: str = "local"  # "local", "kaggle", "hf"
    path_or_identifier: str = ""
    license: str = "Unknown"
    language: str = "en"
    text_column: str = "text"
    file_format: str = "txt"  # "txt", "jsonl", "csv"
    sampling_weight: float = 1.0
    min_chars: int = 20
    max_chars: int = 100_000
    filter_duplicates: bool = True


@dataclass
class DatasetMixConfig:
    """Configuration for mixing multiple datasets into training shards."""

    version: str = "0.1.0"
    tokenizer_path: str = "data/tokenizer"
    max_sequence_length: int = 256
    train_ratio: float = 0.95
    val_ratio: float = 0.05
    test_ratio: float = 0.0
    sources: list[DatasetSourceConfig] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> DatasetMixConfig:
        p = Path(path)
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        sources_raw = data.pop("sources", [])
        sources = [DatasetSourceConfig(**s) for s in sources_raw]
        return cls(sources=sources, **data)
