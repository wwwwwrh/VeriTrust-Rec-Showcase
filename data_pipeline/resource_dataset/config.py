"""Configuration loading for the Resource Dataset build."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DatasetConfig:
    raw: dict[str, Any]
    vocabulary: dict[str, Any]
    config_path: Path
    vocabulary_path: Path
    output_root: Path | None = None

    @property
    def dataset(self) -> dict[str, Any]:
        return self.raw["dataset"]

    def path(self, key: str) -> Path:
        value = Path(self.dataset["paths"][key])
        if value.is_absolute():
            return value
        return (self.output_root or project_root()) / value

    @property
    def priors(self) -> dict[str, float]:
        return self.raw["source_reliability_priors"]


def load_config(
    config_path: Path | None = None, vocabulary_path: Path | None = None,
    *, output_root: Path | None = None,
) -> DatasetConfig:
    root = project_root()
    config_path = config_path or root / "configs" / "resource_dataset.yaml"
    vocabulary_path = vocabulary_path or root / "configs" / "vocabulary.yaml"
    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    with vocabulary_path.open("r", encoding="utf-8") as file:
        vocabulary = yaml.safe_load(file)
    return DatasetConfig(
        raw=raw, vocabulary=vocabulary, config_path=config_path, vocabulary_path=vocabulary_path,
        output_root=output_root.resolve() if output_root else None,
    )
