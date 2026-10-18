"""Configuration for the Phase 3 Evidence Corpus pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class EvidenceConfig:
    raw: dict[str, Any]
    config_path: Path
    output_root: Path | None = None
    resource_manifest_override: Path | None = None

    @property
    def evidence(self) -> dict[str, Any]:
        return self.raw["evidence"]

    @property
    def chunking(self) -> dict[str, Any]:
        return self.raw["chunking"]

    @property
    def normalization(self) -> dict[str, Any]:
        return self.raw["normalization"]

    @property
    def review(self) -> dict[str, Any]:
        return self.raw["review"]

    def path(self, key: str) -> Path:
        value = Path(self.raw["paths"][key])
        return value if value.is_absolute() else (self.output_root or project_root()) / value

    def input_path(self, key: str) -> Path:
        if key == "resource_manifest" and self.resource_manifest_override:
            return self.resource_manifest_override
        value = Path(self.raw["inputs"][key])
        return value if value.is_absolute() else (self.output_root or project_root()) / value


def _validate_config(raw: dict[str, Any]) -> None:
    required = {"evidence", "sources", "normalization", "chunking", "review", "random_seed", "inputs", "paths"}
    missing = required - raw.keys()
    if missing:
        raise ValueError(f"missing evidence config sections: {sorted(missing)}")
    chunking = raw["chunking"]
    maximum = int(chunking["max_chunk_chars"])
    overlap = int(chunking["overlap_chars"])
    minimum = int(chunking["min_chunk_chars"])
    if maximum <= 0 or minimum <= 0 or not 0 <= overlap < maximum:
        raise ValueError("chunk sizes must be positive and overlap must be smaller than max_chunk_chars")
    if minimum > maximum:
        raise ValueError("min_chunk_chars cannot exceed max_chunk_chars")
    if int(raw["review"]["oversized_chunk_chars"]) < maximum:
        raise ValueError("oversized_chunk_chars cannot be smaller than max_chunk_chars")


def load_config(
    path: Path | None = None, *, output_root: Path | None = None,
    resource_manifest: Path | None = None,
) -> EvidenceConfig:
    path = path or project_root() / "configs" / "evidence_corpus.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    _validate_config(raw)
    return EvidenceConfig(
        raw=raw, config_path=path,
        output_root=output_root.resolve() if output_root else None,
        resource_manifest_override=resource_manifest.resolve() if resource_manifest else None,
    )
