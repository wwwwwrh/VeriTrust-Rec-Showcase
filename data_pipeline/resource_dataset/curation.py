"""Explicit, versioned, resource-ID-only curation for Resource Dataset builds."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


RESOURCE_ID_PATTERN = re.compile(r"^github:[1-9][0-9]*$")
HASH_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class CurationManifest:
    path: Path
    raw: dict[str, Any]
    manifest_id: str
    version: str
    raw_run_id: str
    input_resource_build_id: str
    exclusions: tuple[dict[str, str], ...]
    canonical_hash: str

    @property
    def excluded_resource_ids(self) -> frozenset[str]:
        return frozenset(item["resource_id"] for item in self.exclusions)

    def build_metadata(self) -> dict[str, Any]:
        return {
            "manifest_id": self.manifest_id,
            "version": self.version,
            "canonical_hash": self.canonical_hash,
            "decision_scope": "exclude_only",
            "input_raw_run_id": self.raw_run_id,
            "input_resource_build_id": self.input_resource_build_id,
            "exclusions_requested": len(self.exclusions),
            "excluded_resource_ids": sorted(self.excluded_resource_ids),
        }


def load_curation_manifest(path: Path) -> CurationManifest:
    resolved = path.resolve()
    parsed = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError("curation manifest must be a mapping")
    allowed = {"curation_manifest_version", "manifest_id", "input", "decision_scope", "exclusions"}
    unexpected = set(parsed) - allowed
    if unexpected:
        raise ValueError(f"unexpected curation manifest fields: {sorted(unexpected)}")
    version = parsed.get("curation_manifest_version")
    manifest_id = parsed.get("manifest_id")
    inputs = parsed.get("input")
    exclusions = parsed.get("exclusions")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("curation_manifest_version must be a non-empty string")
    if not isinstance(manifest_id, str) or not manifest_id.strip():
        raise ValueError("manifest_id must be a non-empty string")
    if parsed.get("decision_scope") != "exclude_only":
        raise ValueError("curation decision_scope must be exclude_only")
    if not isinstance(inputs, dict) or set(inputs) != {"raw_run_id", "resource_build_id"}:
        raise ValueError("curation input must contain only raw_run_id and resource_build_id")
    if not all(isinstance(inputs.get(key), str) and inputs[key].strip() for key in inputs):
        raise ValueError("curation input identifiers must be non-empty strings")
    if not isinstance(exclusions, list):
        raise ValueError("curation exclusions must be a list")
    normalized: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in exclusions:
        if not isinstance(item, dict) or set(item) != {"resource_id", "reason_code", "reason"}:
            raise ValueError("each exclusion must contain only resource_id, reason_code, and reason")
        if not all(isinstance(item.get(key), str) and item[key].strip() for key in item):
            raise ValueError("exclusion values must be non-empty strings")
        resource_id = item["resource_id"].strip()
        if not RESOURCE_ID_PATTERN.fullmatch(resource_id):
            raise ValueError(f"curation exclusion must use exact GitHub numeric resource_id: {resource_id}")
        if resource_id in seen:
            raise ValueError(f"duplicate curation exclusion: {resource_id}")
        seen.add(resource_id)
        normalized.append({
            "resource_id": resource_id,
            "reason_code": item["reason_code"].strip(),
            "reason": item["reason"].strip(),
        })
    normalized.sort(key=lambda item: item["resource_id"])
    canonical = {
        "curation_manifest_version": version.strip(),
        "manifest_id": manifest_id.strip(),
        "input": {
            "raw_run_id": inputs["raw_run_id"].strip(),
            "resource_build_id": inputs["resource_build_id"].strip(),
        },
        "decision_scope": "exclude_only",
        "exclusions": normalized,
    }
    canonical_hash = "sha256:" + hashlib.sha256(_canonical_json(canonical).encode("utf-8")).hexdigest()
    if not HASH_PATTERN.fullmatch(canonical_hash):
        raise AssertionError("invalid canonical curation hash")
    return CurationManifest(
        path=resolved,
        raw=canonical,
        manifest_id=canonical["manifest_id"],
        version=canonical["curation_manifest_version"],
        raw_run_id=canonical["input"]["raw_run_id"],
        input_resource_build_id=canonical["input"]["resource_build_id"],
        exclusions=tuple(normalized),
        canonical_hash=canonical_hash,
    )
