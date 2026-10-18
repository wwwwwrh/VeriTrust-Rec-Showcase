"""Validated access to the Phase 2 Resource Dataset and Phase 3 Evidence Corpus."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from pydantic import ValidationError

from data_pipeline.evidence.loader import load_evidence_chunks, load_evidence_documents
from data_pipeline.evidence.schema import EvidenceChunk, EvidenceDocument
from data_pipeline.resource_dataset.schema import Resource

from .config import CompatibilityConfig
from .errors import GlobalErrorCode, PipelineError


HASH_RE = re.compile(r"^[0-9a-f]{64}$")
BUILD_ID_RE = re.compile(r"^evidence_build_[0-9a-f]{16}$")


@dataclass(frozen=True)
class EvidenceCorpus:
    documents: list[EvidenceDocument]
    chunks: list[EvidenceChunk]
    manifest: dict[str, Any]


def _read_json(path: Path, *, missing_code: GlobalErrorCode, invalid_code: GlobalErrorCode) -> dict[str, Any]:
    if not path.is_file():
        raise PipelineError(missing_code, f"required file is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(invalid_code, f"cannot parse JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PipelineError(invalid_code, f"expected JSON object in {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    records: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"line {number} is not a JSON object")
        records.append(value)
    return records


def load_resources(path: Path) -> list[Resource]:
    if not path.is_file():
        raise PipelineError(GlobalErrorCode.RESOURCE_DATASET_MISSING, f"Resource Dataset is missing: {path}")
    try:
        if path.suffix.casefold() == ".parquet":
            records = pq.read_table(path).to_pylist()
        elif path.suffix.casefold() == ".jsonl":
            records = read_jsonl(path)
        else:
            raise ValueError("Resource Dataset must be JSONL or Parquet")
        resources = [Resource.model_validate(record) for record in records]
    except (OSError, ValueError, json.JSONDecodeError, ValidationError) as exc:
        raise PipelineError(GlobalErrorCode.RESOURCE_DATASET_INVALID, f"invalid Resource Dataset: {exc}") from exc
    ids = [item.resource_id for item in resources]
    if not resources or len(ids) != len(set(ids)):
        raise PipelineError(GlobalErrorCode.RESOURCE_DATASET_INVALID, "Resource Dataset is empty or has duplicate IDs")
    return sorted(resources, key=lambda item: item.resource_id)


def load_resource_provenance(path: Path) -> list[dict[str, Any]]:
    try:
        records = read_jsonl(path)
    except (OSError, ValueError, json.JSONDecodeError, FileNotFoundError) as exc:
        raise PipelineError(
            GlobalErrorCode.RESOURCE_DATASET_INVALID,
            f"invalid Resource Dataset provenance: {exc}",
        ) from exc
    return records


def _check_version(manifest: dict[str, Any], field: str, allowed: list[str]) -> None:
    value = manifest.get(field)
    if value not in allowed:
        raise PipelineError(
            GlobalErrorCode.EVIDENCE_BUILD_INCOMPATIBLE,
            f"unsupported Evidence {field}: {value!r}; allowed={allowed}",
        )


def load_evidence_corpus(
    *,
    documents_path: Path,
    chunks_path: Path,
    evidence_manifest_path: Path,
    resource_manifest_path: Path,
    resources: list[Resource],
    compatibility: CompatibilityConfig,
) -> EvidenceCorpus:
    manifest = _read_json(
        evidence_manifest_path,
        missing_code=GlobalErrorCode.EVIDENCE_CORPUS_MISSING,
        invalid_code=GlobalErrorCode.EVIDENCE_CORPUS_INVALID,
    )
    resource_manifest = _read_json(
        resource_manifest_path,
        missing_code=GlobalErrorCode.RESOURCE_DATASET_MISSING,
        invalid_code=GlobalErrorCode.RESOURCE_DATASET_INVALID,
    )
    if manifest.get("manifest_role") != "atomic_build_commit_marker":
        raise PipelineError(GlobalErrorCode.EVIDENCE_CORPUS_INVALID, "invalid Evidence manifest commit marker")
    build_id = manifest.get("evidence_build_id")
    if not isinstance(build_id, str) or not BUILD_ID_RE.fullmatch(build_id):
        raise PipelineError(GlobalErrorCode.EVIDENCE_CORPUS_INVALID, "invalid evidence_build_id")
    if manifest.get("input_resource_build_id") != resource_manifest.get("build_id"):
        raise PipelineError(
            GlobalErrorCode.EVIDENCE_BUILD_INCOMPATIBLE,
            "Evidence input_resource_build_id does not match the Resource Dataset build",
        )
    _check_version(manifest, "evidence_schema_version", compatibility.evidence_schema_versions)
    _check_version(manifest, "normalizer_version", compatibility.normalizer_versions)
    _check_version(manifest, "markdown_parser_version", compatibility.markdown_parser_versions)
    _check_version(manifest, "chunker_version", compatibility.chunker_versions)
    config_hash = manifest.get("effective_config_hash")
    if not isinstance(config_hash, str) or not HASH_RE.fullmatch(config_hash):
        raise PipelineError(GlobalErrorCode.EVIDENCE_CORPUS_INVALID, "invalid effective_config_hash")
    if compatibility.effective_config_hashes and config_hash not in compatibility.effective_config_hashes:
        raise PipelineError(GlobalErrorCode.EVIDENCE_BUILD_INCOMPATIBLE, "unsupported Evidence config hash")

    try:
        documents = load_evidence_documents(documents_path, manifest_path=evidence_manifest_path)
        chunks = load_evidence_chunks(chunks_path, manifest_path=evidence_manifest_path)
    except FileNotFoundError as exc:
        raise PipelineError(GlobalErrorCode.EVIDENCE_CORPUS_MISSING, f"Evidence output is missing: {exc}") from exc
    except (OSError, ValueError, ValidationError, json.JSONDecodeError) as exc:
        raise PipelineError(GlobalErrorCode.EVIDENCE_CORPUS_INVALID, f"invalid Evidence Corpus: {exc}") from exc

    resource_ids = {item.resource_id for item in resources}
    document_by_id = {item.evidence_id: item for item in documents}
    if len(document_by_id) != len(documents):
        raise PipelineError(GlobalErrorCode.EVIDENCE_CORPUS_INVALID, "duplicate EvidenceDocument IDs")
    if any(item.resource_id not in resource_ids for item in documents):
        raise PipelineError(GlobalErrorCode.EVIDENCE_CORPUS_INVALID, "orphan EvidenceDocument")
    for chunk in chunks:
        document = document_by_id.get(chunk.evidence_id)
        if document is None or chunk.resource_id not in resource_ids:
            raise PipelineError(GlobalErrorCode.EVIDENCE_CORPUS_INVALID, "orphan EvidenceChunk")
        if (chunk.resource_id, chunk.source_id, chunk.evidence_build_id) != (
            document.resource_id, document.source_id, document.evidence_build_id,
        ):
            raise PipelineError(GlobalErrorCode.EVIDENCE_CORPUS_INVALID, "EvidenceChunk identity mismatch")
    covered = {item.resource_id for item in chunks}
    if covered != resource_ids:
        raise PipelineError(GlobalErrorCode.EVIDENCE_CORPUS_INVALID, "Evidence coverage universe differs from Resources")
    # code_version/git commit fields are intentionally not checked: they are audit provenance only.
    return EvidenceCorpus(documents=documents, chunks=chunks, manifest=manifest)


def searchable_resource_text(resource: Resource) -> str:
    parts = [
        resource.name,
        resource.full_name or "",
        resource.description or "",
        " ".join(resource.topics),
        resource.language or "",
        resource.license,
        " ".join(resource.frameworks),
        " ".join(resource.skills),
    ]
    return " ".join(item for item in parts if item)


__all__ = [
    "EvidenceCorpus", "load_evidence_corpus", "load_resource_provenance",
    "load_resources", "read_jsonl", "searchable_resource_text",
]
