"""Deterministic construction and validation for the Phase 3 Evidence Corpus."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import shutil
import statistics
import subprocess
import tempfile
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import pyarrow.parquet as pq
from pydantic import ValidationError

from data_pipeline.collectors.github_collector import stable_hash
from data_pipeline.resource_dataset.schema import Resource

from .chunker import chunk_sections, line_number
from .config import EvidenceConfig, project_root
from .markdown_parser import parse_markdown
from .normalizer import normalize_markdown
from .schema import EvidenceChunk, EvidenceDocument


METADATA_FIELDS = (
    "repository_id", "full_name", "description", "html_url", "owner_login",
    "owner_type", "language", "topics", "stars", "forks", "watchers",
    "open_issues", "license_spdx_id", "is_archived", "is_fork",
    "default_branch", "created_at", "updated_at", "pushed_at",
)

OUTPUTS = (
    "data/evidence/documents.jsonl",
    "data/evidence/chunks.jsonl",
    "data/quarantine/evidence/invalid_document_schema.jsonl",
    "data/quarantine/evidence/failed_parse.jsonl",
    "data/quarantine/evidence/failed_linking.jsonl",
    "data/review/evidence/duplicate_content_candidates.jsonl",
    "data/review/evidence/suspiciously_short_documents.jsonl",
    "data/review/evidence/oversized_chunks.jsonl",
    "data/review/evidence/missing_expected_sources.jsonl",
    "data/reports/evidence_corpus_quality.json",
    "data/reports/evidence_corpus_report.md",
    "data/reports/sample_evidence_documents.jsonl",
    "data/reports/sample_evidence_chunks.jsonl",
    "data/metadata/evidence_build_manifest.json",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def text_hash(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def project_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root().resolve()).as_posix()
    except ValueError:
        return path.name


def root_relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return project_relative(path)


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=project_root(), check=True,
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(canonical_json(record) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_resources(path: Path) -> list[Resource]:
    records = pq.read_table(path).to_pylist()
    resources = [Resource.model_validate(record) for record in records]
    resources.sort(key=lambda item: item.resource_id)
    if len({item.resource_id for item in resources}) != len(resources):
        raise ValueError("Resource Dataset contains duplicate resource_id values")
    return resources


def _load_snapshot_files(run_dir: Path) -> tuple[list[tuple[Path, dict[str, Any]]], list[dict[str, Any]]]:
    snapshots: list[tuple[Path, dict[str, Any]]] = []
    failed: list[dict[str, Any]] = []
    for source_type in ("github_repository", "github_readme"):
        directory = run_dir / source_type
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.json")):
            raw_bytes = b""
            try:
                raw_bytes = path.read_bytes()
                wrapper = json.loads(raw_bytes.decode("utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raw_file_hash = hashlib.sha256(raw_bytes).hexdigest() if raw_bytes else None
                failed.append({"path": project_relative(path), "raw_file_hash": raw_file_hash, "error_type": type(exc).__name__, "error": str(exc)})
                continue
            snapshots.append((path, wrapper))
    return snapshots, failed


def _validate_wrapper(path: Path, wrapper: dict[str, Any], run_id: str) -> None:
    required = {
        "collector_version", "schema_version", "run_id", "retrieved_at", "source_type",
        "source_id", "source_url", "source_hash", "payload", "provenance",
    }
    missing = required - wrapper.keys()
    if missing:
        raise ValueError(f"missing raw wrapper fields: {sorted(missing)}")
    if wrapper["run_id"] != run_id:
        raise ValueError("raw wrapper run_id does not match input run directory")
    if wrapper["source_type"] not in {"github_repository", "github_readme"}:
        raise ValueError("unsupported raw source_type")
    if stable_hash(wrapper["payload"]) != wrapper["source_hash"]:
        raise ValueError("raw wrapper source_hash does not match payload")
    datetime.fromisoformat(str(wrapper["retrieved_at"]).replace("Z", "+00:00"))
    if not path.name.endswith(".json"):
        raise ValueError("snapshot must be JSON")


def _index_snapshots(
    run_dir: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    run_id = run_dir.name
    items, failed_parse = _load_snapshot_files(run_dir)
    metadata: dict[str, dict[str, Any]] = {}
    readmes: dict[str, dict[str, Any]] = {}
    failed_schema: list[dict[str, Any]] = []
    canonical_hashes: list[str] = sorted(
        f"failed:{item['raw_file_hash']}" for item in failed_parse if item.get("raw_file_hash")
    )
    for path, wrapper in items:
        canonical_hashes.append(digest(wrapper))
        try:
            _validate_wrapper(path, wrapper, run_id)
        except (TypeError, ValueError) as exc:
            failed_schema.append({"path": project_relative(path), "error_type": type(exc).__name__, "error": str(exc)})
            continue
        target = metadata if wrapper["source_type"] == "github_repository" else readmes
        key = str(wrapper["source_id"])
        if key in target:
            if canonical_json(target[key]) != canonical_json(wrapper):
                failed_schema.append({"path": project_relative(path), "source_id": key, "error_type": "DuplicateWrapper", "error": "conflicting wrappers for one source_id"})
            continue
        target[key] = wrapper
    return metadata, readmes, failed_parse, failed_schema, sorted(canonical_hashes)


def _resource_content_hash(resources: list[Resource]) -> str:
    return digest([item.model_dump(mode="json") for item in resources])


def evidence_build_id_for(
    resources: list[Resource], phase2_manifest: dict[str, Any], raw_run_id: str,
    raw_wrapper_hashes: list[str], config: EvidenceConfig,
) -> tuple[str, str, str]:
    resource_hash = _resource_content_hash(resources)
    effective_config = {key: value for key, value in config.raw.items() if key not in {"paths", "inputs"}}
    effective_config_hash = digest(effective_config)
    phase2_semantics = {
        "build_id": phase2_manifest.get("build_id"),
        "schema_version": phase2_manifest.get("schema_version"),
        "normalizer_version": phase2_manifest.get("normalizer_version"),
        "dataset_scale": phase2_manifest.get("dataset_scale"),
        "input_source_hashes": phase2_manifest.get("input_source_hashes", []),
    }
    material = {
        "input_resource_build": phase2_semantics,
        "resource_dataset_content_hash": resource_hash,
        "raw_run_id": raw_run_id,
        "raw_wrapper_canonical_hashes": raw_wrapper_hashes,
        "evidence_schema_version": config.evidence["schema_version"],
        "normalizer_version": config.evidence["normalizer_version"],
        "markdown_parser_version": config.evidence["markdown_parser_version"],
        "chunker_version": config.evidence["chunker_version"],
        "effective_config_hash": effective_config_hash,
    }
    return f"evidence_build_{digest(material)[:16]}", resource_hash, effective_config_hash


def _metadata_values(payload: dict[str, Any]) -> dict[str, Any]:
    owner = payload.get("owner") or {}
    license_data = payload.get("license") or {}
    return {
        "repository_id": payload.get("id"),
        "full_name": payload.get("full_name"),
        "description": payload.get("description"),
        "html_url": payload.get("html_url"),
        "owner_login": owner.get("login"),
        "owner_type": owner.get("type"),
        "language": payload.get("language"),
        "topics": sorted(str(item) for item in (payload.get("topics") or [])),
        "stars": payload.get("stargazers_count"),
        "forks": payload.get("forks_count"),
        "watchers": payload.get("watchers_count"),
        "open_issues": payload.get("open_issues_count"),
        "license_spdx_id": license_data.get("spdx_id"),
        "is_archived": payload.get("archived"),
        "is_fork": payload.get("fork"),
        "default_branch": payload.get("default_branch"),
        "created_at": payload.get("created_at"),
        "updated_at": payload.get("updated_at"),
        "pushed_at": payload.get("pushed_at"),
    }


def serialize_metadata(payload: dict[str, Any]) -> str:
    values = _metadata_values(payload)
    return "\n".join(f"{field}: {canonical_json(values[field])}" for field in METADATA_FIELDS)


def _document(
    resource: Resource, wrapper: dict[str, Any], evidence_type: str,
    text: str, build_id: str, config: EvidenceConfig, raw_path: Path,
    discovery_provenance: dict[str, Any], warnings: list[str],
) -> EvidenceDocument:
    suffix = "readme" if evidence_type == "github_readme" else "metadata"
    source_id = f"{resource.resource_id}:{suffix}"
    content = text_hash(text)
    evidence_id = f"ev_{digest([resource.resource_id, evidence_type, content])[:16]}"
    payload = wrapper["payload"]
    api_response = payload.get("api_response", {}) if isinstance(payload, dict) else {}
    metadata = {
        "raw_snapshot_path": project_relative(raw_path),
        "raw_provenance": wrapper.get("provenance", {}),
        "resource_discovery_provenance": discovery_provenance,
        "resource_record_version": resource.record_version,
        "normalization_warnings": warnings,
    }
    if evidence_type == "github_readme":
        metadata.update({
            "readme_path": api_response.get("path"),
            "readme_blob_sha": api_response.get("sha"),
            "download_url": api_response.get("download_url"),
        })
    return EvidenceDocument(
        evidence_id=evidence_id, document_key=source_id, evidence_build_id=build_id,
        resource_id=resource.resource_id, evidence_source_type=evidence_type,
        source_type=resource.source_type, source_id=source_id,
        raw_source_id=str(wrapper["source_id"]), source_url=str(wrapper["source_url"]),
        raw_source_hash=str(wrapper["source_hash"]),
        title=f"{resource.full_name or resource.name} {'README' if suffix == 'readme' else 'repository metadata'}",
        text=text, text_format="markdown" if suffix == "readme" else "structured_metadata",
        content_hash=content, retrieved_at=str(wrapper["retrieved_at"]),
        published_at=None, updated_at=payload.get("updated_at") if suffix == "metadata" else None,
        source_reliability=resource.source_reliability,
        source_reliability_method=resource.source_reliability_method,
        extraction_method="raw_snapshot" if suffix == "readme" else "api_metadata_serialization",
        run_id=str(wrapper["run_id"]), collector_version=str(wrapper["collector_version"]),
        snapshot_schema_version=str(wrapper["schema_version"]),
        evidence_schema_version=str(config.evidence["schema_version"]),
        normalizer_version=str(config.evidence["normalizer_version"]),
        markdown_parser_version=str(config.evidence["markdown_parser_version"]),
        metadata=metadata,
    )


def _chunk_record(
    document: EvidenceDocument, index: int, start: int, end: int,
    section_path: list[str], section_title: str, section_level: int,
    contains_code: bool, contains_table: bool, config: EvidenceConfig,
    *, overlap_applied: bool = False, new_nonwhitespace_coverage: int | None = None,
) -> EvidenceChunk:
    text = document.text[start:end]
    chunk_id = f"ch_{digest([document.evidence_id, section_path, index, start, end, text])[:16]}"
    return EvidenceChunk(
        chunk_id=chunk_id, evidence_id=document.evidence_id,
        evidence_build_id=document.evidence_build_id, resource_id=document.resource_id,
        evidence_source_type=document.evidence_source_type, source_type=document.source_type,
        source_id=document.source_id, raw_source_id=document.raw_source_id,
        source_url=document.source_url, raw_source_hash=document.raw_source_hash,
        section_path=section_path, section_title=section_title, section_level=section_level,
        chunk_index=index, source_start_line=line_number(document.text, start),
        source_end_line=line_number(document.text, end, end=True),
        source_start_char=start, source_end_char=end, text=text, char_count=len(text),
        content_hash=text_hash(text), contains_code=contains_code, contains_table=contains_table,
        retrieved_at=document.retrieved_at, source_reliability=document.source_reliability,
        source_reliability_method=document.source_reliability_method, run_id=document.run_id,
        collector_version=document.collector_version,
        snapshot_schema_version=document.snapshot_schema_version,
        evidence_schema_version=document.evidence_schema_version,
        normalizer_version=document.normalizer_version,
        markdown_parser_version=document.markdown_parser_version,
        chunker_version=str(config.evidence["chunker_version"]), metadata={
            "overlap_applied": overlap_applied,
            "new_nonwhitespace_coverage": new_nonwhitespace_coverage,
        },
    )


def chunks_for_document(document: EvidenceDocument, config: EvidenceConfig) -> list[EvidenceChunk]:
    if document.evidence_source_type == "repo_metadata":
        return [_chunk_record(document, 0, 0, len(document.text), ["__metadata__"], "__metadata__", 0, False, False, config)]
    parsed = parse_markdown(document.text)
    spans = chunk_sections(document.text, parsed.sections, config)
    unique: list[Any] = []
    seen: set[tuple[int, int, str]] = set()
    for span in spans:
        key = (span.start, span.end, document.text[span.start:span.end])
        if key not in seen:
            seen.add(key)
            unique.append(span)
    return [
        _chunk_record(document, index, span.start, span.end, span.section.section_path,
                      span.section.title, span.section.level, span.contains_code,
                      span.contains_table, config, overlap_applied=span.overlap_applied,
                      new_nonwhitespace_coverage=span.new_nonwhitespace_coverage)
        for index, span in enumerate(unique)
    ]


def _duplicate_candidates(chunks: list[EvidenceChunk]) -> list[dict[str, Any]]:
    groups: dict[str, list[EvidenceChunk]] = defaultdict(list)
    for chunk in chunks:
        groups[chunk.content_hash].append(chunk)
    return [
        {
            "content_hash": key,
            "chunk_ids": [item.chunk_id for item in sorted(group, key=lambda x: x.chunk_id)],
            "resource_ids": sorted({item.resource_id for item in group}),
            "reason": "identical_content_at_distinct_source_positions",
            "decision": "review",
        }
        for key, group in sorted(groups.items())
        if len(group) > 1 and len({(x.evidence_id, x.source_start_char, x.source_end_char) for x in group}) > 1
    ]


HEADING_ONLY_RE = re.compile(r"^[ \t]{0,3}#{1,6}(?:[ \t]+.*)?[ \t]*#*[ \t]*$")
SETEXT_UNDERLINE_RE = re.compile(r"^[ \t]{0,3}(?:=+|-+)[ \t]*$")
STRUCTURAL_ONLY_RE = re.compile(
    r"^[ \t]{0,3}(?:(?:\*[ \t]*){3,}|(?:-[ \t]*){3,}|(?:_[ \t]*){3,})[ \t]*$"
)
STRUCTURAL_HTML_ONLY_RE = re.compile(
    r"^[ \t]*(?:<!--.*-->|</?(?:div|details|center|p|br|hr)(?:[ \t]+[^>]*)?>)[ \t]*$",
    re.IGNORECASE,
)


def _is_heading_or_structural_only(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return True
    if all(STRUCTURAL_ONLY_RE.fullmatch(line) or STRUCTURAL_HTML_ONLY_RE.fullmatch(line) for line in lines):
        return True
    if len(lines) == 1:
        return bool(HEADING_ONLY_RE.fullmatch(lines[0]))
    return len(lines) == 2 and bool(lines[0].strip()) and bool(SETEXT_UNDERLINE_RE.fullmatch(lines[1]))


def _span_quality_counts(
    documents: list[EvidenceDocument], chunks: list[EvidenceChunk], minimum: int, maximum: int,
) -> dict[str, int]:
    document_by_id = {item.evidence_id: item for item in documents}
    groups: dict[tuple[str, tuple[str, ...]], list[EvidenceChunk]] = defaultdict(list)
    for chunk in chunks:
        if chunk.evidence_source_type == "github_readme":
            groups[(chunk.evidence_id, tuple(chunk.section_path))].append(chunk)

    fully_contained = 0
    insufficient_new = 0
    for (evidence_id, _), group in groups.items():
        parent = document_by_id[evidence_id]
        ordered = sorted(group, key=lambda item: item.chunk_index)
        covered: list[tuple[int, int]] = []
        for position, chunk in enumerate(ordered):
            if any(
                other.chunk_id != chunk.chunk_id
                and other.source_start_char <= chunk.source_start_char
                and other.source_end_char >= chunk.source_end_char
                for other in ordered
            ):
                fully_contained += 1
            overlaps_prior = any(
                left < chunk.source_end_char and right > chunk.source_start_char
                for left, right in covered
            )
            new_nonwhitespace = sum(
                not parent.text[offset].isspace()
                and not any(left <= offset < right for left, right in covered)
                for offset in range(chunk.source_start_char, chunk.source_end_char)
            )
            if position and overlaps_prior and new_nonwhitespace < minimum:
                insufficient_new += 1
            covered.append((chunk.source_start_char, chunk.source_end_char))

    return {
        "fully_contained_chunk_spans": fully_contained,
        "chunks_with_insufficient_new_coverage": insufficient_new,
        "heading_or_structural_only_chunks": sum(
            chunk.evidence_source_type == "github_readme" and _is_heading_or_structural_only(chunk.text)
            for chunk in chunks
        ),
        "oversized_ordinary_prose_chunks": sum(
            chunk.evidence_source_type == "github_readme"
            and chunk.char_count > maximum
            and not chunk.contains_code
            and not chunk.contains_table
            for chunk in chunks
        ),
    }


def validate_records(
    documents: list[EvidenceDocument], chunks: list[EvidenceChunk], manifest: dict[str, Any],
    resources: list[Resource] | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    document_by_id = {item.evidence_id: item for item in documents}
    if len(document_by_id) != len(documents):
        errors.append("duplicate evidence_id")
    if len({item.chunk_id for item in chunks}) != len(chunks):
        errors.append("duplicate chunk_id")
    build_id = manifest.get("evidence_build_id")
    if any(item.evidence_build_id != build_id for item in documents + chunks):
        errors.append("manifest/build identity mismatch")
    for document in documents:
        expected_evidence = f"ev_{digest([document.resource_id, document.evidence_source_type, document.content_hash])[:16]}"
        if document.evidence_id != expected_evidence or text_hash(document.text) != document.content_hash:
            errors.append(f"document identity/hash mismatch: {document.evidence_id}")
    per_document_indices: dict[str, list[int]] = defaultdict(list)
    for chunk in chunks:
        parent = document_by_id.get(chunk.evidence_id)
        if parent is None:
            errors.append(f"orphan chunk: {chunk.chunk_id}")
            continue
        if (chunk.resource_id, chunk.source_id, chunk.raw_source_id, chunk.raw_source_hash) != (
            parent.resource_id, parent.source_id, parent.raw_source_id, parent.raw_source_hash,
        ):
            errors.append(f"chunk provenance differs from parent: {chunk.chunk_id}")
        if parent.text[chunk.source_start_char:chunk.source_end_char] != chunk.text:
            errors.append(f"non-contiguous or invalid source slice: {chunk.chunk_id}")
        expected_chunk = f"ch_{digest([chunk.evidence_id, chunk.section_path, chunk.chunk_index, chunk.source_start_char, chunk.source_end_char, chunk.text])[:16]}"
        if chunk.chunk_id != expected_chunk or text_hash(chunk.text) != chunk.content_hash:
            errors.append(f"chunk identity/hash mismatch: {chunk.chunk_id}")
        if line_number(parent.text, chunk.source_start_char) != chunk.source_start_line or line_number(parent.text, chunk.source_end_char, end=True) != chunk.source_end_line:
            errors.append(f"chunk line offsets mismatch: {chunk.chunk_id}")
        per_document_indices[chunk.evidence_id].append(chunk.chunk_index)
    for evidence_id, indices in per_document_indices.items():
        if sorted(indices) != list(range(len(indices))):
            errors.append(f"non-contiguous chunk indexes: {evidence_id}")
    for document in documents:
        if document.evidence_id not in per_document_indices:
            errors.append(f"document has no chunks: {document.evidence_id}")
    minimum = int(manifest.get("min_chunk_chars", 1))
    maximum = int(manifest.get("max_chunk_chars", 2**31 - 1))
    span_quality = _span_quality_counts(documents, chunks, minimum, maximum)
    if span_quality["fully_contained_chunk_spans"]:
        errors.append("fully-contained chunk spans are not allowed")
    if span_quality["chunks_with_insufficient_new_coverage"]:
        errors.append("overlap chunks have insufficient new non-whitespace coverage")
    if span_quality["heading_or_structural_only_chunks"]:
        errors.append("heading-only or structural-only chunks are not allowed")
    if span_quality["oversized_ordinary_prose_chunks"]:
        errors.append("oversized ordinary prose chunks are not allowed")
    if resources is not None:
        resource_ids = {item.resource_id for item in resources}
        resource_by_id = {item.resource_id: item for item in resources}
        if any(item.resource_id not in resource_ids for item in documents):
            errors.append("orphan document")
        for document in documents:
            resource = resource_by_id.get(document.resource_id)
            if resource and (document.source_type, document.source_reliability, document.source_reliability_method) != (
                resource.source_type, resource.source_reliability, resource.source_reliability_method,
            ):
                errors.append(f"source authority prior not inherited: {document.evidence_id}")
        if manifest.get("input_resource_count") != len(resources):
            errors.append("manifest resource count mismatch")
    if errors:
        raise ValueError("Evidence Corpus validation failed:\n- " + "\n- ".join(errors))
    return {
        "valid": True, "documents": len(documents), "chunks": len(chunks),
        "orphan_documents": 0, "orphan_chunks": 0,
        "provenance_completeness": 1.0,
        **span_quality,
    }


def _quality(
    documents: list[EvidenceDocument], chunks: list[EvidenceChunk], manifest: dict[str, Any],
    duplicate_review: list[dict[str, Any]], short_review: list[dict[str, Any]],
    oversized_review: list[dict[str, Any]], missing_review: list[dict[str, Any]],
    failed_parse: list[dict[str, Any]], failed_schema: list[dict[str, Any]],
    failed_linking: list[dict[str, Any]], validation: dict[str, Any],
) -> dict[str, Any]:
    source_types = Counter(item.evidence_source_type for item in documents)
    chunk_sources = Counter(item.evidence_source_type for item in chunks)
    per_resource = Counter(item.resource_id for item in chunks)
    chars = [item.char_count for item in chunks]
    sorted_chars = sorted(chars)
    minimum = int(manifest.get("min_chunk_chars", 1))
    return {
        "evidence_build_id": manifest["evidence_build_id"],
        "dataset_scale": manifest["dataset_scale"],
        "input_resource_count": manifest["input_resource_count"],
        "document_count": len(documents), "chunk_count": len(chunks),
        "document_source_distribution": dict(sorted(source_types.items())),
        "chunk_source_distribution": dict(sorted(chunk_sources.items())),
        "per_resource_chunk_counts": dict(sorted(per_resource.items())),
        "chunk_characters": {
            "min": min(chars) if chars else None, "max": max(chars) if chars else None,
            "mean": round(sum(chars) / len(chars), 3) if chars else None,
            "median": round(float(statistics.median(chars)), 3) if chars else None,
            "p95": sorted_chars[math.ceil(0.95 * len(sorted_chars)) - 1] if sorted_chars else None,
        },
        "chunks_below_min_chunk_chars": sum(item.char_count < minimum for item in chunks),
        "chunks_with_code": sum(item.contains_code for item in chunks),
        "chunks_with_tables": sum(item.contains_table for item in chunks),
        "duplicate_content_review_groups": len(duplicate_review),
        "suspiciously_short_documents": len(short_review),
        "oversized_chunks": len(oversized_review),
        "missing_expected_sources": len(missing_review),
        "failed_parse": len(failed_parse), "invalid_document_schema": len(failed_schema),
        "failed_linking": len(failed_linking), **validation,
        "source_authority_priors": {
            "method": "heuristic_v0", "calibrated": False,
            "interpretation": "Non-calibrated source-authority priors; not trustworthiness, evidence strength, quality, or confidence scores.",
            "distribution": dict(sorted(Counter(item.source_reliability for item in documents).items())),
        },
        "known_limitations": [
            f"Corpus from {manifest['input_resource_count']} public GitHub repositories.",
            "README and repository metadata are the only evidence types in this phase.",
            "No semantic inference, retrieval, claim verification, or trust calibration is performed.",
        ],
    }


def _report(quality: dict[str, Any], manifest: dict[str, Any]) -> str:
    dist = quality["document_source_distribution"]
    return f"""# Evidence Corpus Quality Report

## Build summary

- Evidence build ID: `{quality['evidence_build_id']}`
- Input Resource build ID: `{manifest['input_resource_build_id']}`
- Input Resource Dataset: `{manifest['resources_path']}`
- Input raw run: `{manifest['input_run']}`
- Dataset scale: `{quality['dataset_scale']}`
- Resources: {quality['input_resource_count']}
- Documents: {quality['document_count']}
- Chunks: {quality['chunk_count']}

## Coverage and evidence types

- Repository metadata documents: {dist.get('repo_metadata', 0)}
- GitHub README documents: {dist.get('github_readme', 0)}
- Missing expected sources: {quality['missing_expected_sources']}
- Orphan documents: {quality['orphan_documents']}
- Orphan chunks: {quality['orphan_chunks']}
- Provenance completeness: {quality['provenance_completeness']:.0%}

Document coverage is derived from the explicit Resource Dataset and frozen raw run. The builder does not synthesize or repair absent evidence.

## Chunking summary

- Chunks containing fenced code: {quality['chunks_with_code']}
- Chunks containing Markdown tables: {quality['chunks_with_tables']}
- Minimum / median / mean / p95 / maximum characters: {quality['chunk_characters']['min']} / {quality['chunk_characters']['median']} / {quality['chunk_characters']['mean']} / {quality['chunk_characters']['p95']} / {quality['chunk_characters']['max']}
- Chunks below `min_chunk_chars={manifest['min_chunk_chars']}`: {quality['chunks_below_min_chunk_chars']}
- Fully-contained chunk spans: {quality['fully_contained_chunk_spans']}
- Overlap chunks with insufficient new non-whitespace coverage: {quality['chunks_with_insufficient_new_coverage']}
- Heading-only or structural-only chunks: {quality['heading_or_structural_only_chunks']}
- Oversized ordinary prose chunks: {quality['oversized_ordinary_prose_chunks']}
- Oversized chunks retained for review: {quality['oversized_chunks']}

README chunks are exact contiguous slices of normalized EvidenceDocument text. Overlap is boundary-aware and is not applied to code blocks or tables.
`min_chunk_chars` is a prototype soft baseline for retrieval-unit quality, not a research result. A genuinely short section may remain when it cannot be merged safely across a section or atomic boundary.

## Review and quarantine

- Identical-content review groups: {quality['duplicate_content_review_groups']}
- Suspiciously short documents: {quality['suspiciously_short_documents']}
- Failed parsing records: {quality['failed_parse']}
- Invalid schema records: {quality['invalid_document_schema']}
- Failed linking records: {quality['failed_linking']}

Identical text at different source positions is retained. Only duplicate-generation artifacts with the same document, source span, and content are suppressed.

## Source authority prior

`heuristic_v0` values are non-calibrated source-authority priors. They are not trustworthiness, evidence-strength, quality, calibrated-confidence, or verification scores.

## Scope and limitations

This corpus is derived from {quality['input_resource_count']} public GitHub repositories. It contains only README and structured repository metadata evidence. Phase 3 performs no semantic enrichment, KG construction, embedding, retrieval, RAG, claim verification, evidence fusion, uncertainty estimation, calibration, abstention, recommendation, or UI work.
"""


def _stage_path(stage: Path, relative: str) -> Path:
    return stage / Path(relative)


def _atomic_publish(stage: Path, root: Path, replacer: Callable[[str | os.PathLike[str], str | os.PathLike[str]], None] = os.replace) -> None:
    backup = Path(tempfile.mkdtemp(prefix="backup-", dir=stage.parent))
    replaced: list[str] = []
    try:
        for relative in OUTPUTS:
            source = _stage_path(stage, relative)
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                backup_target = backup / relative
                backup_target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, backup_target)
            replacer(source, target)
            replaced.append(relative)
    except Exception:
        for relative in reversed(replaced):
            target = root / relative
            backup_target = backup / relative
            if backup_target.exists():
                os.replace(backup_target, target)
            elif target.exists():
                target.unlink()
        raise
    finally:
        shutil.rmtree(backup, ignore_errors=True)


def build_corpus(
    resources_path: Path, input_run: Path, config: EvidenceConfig,
    *, publish: bool = True, replacer: Callable[[str | os.PathLike[str], str | os.PathLike[str]], None] = os.replace,
    output_root: Path | None = None,
) -> dict[str, Any]:
    root = (output_root or project_root()).resolve()
    resources_path, input_run = resources_path.resolve(), input_run.resolve()
    if not resources_path.is_file() or not input_run.is_dir():
        raise FileNotFoundError("explicit Resource Dataset and raw input run are required")
    phase2_manifest_path = config.input_path("resource_manifest")
    phase2_manifest = json.loads(phase2_manifest_path.read_text(encoding="utf-8"))
    resources = load_resources(resources_path)
    if phase2_manifest.get("final_resources") != len(resources):
        raise ValueError("Phase 2 manifest does not match Resource Dataset")
    metadata, readmes, failed_parse, failed_schema, wrapper_hashes = _index_snapshots(input_run)
    build_id, resource_hash, config_hash = evidence_build_id_for(resources, phase2_manifest, input_run.name, wrapper_hashes, config)

    documents: list[EvidenceDocument] = []
    failed_linking: list[dict[str, Any]] = []
    missing_review: list[dict[str, Any]] = []
    raw_paths = {
        (source_type, path.stem): path
        for source_type in ("github_repository", "github_readme")
        for path in (input_run / source_type).glob("*.json")
    }
    linked_ids = {item.raw_source_id for item in resources}
    for raw_id in sorted((set(metadata) | set(readmes)) - linked_ids):
        failed_linking.append({"raw_source_id": raw_id, "error": "raw source is not present in Resource Dataset"})
    for resource in resources:
        raw_id = resource.raw_source_id
        metadata_wrapper = metadata.get(raw_id)
        readme_wrapper = readmes.get(raw_id)
        if metadata_wrapper is None:
            missing_review.append({"resource_id": resource.resource_id, "raw_source_id": raw_id, "missing": "repo_metadata", "decision": "review"})
            continue
        payload = metadata_wrapper["payload"]
        if payload.get("full_name") != raw_id or f"github:{payload.get('id')}" != resource.resource_id:
            failed_linking.append({"resource_id": resource.resource_id, "raw_source_id": raw_id, "error": "metadata identity conflict"})
            continue
        metadata_filename = raw_id.replace("/", "__").replace(":", "_")
        if config.raw["sources"]["include_repo_metadata"]:
            try:
                metadata_text = serialize_metadata(payload)
                documents.append(_document(resource, metadata_wrapper, "repo_metadata", metadata_text, build_id, config,
                                           raw_paths[("github_repository", metadata_filename)], metadata_wrapper.get("provenance", {}), []))
            except (KeyError, TypeError, ValidationError, ValueError) as exc:
                failed_schema.append({"resource_id": resource.resource_id, "evidence_source_type": "repo_metadata", "error_type": type(exc).__name__, "error": str(exc)})
        if not config.raw["sources"]["include_github_readme"]:
            continue
        if readme_wrapper is None:
            missing_review.append({"resource_id": resource.resource_id, "raw_source_id": raw_id, "missing": "github_readme", "decision": "review"})
            continue
        if readme_wrapper["run_id"] != metadata_wrapper["run_id"] or readme_wrapper["source_id"] != raw_id:
            failed_linking.append({"resource_id": resource.resource_id, "raw_source_id": raw_id, "error": "README/metadata identity conflict"})
            continue
        try:
            raw_text = readme_wrapper["payload"]["decoded_text"]
            if not isinstance(raw_text, str):
                raise TypeError("README decoded_text must be a string")
            normalized = normalize_markdown(raw_text, config)
            parsed = parse_markdown(normalized)
            readme_filename = raw_id.replace("/", "__").replace(":", "_")
            documents.append(_document(resource, readme_wrapper, "github_readme", normalized, build_id, config,
                                       raw_paths[("github_readme", readme_filename)], metadata_wrapper.get("provenance", {}), parsed.warnings))
        except (KeyError, TypeError, ValidationError, ValueError) as exc:
            failed_schema.append({"resource_id": resource.resource_id, "evidence_source_type": "github_readme", "error_type": type(exc).__name__, "error": str(exc)})

    documents.sort(key=lambda item: (item.resource_id, item.evidence_source_type, item.evidence_id))
    chunks = [chunk for document in documents for chunk in chunks_for_document(document, config)]
    chunks.sort(key=lambda item: (item.resource_id, item.evidence_source_type, item.evidence_id, item.chunk_index))
    duplicate_review = _duplicate_candidates(chunks)
    short_limit = int(config.review["short_document_chars"])
    short_review = [{"evidence_id": item.evidence_id, "resource_id": item.resource_id, "char_count": len(item.text), "decision": "review"} for item in documents if len(item.text) < short_limit]
    oversized_limit = int(config.review["oversized_chunk_chars"])
    oversized_review = [{"chunk_id": item.chunk_id, "evidence_id": item.evidence_id, "char_count": item.char_count, "contains_code": item.contains_code, "contains_table": item.contains_table, "decision": "review"} for item in chunks if item.char_count > oversized_limit]

    manifest = {
        "evidence_build_id": build_id, "built_at": utc_now(),
        "dataset_scale": config.evidence["dataset_scale"],
        "input_resource_build_id": phase2_manifest["build_id"],
        "input_resource_schema_version": phase2_manifest.get("schema_version"),
        "input_resource_normalizer_version": phase2_manifest.get("normalizer_version"),
        "resources_path": root_relative(resources_path, root), "input_run": project_relative(input_run),
        "raw_run_id": input_run.name, "input_resource_count": len(resources),
        "resource_dataset_content_hash": resource_hash,
        "raw_wrapper_canonical_hashes": wrapper_hashes,
        "effective_config_hash": config_hash,
        "evidence_schema_version": config.evidence["schema_version"],
        "normalizer_version": config.evidence["normalizer_version"],
        "markdown_parser_version": config.evidence["markdown_parser_version"],
        "chunker_version": config.evidence["chunker_version"],
        "min_chunk_chars": int(config.chunking["min_chunk_chars"]),
        "max_chunk_chars": int(config.chunking["max_chunk_chars"]),
        "oversized_chunk_chars": int(config.review["oversized_chunk_chars"]),
        "document_count": len(documents), "chunk_count": len(chunks),
        "source_reliability_method": "heuristic_v0",
        "code_version": "summer_s3_5_p0",
        "git_commit": git_commit(),
        "manifest_role": "atomic_build_commit_marker",
    }
    validation = validate_records(documents, chunks, manifest, resources)
    quality = _quality(documents, chunks, manifest, duplicate_review, short_review, oversized_review,
                       missing_review, failed_parse, failed_schema, failed_linking, validation)

    staging_parent = config.path("staging_root")
    staging_parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f"{build_id}-", dir=staging_parent))
    try:
        write_jsonl(_stage_path(stage, OUTPUTS[0]), [item.model_dump(mode="json") for item in documents])
        write_jsonl(_stage_path(stage, OUTPUTS[1]), [item.model_dump(mode="json") for item in chunks])
        for relative, records in (
            (OUTPUTS[2], failed_schema), (OUTPUTS[3], failed_parse), (OUTPUTS[4], failed_linking),
            (OUTPUTS[5], duplicate_review), (OUTPUTS[6], short_review),
            (OUTPUTS[7], oversized_review), (OUTPUTS[8], missing_review),
        ):
            write_jsonl(_stage_path(stage, relative), records)
        quality_path = _stage_path(stage, OUTPUTS[9])
        quality_path.parent.mkdir(parents=True, exist_ok=True)
        quality_path.write_text(json.dumps(quality, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        report_path = _stage_path(stage, OUTPUTS[10])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(_report(quality, manifest), encoding="utf-8")
        rng = random.Random(int(config.raw["random_seed"]))
        doc_sample = sorted(rng.sample(documents, min(5, len(documents))), key=lambda item: item.evidence_id)
        chunk_sample = sorted(rng.sample(chunks, min(10, len(chunks))), key=lambda item: item.chunk_id)
        write_jsonl(_stage_path(stage, OUTPUTS[11]), [item.model_dump(mode="json") for item in doc_sample])
        write_jsonl(_stage_path(stage, OUTPUTS[12]), [item.model_dump(mode="json") for item in chunk_sample])
        manifest_path = _stage_path(stage, OUTPUTS[13])
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        staged_docs = [EvidenceDocument.model_validate(item) for item in read_jsonl(_stage_path(stage, OUTPUTS[0]))]
        staged_chunks = [EvidenceChunk.model_validate(item) for item in read_jsonl(_stage_path(stage, OUTPUTS[1]))]
        staged_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_records(staged_docs, staged_chunks, staged_manifest, resources)
        if publish:
            _atomic_publish(stage, root, replacer)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return {**validation, "evidence_build_id": build_id, "quality": quality, "manifest": manifest}
