"""Validation and reporting for frozen S3.5 stage outputs."""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from data_pipeline.collectors.github_collector import load_baseline_snapshots
from data_pipeline.evidence.schema import EvidenceChunk, EvidenceDocument
from data_pipeline.resource_dataset.config import load_config as load_resource_config
from data_pipeline.resource_dataset.core import _contains_term, _relevance_text
from data_pipeline.resource_dataset.curation import CurationManifest
from data_pipeline.resource_dataset.schema import Resource


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _ids(path: Path) -> set[str]:
    return {str(item["resource_id"]) for item in _jsonl(path) if "resource_id" in item}


def _distribution(values: list[Any]) -> dict[str, int]:
    return dict(sorted(Counter("unknown" if value is None else str(value) for value in values).items()))


def _performance(root: Path) -> dict[str, float | None]:
    values: dict[str, float | None] = {
        "collection_seconds": None,
        "resource_build_seconds": None,
        "evidence_build_seconds": None,
        "resource_validation_seconds": None,
        "evidence_validation_seconds": None,
        "expansion_validation_seconds": None,
        "validation_seconds": None,
        "smoke_seconds": None,
    }
    path = root / "data" / "metadata" / "s3_5_performance.json"
    if not path.is_file():
        return values
    stored = json.loads(path.read_text(encoding="utf-8"))
    for key in values:
        value = stored.get(key)
        if isinstance(value, (int, float)) and value >= 0:
            values[key] = round(float(value), 3)
    validation_parts = [
        values["resource_validation_seconds"],
        values["evidence_validation_seconds"],
        values["expansion_validation_seconds"],
    ]
    if any(value is not None for value in validation_parts):
        values["validation_seconds"] = round(sum(value or 0.0 for value in validation_parts), 3)
    return values


def weak_relevance_candidates(
    resources: list[Resource], raw_repositories: dict[int, dict[str, Any]],
    raw_readmes: dict[int, dict[str, Any]], *, output_root: Path,
) -> list[dict[str, Any]]:
    config = load_resource_config(output_root=output_root)
    relevance = config.vocabulary["domain_relevance"]
    terms = [_relevance_text(term) for term in relevance["terms"]]
    weights = {key: float(value) for key, value in relevance["weights"].items()}
    records: list[dict[str, Any]] = []
    for resource in sorted(resources, key=lambda item: item.resource_id):
        numeric_id = int(resource.resource_id.split(":", 1)[1])
        wrapper = raw_repositories[numeric_id]
        readme = raw_readmes.get(numeric_id)
        metadata = wrapper["payload"]
        texts = {
            "topics": _relevance_text(" ".join(str(item) for item in metadata.get("topics") or [])),
            "description": _relevance_text(metadata.get("description") or ""),
            "readme": _relevance_text(((readme or {}).get("payload") or {}).get("decoded_text") or ""),
        }
        matched_terms = {
            field: [term for term in terms if _contains_term(text, term)]
            for field, text in texts.items()
        }
        matched_fields = [field for field in ("topics", "description", "readme") if matched_terms[field]]
        reasons: list[str] = []
        if resource.domain_relevance_score <= 0.30:
            reasons.append("score_at_or_below_0_30")
        if matched_fields == ["readme"]:
            reasons.append("readme_only_match")
        if (
            matched_fields == ["description"]
            and len(matched_terms["description"]) == 1
            and len(matched_terms["description"][0]) <= 3
        ):
            reasons.append("single_short_description_term")
        if not reasons:
            continue
        provenance = wrapper.get("provenance") or {}
        discovery_matches = provenance.get("discovery_matches") or [provenance]
        groups = sorted({
            str(item["query_group"])
            for item in discovery_matches
            if item.get("query_group")
        })
        records.append({
            "resource_id": resource.resource_id,
            "repository": resource.full_name,
            "score": resource.domain_relevance_score,
            "query_groups": groups,
            "matched_terms": matched_terms,
            "matched_field": matched_fields,
            "contribution": {field: weights[field] for field in matched_fields},
            "review_reasons": reasons,
        })
    return records


def _report(quality: dict[str, Any]) -> str:
    performance = quality["performance"]
    return f"""# S3.5 Resource Expansion Quality Report

## Frozen input identity

- Raw run ID: `{quality['raw_run_id']}`
- Resource build ID: `{quality.get('resource_build_id')}`
- Evidence build ID: `{quality.get('evidence_build_id')}`
- Git commit: `{quality.get('git_commit')}`

## Coverage and integrity

- Resources: {quality['resource_count']}
- Raw metadata coverage: {quality['metadata_coverage']:.1%}
- Raw README snapshot coverage: {quality['raw_readme_coverage']:.1%}
- README Evidence coverage: {quality['readme_coverage']:.1%}
- Any-evidence coverage: {quality['any_evidence_coverage']:.1%}
- Preserved baseline IDs: {quality['baseline_ids_preserved']}/{quality['baseline_id_count']}
- Duplicate numeric IDs: {quality['duplicate_resource_ids']}
- Orphan documents: {quality['orphan_documents']}
- Orphan chunks: {quality['orphan_chunks']}
- Hard errors: {len(quality['hard_errors'])}

## Explicit curation

- Manifest: `{quality.get('curation_manifest_id')}`
- Version: `{quality.get('curation_manifest_version')}`
- Canonical hash: `{quality.get('curation_manifest_hash')}`
- Expected exclusions: {quality.get('curation_exclusions_expected', 0)}
- Exclusions absent from Resources: {quality.get('curation_exclusions_absent_from_resources', 0)}
- Exclusions present in Documents: `{quality.get('curation_exclusions_present_in_documents', [])}`
- Exclusions present in Chunks: `{quality.get('curation_exclusions_present_in_chunks', [])}`

## Weak-relevance review (audit only)

- Candidates: {quality['weak_relevance_count']}
- Reason distribution: `{quality['weak_relevance_reason_distribution']}`

Weak-relevance records are review-only. They do not modify the canonical Resource schema, admission policy, quarantine status, or dataset membership.

## Wall-clock performance

- Collection: {performance['collection_seconds']} seconds
- Phase 2 build: {performance['resource_build_seconds']} seconds
- Phase 3 build: {performance['evidence_build_seconds']} seconds
- Expansion validation: {performance['validation_seconds']} seconds
- Phase 2 validation: {performance['resource_validation_seconds']} seconds
- Phase 3 validation: {performance['evidence_validation_seconds']} seconds
- S3.5 validation: {performance['expansion_validation_seconds']} seconds
- S1 smoke: {performance['smoke_seconds']} seconds

README coverage of at least 95% is a strong quality target, not an admission gate. A shortfall must be investigated and reported; it must not be hidden by replacing high-quality resources with weaker ones.

## Reproducibility boundary

Live GitHub Search is not deterministic. Reproducibility begins at this frozen, self-contained raw run. The raw run ID, Resource build ID, Evidence build ID, and final Git commit/tag form the frozen dataset identity for S4.
"""


def validate_expansion(
    raw_run: Path, resources_path: Path, *, baseline_resources: Path | None = None,
    output_root: Path | None = None, write_report: bool = True, require_evidence: bool = False,
    curation: CurationManifest | None = None,
) -> dict[str, Any]:
    raw_run, resources_path = raw_run.resolve(), resources_path.resolve()
    root = (output_root or Path(__file__).resolve().parents[2]).resolve()
    raw_repositories, raw_readmes = load_baseline_snapshots(raw_run)
    resource_rows = _jsonl(resources_path)
    resources = [Resource.model_validate(item) for item in resource_rows]
    resource_ids = [item.resource_id for item in resources]
    resource_numeric_ids = {int(item.resource_id.split(":", 1)[1]) for item in resources}
    raw_ids = set(raw_repositories)
    baseline_ids = _ids(baseline_resources) if baseline_resources else set()
    excluded_ids = set(curation.excluded_resource_ids) if curation else set()
    current_ids = set(resource_ids)

    manifest_path = root / "data" / "metadata" / "build_manifest.json"
    resource_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    evidence_manifest_path = root / "data" / "metadata" / "evidence_build_manifest.json"
    evidence_manifest = json.loads(evidence_manifest_path.read_text(encoding="utf-8")) if evidence_manifest_path.is_file() else {}
    collector_summary_path = raw_run / "validation_summary.json"
    collector_summary = json.loads(collector_summary_path.read_text(encoding="utf-8")) if collector_summary_path.is_file() else {}
    documents = [EvidenceDocument.model_validate(item) for item in _jsonl(root / "data" / "evidence" / "documents.jsonl")]
    chunks = [EvidenceChunk.model_validate(item) for item in _jsonl(root / "data" / "evidence" / "chunks.jsonl")]
    document_resource_ids = {item.resource_id for item in documents}
    metadata_document_resource_ids = {
        item.resource_id for item in documents if item.evidence_source_type == "repo_metadata"
    }
    readme_document_resource_ids = {
        item.resource_id for item in documents if item.evidence_source_type == "github_readme"
    }
    chunk_document_ids = {item.evidence_id for item in chunks}
    document_ids = {item.evidence_id for item in documents}

    hard_errors: list[str] = []
    if not resource_manifest:
        hard_errors.append("resource_manifest_missing")
    elif resource_manifest.get("final_resources") != len(resources):
        hard_errors.append("resource_manifest_count_mismatch")
    if len(resource_ids) != len(set(resource_ids)):
        hard_errors.append("duplicate_resource_ids")
    expected_raw_ids = raw_ids - {int(item.split(":", 1)[1]) for item in excluded_ids}
    if resource_numeric_ids != expected_raw_ids:
        hard_errors.append("resource_raw_identity_mismatch")
    if baseline_ids - current_ids - excluded_ids:
        hard_errors.append("baseline_ids_missing")
    if excluded_ids & current_ids:
        hard_errors.append("curation_exclusions_in_resources")
    excluded_document_ids = excluded_ids & document_resource_ids
    excluded_chunk_ids = {item.resource_id for item in chunks} & excluded_ids
    if excluded_document_ids:
        hard_errors.append("curation_exclusions_in_documents")
    if excluded_chunk_ids:
        hard_errors.append("curation_exclusions_in_chunks")
    if curation:
        embedded = resource_manifest.get("curation") or {}
        if (
            embedded.get("manifest_id") != curation.manifest_id
            or embedded.get("version") != curation.version
            or embedded.get("canonical_hash") != curation.canonical_hash
        ):
            hard_errors.append("curation_manifest_mismatch")
    orphan_documents = sum(item.resource_id not in current_ids for item in documents)
    orphan_chunks = sum(item.resource_id not in current_ids or item.evidence_id not in document_ids for item in chunks)
    if orphan_documents:
        hard_errors.append("orphan_documents")
    if orphan_chunks:
        hard_errors.append("orphan_chunks")
    metadata_coverage = len(resource_numeric_ids & raw_ids) / len(resources) if resources else 0.0
    any_evidence_coverage = len(current_ids & document_resource_ids) / len(resources) if resources and documents else metadata_coverage
    raw_readme_coverage = len(resource_numeric_ids & set(raw_readmes)) / len(resources) if resources else 0.0
    readme_coverage = (
        len(current_ids & readme_document_resource_ids) / len(resources)
        if resources and documents else raw_readme_coverage
    )
    if metadata_coverage != 1.0:
        hard_errors.append("metadata_coverage_incomplete")
    if any_evidence_coverage != 1.0:
        hard_errors.append("any_evidence_coverage_incomplete")
    if require_evidence and not evidence_manifest:
        hard_errors.append("evidence_corpus_missing")
    if evidence_manifest:
        if evidence_manifest.get("document_count") != len(documents) or evidence_manifest.get("chunk_count") != len(chunks):
            hard_errors.append("evidence_manifest_count_mismatch")
        if any(item.evidence_build_id != evidence_manifest.get("evidence_build_id") for item in [*documents, *chunks]):
            hard_errors.append("evidence_build_id_mismatch")
    stars = sorted(item.stars for item in resources if item.stars is not None)
    weak_relevance = weak_relevance_candidates(
        resources, raw_repositories, raw_readmes, output_root=root,
    )
    weak_reason_distribution = dict(sorted(Counter(
        reason for item in weak_relevance for reason in item["review_reasons"]
    ).items()))
    quality = {
        "valid": not hard_errors,
        "hard_errors": hard_errors,
        "raw_run_id": raw_run.name,
        "self_contained": True,
        "resource_build_id": resource_manifest.get("build_id"),
        "evidence_build_id": evidence_manifest.get("evidence_build_id"),
        "git_commit": evidence_manifest.get("git_commit"),
        "resource_count": len(resources),
        "raw_repository_count": len(raw_repositories),
        "raw_readme_count": len(raw_readmes),
        "metadata_coverage": metadata_coverage,
        "metadata_evidence_coverage": (
            len(current_ids & metadata_document_resource_ids) / len(resources)
            if resources and documents else metadata_coverage
        ),
        "raw_readme_coverage": raw_readme_coverage,
        "readme_coverage": readme_coverage,
        "readme_coverage_target": 0.95,
        "readme_target_met": readme_coverage >= 0.95 if resources else False,
        "missing_readme_evidence_ids": sorted(current_ids - readme_document_resource_ids) if documents else [],
        "any_evidence_coverage": any_evidence_coverage,
        "baseline_id_count": len(baseline_ids),
        "baseline_ids_preserved": len(baseline_ids & current_ids),
        "missing_baseline_ids": sorted(baseline_ids - current_ids - excluded_ids),
        "curation_manifest_id": curation.manifest_id if curation else None,
        "curation_manifest_version": curation.version if curation else None,
        "curation_manifest_hash": curation.canonical_hash if curation else None,
        "curation_exclusions_expected": len(excluded_ids),
        "curation_exclusions_absent_from_resources": len(excluded_ids - current_ids),
        "curation_exclusions_present_in_documents": sorted(excluded_document_ids),
        "curation_exclusions_present_in_chunks": sorted(excluded_chunk_ids),
        "duplicate_resource_ids": len(resource_ids) - len(set(resource_ids)),
        "orphan_documents": orphan_documents,
        "orphan_chunks": orphan_chunks,
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "source_type_distribution": _distribution([item.source_type for item in resources]),
        "language_distribution": _distribution([item.language for item in resources]),
        "topic_distribution": dict(sorted(Counter(topic for item in resources for topic in item.topics).items())),
        "license_distribution": _distribution([item.license for item in resources]),
        "archived_resources": sum(item.is_archived for item in resources),
        "fork_resources": sum(item.is_fork for item in resources),
        "stars": {
            "min": min(stars) if stars else None,
            "median": statistics.median(stars) if stars else None,
            "max": max(stars) if stars else None,
        },
        "weak_relevance_count": len(weak_relevance),
        "weak_relevance_reason_distribution": weak_reason_distribution,
        "collector": {
            "profile": collector_summary.get("profile"), "requests": collector_summary.get("requests"),
            "cache_hits": collector_summary.get("cache_hits"), "cache_revalidations": collector_summary.get("cache_revalidations"),
            "retries": collector_summary.get("retries"), "rate_limits": collector_summary.get("rate_limits", {}),
        },
        "performance": _performance(root),
    }
    if write_report:
        reports = root / "data" / "reports"
        reports.mkdir(parents=True, exist_ok=True)
        review = root / "data" / "review" / "expansion"
        review.mkdir(parents=True, exist_ok=True)
        (review / "weak_relevance_candidates.jsonl").write_text(
            "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in weak_relevance),
            encoding="utf-8",
        )
        (reports / "s3_5_expansion_quality.json").write_text(json.dumps(quality, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (reports / "s3_5_resource_expansion_report.md").write_text(_report(quality), encoding="utf-8")
    return quality
