"""Deterministic, provenance-preserving Phase 2 transformation pipeline."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
import tempfile
import unicodedata
from collections import Counter
from datetime import UTC, datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pandas as pd
from pydantic import ValidationError

from .config import DatasetConfig
from .curation import CurationManifest, HASH_PATTERN
from .schema import Resource


TRACKING_QUERY_PREFIXES = ("utm_",)
TRACKING_QUERY_KEYS = {"gclid", "fbclid"}

OUTPUT_FILES = (
    ("processed_root", "resources.jsonl"),
    ("processed_root", "resources.parquet"),
    ("processed_root", "resource_aliases.jsonl"),
    ("processed_root", "resource_provenance.jsonl"),
    ("quarantine_root", "invalid_schema.jsonl"),
    ("quarantine_root", "failed_parse.jsonl"),
    ("review_root", "duplicate_candidates.jsonl"),
    ("review_root", "low_relevance.jsonl"),
    ("reports_root", "resource_dataset_quality.json"),
    ("reports_root", "resource_dataset_summary.csv"),
    ("reports_root", "resource_dataset_report.md"),
    ("reports_root", "sample_resources.jsonl"),
    ("reports_root", "resource_dataset_validation.json"),
    ("metadata_root", "build_manifest.json"),
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def utc_iso(value: str | datetime) -> str:
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def clean_text(value: str | None) -> str | None:
    if value is None:
        return None
    value = unicodedata.normalize("NFKC", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value or None


def canonical_url(value: str) -> str:
    parts = urlsplit(value)
    retained_query = [(key, item) for key, item in parse_qsl(parts.query, keep_blank_values=True)
                      if not key.lower().startswith(TRACKING_QUERY_PREFIXES) and key.lower() not in TRACKING_QUERY_KEYS]
    path = parts.path.rstrip("/") or "/"
    if path == "/" and parts.netloc.lower() == "github.com":
        path = ""
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, urlencode(retained_query, doseq=True), parts.fragment))


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(canonical_json(record) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def iter_snapshots(run_dir: Path) -> Iterable[tuple[Path, dict[str, Any]]]:
    for path in sorted(run_dir.rglob("*.json")):
        if path.name == "validation_summary.json":
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            yield path, {"_parse_error": str(exc)}
            continue
        yield path, payload


def validate_wrapper(snapshot: dict[str, Any]) -> None:
    required = {"collector_version", "schema_version", "run_id", "retrieved_at", "source_type", "source_id", "source_url", "source_hash", "payload"}
    missing = required - snapshot.keys()
    if missing:
        raise ValueError(f"missing snapshot wrapper fields: {sorted(missing)}")
    utc_iso(snapshot["retrieved_at"])


def build_id_for(
    run_dir: Path, config: DatasetConfig, curation: CurationManifest | None = None,
) -> str:
    hashes: list[str] = []
    for _, snapshot in iter_snapshots(run_dir):
        if "source_hash" in snapshot:
            hashes.append(snapshot["source_hash"])
    semantic_config = {**config.raw, "dataset": {**config.dataset}}
    semantic_config["dataset"].pop("paths", None)
    material = {
        "input_source_hashes": sorted(hashes),
        "config": semantic_config,
        "vocabulary": config.vocabulary,
        "schema_version": config.dataset["schema_version"],
        "normalizer_version": config.dataset["normalizer_version"],
    }
    if curation is not None:
        material["curation_manifest_version"] = curation.version
        material["curation_manifest_hash"] = curation.canonical_hash
    return f"build_{sha256(material)[:16]}"


def _source_type(metadata: dict[str, Any], config: DatasetConfig) -> str:
    owner = str(metadata.get("owner", {}).get("login", "")).casefold()
    official = {str(item).casefold() for item in config.dataset["official_github_organizations"]}
    return "official_project" if owner and owner in official else "community"


def _score_age(timestamp: str | None, reference: datetime, tau_days: float) -> float | None:
    if not timestamp:
        return None
    age = max(0.0, (reference - datetime.fromisoformat(timestamp.replace("Z", "+00:00"))).total_seconds() / 86400)
    return round(math.exp(-age / tau_days), 6)


def _relevance_text(value: Any) -> str:
    return re.sub(r"[-_\s]+", " ", str(value).casefold()).strip()


def _contains_term(text: str, term: str) -> bool:
    normalized = _relevance_text(term)
    return bool(normalized and re.search(rf"(?<!\w){re.escape(normalized)}(?!\w)", text))


def _domain_relevance(metadata: dict[str, Any], readme: dict[str, Any] | None, config: DatasetConfig) -> tuple[float, dict[str, bool]]:
    relevance = config.vocabulary["domain_relevance"]
    terms = [_relevance_text(term) for term in relevance["terms"]]
    weights = relevance["weights"]
    topics_text = _relevance_text(" ".join(str(item) for item in metadata.get("topics") or []))
    description = _relevance_text(metadata.get("description") or "")
    readme_text = _relevance_text(((readme or {}).get("payload") or {}).get("decoded_text") or "")
    matches = {
        "topics": any(_contains_term(topics_text, term) for term in terms),
        "description": any(_contains_term(description, term) for term in terms),
        "readme": any(_contains_term(readme_text, term) for term in terms),
    }
    score = sum(float(weights[key]) for key, matched in matches.items() if matched)
    return round(min(1.0, score), 6), matches


def _metadata_completeness(record: dict[str, Any], fields: list[str]) -> float:
    def filled(value: Any) -> bool:
        return value is not None and value != "" and value != []
    return round(sum(filled(record.get(field)) for field in fields) / len(fields), 6)


def _to_resource(snapshot: dict[str, Any], readme: dict[str, Any] | None, config: DatasetConfig) -> tuple[Resource, dict[str, Any]]:
    metadata = snapshot["payload"]
    github_id = metadata.get("id")
    if not isinstance(github_id, int):
        raise ValueError("GitHub metadata payload.id must be an integer")
    url = metadata.get("html_url")
    if not isinstance(url, str):
        raise ValueError("GitHub metadata payload.html_url is required")
    reference = datetime.fromisoformat(config.dataset["reference_time"].replace("Z", "+00:00"))
    source_type = _source_type(metadata, config)
    owner = metadata.get("owner") or {}
    owner_type = clean_text(owner.get("type"))
    relevance, relevance_matches = _domain_relevance(metadata, readme, config)
    record: dict[str, Any] = {
        "resource_id": f"github:{github_id}", "resource_type": "github_project",
        "name": clean_text(metadata.get("name")) or metadata.get("full_name") or str(github_id),
        "canonical_name": clean_text(metadata.get("name")) or metadata.get("full_name") or str(github_id),
        "full_name": clean_text(metadata.get("full_name")), "description": clean_text(metadata.get("description")),
        "url": url, "canonical_url": canonical_url(url), "source_platform": "github",
        "source_id": f"github:{github_id}", "source_type": source_type,
        "source_reliability": float(config.priors[source_type]), "source_reliability_method": "heuristic_v0",
        "owner_name": clean_text(owner.get("login")), "owner_type": owner_type.casefold() if owner_type else None,
        "organization_name": clean_text(owner.get("login")) if owner_type == "Organization" else None,
        "language": clean_text(metadata.get("language")),
        "topics": sorted({clean_text(str(topic)).casefold() for topic in metadata.get("topics") or [] if clean_text(str(topic))}),
        "frameworks": [], "skills": [], "prerequisites": [], "difficulty": "unknown", "learning_style": [], "tasks": [],
        "stars": metadata.get("stargazers_count"), "forks": metadata.get("forks_count"),
        "watchers": metadata.get("watchers_count"), "open_issues": metadata.get("open_issues_count"),
        "license": clean_text((metadata.get("license") or {}).get("spdx_id")) or "unknown",
        "is_archived": bool(metadata.get("archived", False)), "is_fork": bool(metadata.get("fork", False)),
        "created_at": metadata.get("created_at"), "updated_at": metadata.get("updated_at"), "pushed_at": metadata.get("pushed_at"),
        "activity_score": _score_age(metadata.get("pushed_at") or metadata.get("updated_at"), reference, float(config.dataset["activity_tau_days"])),
        "freshness_score": _score_age(metadata.get("updated_at"), reference, float(config.dataset["freshness_tau_days"])),
        "metadata_completeness": 0.0, "domain_relevance_score": relevance, "data_quality_score": 0.0,
        "raw_source_id": snapshot["source_id"], "retrieved_at": utc_iso(snapshot["retrieved_at"]),
        "record_version": config.dataset["record_version"],
    }
    record["metadata_completeness"] = _metadata_completeness(record, config.dataset["important_fields"])
    record["data_quality_score"] = round((1.0 + 1.0 + record["metadata_completeness"]) / 3, 6)
    return Resource.model_validate(record), relevance_matches


def _provenance_records(resource: Resource, snapshot: dict[str, Any], readme: dict[str, Any] | None, relevance_matches: dict[str, bool], build_id: str, config: DatasetConfig) -> list[dict[str, Any]]:
    raw_fields = {"name", "full_name", "description", "url", "source_id", "owner_name", "owner_type", "organization_name", "language", "topics", "stars", "forks", "watchers", "open_issues", "license", "is_archived", "is_fork", "created_at", "updated_at", "pushed_at"}
    derived_fields = {"activity_score", "freshness_score", "metadata_completeness", "domain_relevance_score", "data_quality_score"}
    base = {
        "resource_id": resource.resource_id, "raw_source_id": snapshot["source_id"], "source_url": snapshot["source_url"],
        "source_hash": snapshot["source_hash"], "retrieved_at": snapshot["retrieved_at"], "run_id": snapshot["run_id"],
        "collector_version": snapshot["collector_version"], "snapshot_schema_version": snapshot["schema_version"],
        "search_provenance": snapshot.get("provenance", {}), "build_id": build_id,
        "normalizer_version": config.dataset["normalizer_version"],
    }
    records: list[dict[str, Any]] = []
    for field, value in resource.model_dump(mode="json").items():
        method = "api_metadata" if field in raw_fields else "derived_feature" if field in derived_fields else "normalization"
        record = {**base, "field": field, "value": value, "extraction_method": method, "extraction_version": config.dataset["normalizer_version"], "evidence_source_type": "repo_metadata"}
        if field == "domain_relevance_score" and readme and relevance_matches.get("readme"):
            record.update({"evidence_source_type": "github_readme", "evidence_source_url": readme["source_url"], "evidence_source_hash": readme["source_hash"]})
        records.append(record)
    return records


def _review_near_duplicates(resources: list[Resource], threshold: float) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for index, left in enumerate(resources):
        for right in resources[index + 1:]:
            if left.owner_name != right.owner_name:
                continue
            similarity = SequenceMatcher(None, left.canonical_name.casefold(), right.canonical_name.casefold()).ratio()
            if similarity >= threshold:
                candidates.append({"resource_id_a": left.resource_id, "resource_id_b": right.resource_id, "reason": "near_duplicate_name", "similarity": round(similarity, 6), "decision": "review"})
    return candidates


def _write_reports(resources: list[Resource], summary: dict[str, Any], config: DatasetConfig) -> None:
    reports = config.path("reports_root")
    reports.mkdir(parents=True, exist_ok=True)
    distributions = {
        "resource_type": Counter(item.resource_type for item in resources),
        "source_type": Counter(item.source_type for item in resources),
        "language": Counter(item.language or "unknown" for item in resources),
        "difficulty": Counter(item.difficulty for item in resources),
    }
    missing_fields = ["description", "language", "frameworks", "difficulty", "skills", "prerequisites"]
    report = {**summary, "distributions": {key: dict(sorted(value.items())) for key, value in distributions.items()}, "missingness": {field: round(sum(not getattr(item, field) or getattr(item, field) == "unknown" for item in resources) / len(resources), 6) if resources else 0.0 for field in missing_fields}, "archived_resources": sum(item.is_archived for item in resources), "fork_resources": sum(item.is_fork for item in resources), "domain_relevance": [item.domain_relevance_score for item in resources], "freshness": [item.freshness_score for item in resources if item.freshness_score is not None], "source_reliability_priors": dict(sorted(Counter(item.source_reliability for item in resources).items()))}
    (reports / "resource_dataset_quality.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (reports / "resource_dataset_summary.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in summary.items():
            writer.writerow({"metric": key, "value": canonical_json(value) if isinstance(value, (dict, list)) else value})
    markdown = "# Resource Dataset Report\n\n" + "\n".join(f"- **{key}**: `{value}`" for key, value in summary.items())
    markdown += f"\n\nThis build contains {len(resources)} resources. Dataset scale is `{summary['dataset_scale']}`.\n"
    (reports / "resource_dataset_report.md").write_text(markdown, encoding="utf-8")
    randomizer = random.Random(int(config.dataset["random_seed"]))
    sample = sorted(resources, key=lambda resource: resource.resource_id)
    randomizer.shuffle(sample)
    write_jsonl(reports / "sample_resources.jsonl", [item.model_dump(mode="json") for item in sample[:10]])


def _project_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path(__file__).resolve().parents[2]).as_posix()
    except ValueError:
        return path.name


def _build_dataset_files(
    run_dir: Path, config: DatasetConfig, curation: CurationManifest | None = None,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"input run directory does not exist: {run_dir}")
    uncurated_build_id = build_id_for(run_dir, config)
    build_id = build_id_for(run_dir, config, curation)
    metadata: list[dict[str, Any]] = []
    readmes: dict[str, dict[str, Any]] = {}
    invalid: list[dict[str, Any]] = []
    failed_parse: list[dict[str, Any]] = []
    input_source_hashes: list[str] = []
    for path, snapshot in iter_snapshots(run_dir):
        if "_parse_error" in snapshot:
            failed_parse.append({"path": str(path), "error": snapshot["_parse_error"], "build_id": build_id})
            continue
        try:
            validate_wrapper(snapshot)
        except (ValueError, KeyError) as exc:
            invalid.append({"path": str(path), "error": str(exc), "snapshot": snapshot, "build_id": build_id})
            continue
        input_source_hashes.append(snapshot["source_hash"])
        if snapshot["source_type"] == "github_repository":
            metadata.append(snapshot)
        elif snapshot["source_type"] == "github_readme":
            readmes[snapshot["source_id"]] = snapshot
    winners: dict[str, dict[str, Any]] = {}
    aliases: list[dict[str, Any]] = []
    for snapshot in sorted(metadata, key=lambda item: (item["retrieved_at"], item["source_hash"]), reverse=True):
        github_id = (snapshot.get("payload") or {}).get("id")
        key = f"github:{github_id}" if isinstance(github_id, int) else f"invalid:{snapshot['source_hash']}"
        if key in winners:
            aliases.append({"resource_id": key, "kept_raw_source_id": winners[key]["source_id"], "aliased_raw_source_id": snapshot["source_id"], "reason": "duplicate_stable_source_id", "build_id": build_id})
        else:
            winners[key] = snapshot
    excluded_ids: frozenset[str] = frozenset()
    curation_metadata: dict[str, Any] | None = None
    if curation is not None:
        run_ids = {str(item["run_id"]) for item in [*metadata, *readmes.values()]}
        if run_ids != {curation.raw_run_id}:
            raise ValueError(
                f"curation raw_run_id mismatch: expected {curation.raw_run_id}, found {sorted(run_ids)}"
            )
        if curation.input_resource_build_id != uncurated_build_id:
            raise ValueError(
                "curation input Resource build mismatch: "
                f"expected {curation.input_resource_build_id}, computed {uncurated_build_id}"
            )
        available_ids = {key for key in winners if key.startswith("github:")}
        unknown_ids = sorted(curation.excluded_resource_ids - available_ids)
        if unknown_ids:
            raise ValueError(f"curation exclusions not present in input run: {unknown_ids}")
        excluded_ids = curation.excluded_resource_ids
        curation_metadata = curation.build_metadata()
    resources: list[Resource] = []
    provenance: list[dict[str, Any]] = []
    relevance_review: list[dict[str, Any]] = []
    seen_urls: dict[str, Resource] = {}
    for snapshot in winners.values():
        try:
            resource, relevance_matches = _to_resource(snapshot, readmes.get(snapshot["source_id"]), config)
            if resource.resource_id in excluded_ids:
                continue
            if resource.canonical_url in seen_urls:
                aliases.append({"resource_id": seen_urls[resource.canonical_url].resource_id, "aliased_raw_source_id": resource.raw_source_id, "reason": "duplicate_canonical_url", "build_id": build_id})
                continue
            seen_urls[resource.canonical_url] = resource
            resources.append(resource)
            provenance.extend(_provenance_records(resource, snapshot, readmes.get(snapshot["source_id"]), relevance_matches, build_id, config))
            if resource.domain_relevance_score == 0:
                relevance_review.append({"resource_id": resource.resource_id, "reason": "no_configured_domain_term_match", "decision": "review_only", "build_id": build_id})
        except (ValueError, ValidationError, KeyError, TypeError) as exc:
            invalid.append({"raw_source_id": snapshot["source_id"], "error": str(exc), "snapshot_hash": snapshot["source_hash"], "build_id": build_id})
    resources.sort(key=lambda item: item.resource_id)
    aliases = [item for item in aliases if item.get("resource_id") not in excluded_ids]
    duplicate_review = _review_near_duplicates(resources, float(config.dataset["near_duplicate_threshold"]))
    processed = config.path("processed_root")
    quarantine = config.path("quarantine_root")
    review = config.path("review_root")
    metadata_root = config.path("metadata_root")
    for directory in (processed, quarantine, review, metadata_root):
        directory.mkdir(parents=True, exist_ok=True)
    records = [resource.model_dump(mode="json") for resource in resources]
    write_jsonl(processed / "resources.jsonl", records)
    pd.DataFrame(records).to_parquet(processed / "resources.parquet", index=False)
    write_jsonl(processed / "resource_aliases.jsonl", aliases)
    write_jsonl(processed / "resource_provenance.jsonl", provenance)
    write_jsonl(quarantine / "invalid_schema.jsonl", invalid)
    write_jsonl(quarantine / "failed_parse.jsonl", failed_parse)
    write_jsonl(review / "duplicate_candidates.jsonl", duplicate_review)
    write_jsonl(review / "low_relevance.jsonl", relevance_review)
    summary = {"build_id": build_id, "dataset_scale": config.dataset["dataset_scale"], "input_run": _project_relative(run_dir), "raw_metadata_records": len(metadata), "raw_readme_records": len(readmes), "final_resources": len(resources), "invalid_schema_records": len(invalid), "failed_parse_records": len(failed_parse), "merged_duplicates": len(aliases), "duplicate_review_candidates": len(duplicate_review), "low_relevance_review_records": len(relevance_review), "schema_version": config.dataset["schema_version"], "normalizer_version": config.dataset["normalizer_version"], "source_reliability_method": "heuristic_v0", "curation_manifest_id": curation.manifest_id if curation else None, "curation_manifest_version": curation.version if curation else None, "curation_manifest_hash": curation.canonical_hash if curation else None, "curation_exclusions_applied": len(excluded_ids)}
    manifest = {**summary, "input_source_hashes": sorted(input_source_hashes), "config": config.raw, "vocabulary": config.vocabulary, "curation": curation_metadata}
    (metadata_root / "build_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_reports(resources, summary, config)
    return summary


def _staging_config(config: DatasetConfig, stage: Path) -> DatasetConfig:
    raw = json.loads(json.dumps(config.raw))
    raw["dataset"]["paths"] = {key: str(stage / key) for key in raw["dataset"]["paths"]}
    return DatasetConfig(raw, config.vocabulary, config.config_path, config.vocabulary_path)


def _atomic_publish_dataset(stage_config: DatasetConfig, config: DatasetConfig) -> None:
    backup = Path(tempfile.mkdtemp(prefix="resource-backup-", dir=config.path("processed_root").parent))
    replaced: list[tuple[str, str]] = []
    try:
        for key, name in OUTPUT_FILES:
            source, target = stage_config.path(key) / name, config.path(key) / name
            target.parent.mkdir(parents=True, exist_ok=True)
            backup_target = backup / key / name
            if target.exists():
                backup_target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, backup_target)
            if not source.exists():
                raise FileNotFoundError(f"validated staging output is missing: {source}")
            os.replace(source, target)
            replaced.append((key, name))
    except Exception:
        for key, name in reversed(replaced):
            target, backup_target = config.path(key) / name, backup / key / name
            if backup_target.exists():
                os.replace(backup_target, target)
            elif target.exists():
                target.unlink()
        raise
    finally:
        shutil.rmtree(backup, ignore_errors=True)


def build_dataset(
    run_dir: Path, config: DatasetConfig, *, curation: CurationManifest | None = None,
) -> dict[str, Any]:
    staging_parent = config.path("processed_root").parent / ".resource-staging"
    staging_parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="build-", dir=staging_parent))
    stage_config = _staging_config(config, stage)
    try:
        summary = _build_dataset_files(run_dir, stage_config, curation)
        staged_manifest_path = stage_config.path("metadata_root") / "build_manifest.json"
        staged_manifest = json.loads(staged_manifest_path.read_text(encoding="utf-8"))
        staged_manifest["config"] = config.raw
        staged_manifest_path.write_text(json.dumps(staged_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        validate_dataset(stage_config)
        _atomic_publish_dataset(stage_config, config)
        return summary
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def validate_dataset(config: DatasetConfig) -> dict[str, Any]:
    processed = config.path("processed_root")
    records = read_jsonl(processed / "resources.jsonl")
    resources = [Resource.model_validate(record) for record in records]
    ids = [resource.resource_id for resource in resources]
    if len(ids) != len(set(ids)):
        raise ValueError("resource_id values are not unique")
    parquet_ids = pd.read_parquet(processed / "resources.parquet")["resource_id"].tolist()
    if ids != parquet_ids:
        raise ValueError("JSONL and Parquet resource_id ordering differs")
    manifest = json.loads((config.path("metadata_root") / "build_manifest.json").read_text(encoding="utf-8"))
    provenance = read_jsonl(processed / "resource_provenance.jsonl")
    if any(record.get("build_id") != manifest["build_id"] for record in provenance):
        raise ValueError("provenance build_id does not match manifest")
    curation = manifest.get("curation")
    if curation is not None:
        if not HASH_PATTERN.fullmatch(str(curation.get("canonical_hash", ""))):
            raise ValueError("curation manifest canonical hash is invalid")
        excluded_ids = set(curation.get("excluded_resource_ids") or [])
        if excluded_ids & set(ids):
            raise ValueError("curation exclusions remain in canonical Resource Dataset")
        if manifest.get("curation_exclusions_applied") != len(excluded_ids):
            raise ValueError("curation exclusion count does not match manifest")
    result = {"valid": True, "build_id": manifest["build_id"], "resources": len(resources), "provenance_records": len(provenance)}
    (config.path("reports_root") / "resource_dataset_validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
