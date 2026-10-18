import copy
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

import data_pipeline.resource_dataset.core as resource_core
from data_pipeline.resource_dataset.config import load_config
from data_pipeline.resource_dataset.core import _domain_relevance, build_dataset, build_id_for, read_jsonl, validate_dataset
from data_pipeline.resource_dataset.curation import load_curation_manifest
from data_pipeline.resource_dataset.schema import Resource


def _snapshot(source_type: str, source_id: str, payload: dict, *, retrieved_at: str = "2026-08-29T08:51:25Z") -> dict:
    return {
        "collector_version": "0.1.0", "schema_version": "1.0.0", "run_id": "fixture-run",
        "retrieved_at": retrieved_at, "source_type": source_type, "source_id": source_id,
        "source_url": f"https://github.com/{source_id}", "source_hash": f"hash-{source_type}-{source_id}-{retrieved_at}",
        "provenance": {"query_used": "topic:machine-learning", "search_page": 1, "search_position": 1}, "payload": payload,
    }


def make_fixture_run(tmp_path: Path, count: int = 12) -> Path:
    run = tmp_path / "raw-run"
    (run / "github_repository").mkdir(parents=True)
    (run / "github_readme").mkdir()
    for index in range(count):
        full_name = f"owner{index}/ml-repo-{index}"
        metadata = {
            "id": 1000 + index, "name": f"ml-repo-{index}", "full_name": full_name,
            "description": "Machine learning tutorial" if index != 0 else "generic repository",
            "html_url": f"https://github.com/{full_name}/", "owner": {"login": f"owner{index}", "type": "Organization" if index % 2 else "User"},
            "language": "Python", "topics": ["machine-learning"] if index != 0 else [],
            "stargazers_count": index, "forks_count": index, "watchers_count": index,
            "open_issues_count": 0, "license": {"spdx_id": "MIT"}, "archived": index == 1, "fork": index == 2,
            "created_at": "2025-01-01T00:00:00Z", "updated_at": "2026-08-01T00:00:00Z", "pushed_at": "2026-08-15T00:00:00Z",
        }
        (run / "github_repository" / f"{index}.json").write_text(json.dumps(_snapshot("github_repository", full_name, metadata)), encoding="utf-8")
        readme = {"decoded_text": "generic repository notes" if index == 0 else "PyTorch framework appears here but must not populate frameworks."}
        (run / "github_readme" / f"{index}.json").write_text(json.dumps(_snapshot("github_readme", full_name, readme)), encoding="utf-8")
    return run


def fixture_config(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    raw = yaml.safe_load((root / "configs" / "resource_dataset.yaml").read_text(encoding="utf-8"))
    vocab = yaml.safe_load((root / "configs" / "vocabulary.yaml").read_text(encoding="utf-8"))
    raw = copy.deepcopy(raw)
    raw["dataset"]["paths"] = {
        "processed_root": str(tmp_path / "processed"), "quarantine_root": str(tmp_path / "quarantine"),
        "review_root": str(tmp_path / "review"), "metadata_root": str(tmp_path / "metadata"), "reports_root": str(tmp_path / "reports"),
    }
    config_path, vocab_path = tmp_path / "config.yaml", tmp_path / "vocabulary.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    vocab_path.write_text(yaml.safe_dump(vocab), encoding="utf-8")
    return load_config(config_path, vocab_path)


def test_resource_requires_stable_numeric_github_identity():
    with pytest.raises(ValidationError):
        Resource.model_validate({"resource_id": "gh_owner_repo"})


def test_build_preserves_conservative_semantic_fields_and_provenance(tmp_path):
    config = fixture_config(tmp_path)
    summary = build_dataset(make_fixture_run(tmp_path), config)
    assert summary["final_resources"] == 12
    records = read_jsonl(config.path("processed_root") / "resources.jsonl")
    first = records[0]
    assert first["resource_id"] == "github:1000"
    assert first["frameworks"] == []
    assert first["difficulty"] == "unknown"
    assert first["is_archived"] is False
    provenance = read_jsonl(config.path("processed_root") / "resource_provenance.jsonl")
    assert all(row["build_id"] == summary["build_id"] for row in provenance)
    assert any(row["field"] == "domain_relevance_score" and row["evidence_source_type"] == "github_readme" for row in provenance)
    manifest = json.loads((config.path("metadata_root") / "build_manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["input_source_hashes"]) == 24


def test_outputs_are_consistent_and_deterministic(tmp_path):
    config = fixture_config(tmp_path)
    run = make_fixture_run(tmp_path)
    first = build_dataset(run, config)
    second = build_dataset(run, config)
    assert first["build_id"] == second["build_id"]
    assert validate_dataset(config)["valid"] is True
    jsonl = read_jsonl(config.path("processed_root") / "resources.jsonl")
    assert [row["resource_id"] for row in jsonl] == pd.read_parquet(config.path("processed_root") / "resources.parquet")["resource_id"].tolist()


def test_invalid_snapshots_are_quarantined_and_low_relevance_is_review_only(tmp_path):
    config = fixture_config(tmp_path)
    run = make_fixture_run(tmp_path)
    (run / "github_repository" / "invalid.json").write_text(json.dumps({"source_type": "github_repository"}), encoding="utf-8")
    (run / "github_repository" / "broken.json").write_text("{bad json", encoding="utf-8")
    summary = build_dataset(run, config)
    assert summary["final_resources"] == 12
    assert summary["invalid_schema_records"] == 1
    assert summary["failed_parse_records"] == 1
    assert len(read_jsonl(config.path("review_root") / "low_relevance.jsonl")) == 1
    assert len(read_jsonl(config.path("quarantine_root") / "invalid_schema.jsonl")) == 1


def test_archived_and_fork_resources_remain_in_canonical_dataset(tmp_path):
    config = fixture_config(tmp_path)
    build_dataset(make_fixture_run(tmp_path), config)
    records = {row["resource_id"]: row for row in read_jsonl(config.path("processed_root") / "resources.jsonl")}
    assert records["github:1001"]["is_archived"] is True
    assert records["github:1002"]["is_fork"] is True


def test_relevance_normalizes_separators_and_ai_uses_word_boundaries(tmp_path):
    config = fixture_config(tmp_path)
    score, matches = _domain_relevance(
        {"topics": ["machine_learning"], "description": "A chair catalog"},
        {"payload": {"decoded_text": "ordinary documentation"}}, config,
    )
    assert score == 0.6
    assert matches == {"topics": True, "description": False, "readme": False}
    score, matches = _domain_relevance(
        {"topics": [], "description": "An AI tutorial"}, None, config,
    )
    assert score == 0.3 and matches["description"] is True


def test_output_root_is_isolated_and_build_identity_does_not_depend_on_it(tmp_path):
    run = make_fixture_run(tmp_path)
    root = Path(__file__).resolve().parents[1]
    first = load_config(output_root=tmp_path / "stage-one")
    second = load_config(output_root=tmp_path / "stage-two")
    first_result = build_dataset(run, first)
    second_result = build_dataset(run, second)
    assert first_result["build_id"] == second_result["build_id"]
    assert (tmp_path / "stage-one" / "data" / "processed" / "resources.jsonl").is_file()
    assert (tmp_path / "stage-two" / "data" / "metadata" / "build_manifest.json").is_file()
    assert validate_dataset(first)["resources"] == 12


def test_atomic_build_failure_preserves_previous_outputs(tmp_path, monkeypatch):
    config = fixture_config(tmp_path)
    run = make_fixture_run(tmp_path)
    build_dataset(run, config)
    target = config.path("processed_root") / "resources.jsonl"
    before = target.read_bytes()

    def fail_publish(stage_config, target_config):
        raise OSError("simulated publication failure")

    monkeypatch.setattr(resource_core, "_atomic_publish_dataset", fail_publish)
    with pytest.raises(OSError, match="simulated publication failure"):
        build_dataset(run, config)
    assert target.read_bytes() == before


def test_atomic_publish_rolls_back_partial_replacement(tmp_path, monkeypatch):
    config = fixture_config(tmp_path)
    stage_config = resource_core._staging_config(config, tmp_path / "stage")
    for key, name in resource_core.OUTPUT_FILES:
        target = config.path(key) / name
        source = stage_config.path(key) / name
        target.parent.mkdir(parents=True, exist_ok=True)
        source.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("old", encoding="utf-8")
        source.write_text("new", encoding="utf-8")
    original_replace = resource_core.os.replace
    calls = 0

    def failing_replace(source, target):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise OSError("simulated partial publication failure")
        original_replace(source, target)

    monkeypatch.setattr(resource_core.os, "replace", failing_replace)
    with pytest.raises(OSError, match="partial publication"):
        resource_core._atomic_publish_dataset(stage_config, config)
    assert all((config.path(key) / name).read_text(encoding="utf-8") == "old" for key, name in resource_core.OUTPUT_FILES)


def _curation_file(tmp_path: Path, run: Path, config, exclusions: list[str]) -> Path:
    path = tmp_path / "curation.yaml"
    path.write_text(yaml.safe_dump({
        "curation_manifest_version": "1.0.0",
        "manifest_id": "fixture_curation_v1",
        "input": {"raw_run_id": "fixture-run", "resource_build_id": build_id_for(run, config)},
        "decision_scope": "exclude_only",
        "exclusions": [
            {"resource_id": resource_id, "reason_code": "manual_out_of_domain", "reason": "Human-reviewed exclusion."}
            for resource_id in exclusions
        ],
    }, sort_keys=False), encoding="utf-8")
    return path


def test_explicit_curation_is_exact_deterministic_and_part_of_build_identity(tmp_path):
    config = fixture_config(tmp_path)
    run = make_fixture_run(tmp_path)
    raw_before = {path: path.read_bytes() for path in run.rglob("*.json")}
    path = _curation_file(tmp_path, run, config, ["github:1001", "github:1003"])
    curation = load_curation_manifest(path)
    uncurated_id = build_id_for(run, config)

    first = build_dataset(run, config, curation=curation)
    first_records = (config.path("processed_root") / "resources.jsonl").read_bytes()
    second = build_dataset(run, config, curation=curation)

    assert first["build_id"] == second["build_id"] != uncurated_id
    assert first["final_resources"] == 10
    assert first["curation_exclusions_applied"] == 2
    assert first_records == (config.path("processed_root") / "resources.jsonl").read_bytes()
    assert [row["resource_id"] for row in read_jsonl(config.path("processed_root") / "resources.jsonl")] == [
        f"github:{value}" for value in [1000, 1002, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011]
    ]
    manifest = json.loads((config.path("metadata_root") / "build_manifest.json").read_text(encoding="utf-8"))
    assert manifest["curation"]["manifest_id"] == "fixture_curation_v1"
    assert manifest["curation"]["version"] == "1.0.0"
    assert manifest["curation"]["canonical_hash"] == curation.canonical_hash
    assert manifest["curation"]["excluded_resource_ids"] == ["github:1001", "github:1003"]
    assert all(row["resource_id"] not in curation.excluded_resource_ids for row in read_jsonl(config.path("processed_root") / "resource_provenance.jsonl"))
    assert raw_before == {raw_path: raw_path.read_bytes() for raw_path in run.rglob("*.json")}


def test_curation_rejects_non_numeric_and_unknown_resource_id(tmp_path):
    config = fixture_config(tmp_path)
    run = make_fixture_run(tmp_path)
    invalid = _curation_file(tmp_path, run, config, ["owner/repo"])
    with pytest.raises(ValueError, match="exact GitHub numeric resource_id"):
        load_curation_manifest(invalid)

    unknown = _curation_file(tmp_path, run, config, ["github:999999"])
    with pytest.raises(ValueError, match="not present in input run"):
        build_dataset(run, config, curation=load_curation_manifest(unknown))
