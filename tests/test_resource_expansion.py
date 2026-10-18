from __future__ import annotations

import base64
import json
import logging
import shutil
from dataclasses import replace
from pathlib import Path

import httpx
import yaml

from data_pipeline.collectors.github_collector import GitHubClient, GitHubCollector, load_config, stable_hash
from data_pipeline.evidence.config import load_config as load_evidence_config
from data_pipeline.evidence.core import build_corpus
from data_pipeline.expansion.core import validate_expansion
from data_pipeline.expansion.timing import record_duration
from data_pipeline.resource_dataset.config import load_config as load_resource_config
from data_pipeline.resource_dataset.core import build_dataset, build_id_for, read_jsonl
from data_pipeline.resource_dataset.curation import load_curation_manifest
from recommender.config import load_config as load_summer_config
from recommender.pipeline import run_core_pipeline
from recommender.schemas import StructuredIntent


STAMP = "2026-08-29T08:51:25Z"


def _wrapper(run: Path, source_type: str, source_id: str, payload: dict) -> None:
    target = run / source_type / f"{source_id.replace('/', '__')}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    wrapper = {
        "collector_version": "0.1.0", "schema_version": "1.0.0", "run_id": run.name,
        "retrieved_at": STAMP, "source_type": source_type, "source_id": source_id,
        "source_url": f"https://github.com/{source_id}", "source_hash": stable_hash(payload),
        "provenance": {"query_used": "topic:machine-learning", "search_page": 1, "search_position": 1},
        "payload": payload,
    }
    target.write_text(json.dumps(wrapper), encoding="utf-8")


def _baseline(tmp_path: Path, count: int = 25) -> Path:
    run = tmp_path / "baseline-run"
    for index in range(1, count + 1):
        full_name = f"owner/repo-{index}"
        metadata = {
            "id": index, "name": f"repo-{index}", "full_name": full_name,
            "description": "Machine learning tutorial", "html_url": f"https://github.com/{full_name}",
            "owner": {"login": "owner", "type": "Organization"}, "language": "Python",
            "topics": ["machine-learning"], "stargazers_count": index, "forks_count": 0,
            "watchers_count": index, "open_issues_count": 0, "license": {"spdx_id": "MIT"},
            "archived": False, "fork": False, "default_branch": "main",
            "created_at": STAMP, "updated_at": STAMP, "pushed_at": STAMP,
        }
        readme_api = {
            "encoding": "base64", "content": base64.b64encode(f"# Repo {index}\nMachine learning evidence.".encode()).decode(),
            "sha": f"sha-{index}", "html_url": f"https://github.com/{full_name}/blob/main/README.md",
        }
        _wrapper(run, "github_repository", full_name, metadata)
        _wrapper(run, "github_readme", full_name, {"api_response": readme_api, "decoded_text": f"# Repo {index}\nMachine learning evidence."})
    return run


def test_self_contained_stage_rebuilds_phase2_and_phase3_after_baseline_is_removed(tmp_path: Path) -> None:
    baseline = _baseline(tmp_path)
    collector_config = replace(
        load_config(), queries=[], query_to_group={}, group_shares={}, max_total_repos=25,
        raw_root=tmp_path / "stage-raw-root", cache_enabled=False,
    )
    client = GitHubClient(collector_config, logging.getLogger("test"), sleeper=lambda _: None)
    client.http.close()
    client.http = httpx.Client(transport=httpx.MockTransport(lambda request: (_ for _ in ()).throw(AssertionError(request))))
    collector = GitHubCollector(collector_config, client=client, baseline_run=baseline)
    collector.run()
    stage_run = collector.run_dir
    baseline_resources = tmp_path / "baseline-resources.jsonl"
    baseline_resources.write_text("".join(json.dumps({"resource_id": f"github:{index}"}) + "\n" for index in range(1, 26)), encoding="utf-8")
    shutil.rmtree(baseline)

    output = tmp_path / "isolated-stage"
    resource_config = load_resource_config(output_root=output)
    resource_result = build_dataset(stage_run, resource_config)
    resources_jsonl = output / "data/processed/resources.jsonl"
    assert resource_result["final_resources"] == 25
    assert len(read_jsonl(resources_jsonl)) == 25

    resource_manifest = output / "data/metadata/build_manifest.json"
    evidence_config = load_evidence_config(output_root=output, resource_manifest=resource_manifest)
    evidence_result = build_corpus(
        output / "data/processed/resources.parquet", stage_run, evidence_config, output_root=output,
    )
    assert evidence_result["documents"] == 50

    quality = validate_expansion(
        stage_run, resources_jsonl, baseline_resources=baseline_resources,
        output_root=output, write_report=True,
    )
    assert quality["valid"] is True
    assert quality["resource_count"] == 25
    assert quality["metadata_coverage"] == 1.0
    assert quality["metadata_evidence_coverage"] == 1.0
    assert quality["raw_readme_coverage"] == 1.0
    assert quality["readme_coverage"] == 1.0
    assert quality["baseline_ids_preserved"] == 25
    assert quality["orphan_documents"] == quality["orphan_chunks"] == 0

    summer_config = load_summer_config().model_copy(
        update={"logging": load_summer_config().logging.model_copy(update={"save_sessions": False})},
    )
    smoke = run_core_pipeline(
        StructuredIntent(topics=["Machine Learning"]), config=summer_config, root=output,
    )
    assert smoke.status in {"OK", "ABSTAINED"}
    assert all(
        claim.resource_id == evidence.resource_id
        for assessment in smoke.assessments
        for claim in assessment.claims
        for evidence in assessment.evidence_by_claim.get(claim.claim_id, [])
    )


def test_readme_target_is_reported_but_not_a_hard_admission_gate(tmp_path: Path) -> None:
    baseline = _baseline(tmp_path, 2)
    next((baseline / "github_readme").glob("*.json")).unlink()
    output = tmp_path / "stage"
    resource_config = load_resource_config(output_root=output)
    build_dataset(baseline, resource_config)
    quality = validate_expansion(
        baseline, output / "data/processed/resources.jsonl", output_root=output, write_report=False,
    )
    assert quality["readme_coverage"] == 0.5
    assert quality["readme_target_met"] is False
    assert quality["valid"] is True


def test_expansion_report_reads_recorded_wall_clock_timings(tmp_path: Path) -> None:
    baseline = _baseline(tmp_path, 2)
    output = tmp_path / "stage"
    build_dataset(baseline, load_resource_config(output_root=output))
    record_duration(output, "collection", 1.23456)
    record_duration(output, "phase2", 2.5)
    record_duration(output, "phase3", 3.0)
    record_duration(output, "phase2-validation", 0.2)
    record_duration(output, "phase3-validation", 0.3)
    record_duration(output, "expansion-validation", 0.25)
    record_duration(output, "smoke", 0.125)

    quality = validate_expansion(
        baseline, output / "data/processed/resources.jsonl", output_root=output, write_report=True,
    )

    assert quality["performance"] == {
        "collection_seconds": 1.235,
        "resource_build_seconds": 2.5,
        "evidence_build_seconds": 3.0,
        "resource_validation_seconds": 0.2,
        "evidence_validation_seconds": 0.3,
        "expansion_validation_seconds": 0.25,
        "validation_seconds": 0.75,
        "smoke_seconds": 0.125,
    }
    report = (output / "data/reports/s3_5_resource_expansion_report.md").read_text(encoding="utf-8")
    assert "Phase 3 build: 3.0 seconds" in report


def test_weak_relevance_review_is_auditable_and_never_removes_resources(tmp_path: Path) -> None:
    run = tmp_path / "weak-run"
    fixtures = [
        (1, "AI internship directory", [], "General listings", "single_short_description_term"),
        (2, "General developer resource", [], "A machine learning tutorial", "readme_only_match"),
        (3, "General software project", [], "No domain evidence", "score_at_or_below_0_30"),
        (4, "Deep learning framework", ["machine-learning"], "PyTorch machine learning examples", None),
    ]
    for repo_id, description, topics, readme_text, _ in fixtures:
        full_name = f"owner/repo-{repo_id}"
        metadata = {
            "id": repo_id, "name": f"repo-{repo_id}", "full_name": full_name,
            "description": description, "html_url": f"https://github.com/{full_name}",
            "owner": {"login": "owner", "type": "Organization"}, "language": "Python",
            "topics": topics, "stargazers_count": 1, "forks_count": 0,
            "watchers_count": 1, "open_issues_count": 0, "license": {"spdx_id": "MIT"},
            "archived": False, "fork": False, "default_branch": "main",
            "created_at": STAMP, "updated_at": STAMP, "pushed_at": STAMP,
        }
        _wrapper(run, "github_repository", full_name, metadata)
        _wrapper(run, "github_readme", full_name, {
            "api_response": {"sha": f"sha-{repo_id}"}, "decoded_text": readme_text,
        })
    output = tmp_path / "stage"
    result = build_dataset(run, load_resource_config(output_root=output))
    quality = validate_expansion(
        run, output / "data/processed/resources.jsonl", output_root=output, write_report=True,
    )
    review = [
        json.loads(line)
        for line in (output / "data/review/expansion/weak_relevance_candidates.jsonl")
        .read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    by_id = {item["resource_id"]: item for item in review}

    assert result["final_resources"] == quality["resource_count"] == 4
    assert quality["weak_relevance_count"] == 3
    assert "single_short_description_term" in by_id["github:1"]["review_reasons"]
    assert by_id["github:1"]["matched_terms"]["description"] == ["ai"]
    assert by_id["github:1"]["matched_field"] == ["description"]
    assert by_id["github:1"]["contribution"] == {"description": 0.3}
    assert "readme_only_match" in by_id["github:2"]["review_reasons"]
    assert by_id["github:2"]["matched_field"] == ["readme"]
    assert "score_at_or_below_0_30" in by_id["github:3"]["review_reasons"]
    assert "github:4" not in by_id


def test_curated_expansion_preserves_every_identity_except_explicit_exclusions(tmp_path: Path) -> None:
    run = _baseline(tmp_path, 5)
    output = tmp_path / "curated-stage"
    resource_config = load_resource_config(output_root=output)
    baseline_resources = tmp_path / "all-stage-resources.jsonl"
    baseline_resources.write_text(
        "".join(json.dumps({"resource_id": f"github:{index}"}) + "\n" for index in range(1, 6)),
        encoding="utf-8",
    )
    curation_path = tmp_path / "curation.yaml"
    curation_path.write_text(yaml.safe_dump({
        "curation_manifest_version": "1.0.0",
        "manifest_id": "fixture_final_v1",
        "input": {"raw_run_id": run.name, "resource_build_id": build_id_for(run, resource_config)},
        "decision_scope": "exclude_only",
        "exclusions": [
            {"resource_id": "github:2", "reason_code": "manual", "reason": "Reviewed exclusion."},
            {"resource_id": "github:4", "reason_code": "manual", "reason": "Reviewed exclusion."},
        ],
    }, sort_keys=False), encoding="utf-8")
    curation = load_curation_manifest(curation_path)
    resource_result = build_dataset(run, resource_config, curation=curation)
    evidence_config = load_evidence_config(
        output_root=output, resource_manifest=output / "data/metadata/build_manifest.json",
    )
    evidence_result = build_corpus(
        output / "data/processed/resources.parquet", run, evidence_config, output_root=output,
    )
    quality = validate_expansion(
        run, output / "data/processed/resources.jsonl", baseline_resources=baseline_resources,
        output_root=output, write_report=True, require_evidence=True, curation=curation,
    )

    assert resource_result["final_resources"] == 3
    assert evidence_result["documents"] == 6
    assert quality["valid"] is True
    assert quality["baseline_ids_preserved"] == 3
    assert quality["missing_baseline_ids"] == []
    assert quality["curation_exclusions_absent_from_resources"] == 2
    assert quality["curation_exclusions_present_in_documents"] == []
    assert quality["curation_exclusions_present_in_chunks"] == []
    documents = read_jsonl(output / "data/evidence/documents.jsonl")
    chunks = read_jsonl(output / "data/evidence/chunks.jsonl")
    assert not {"github:2", "github:4"} & {row["resource_id"] for row in [*documents, *chunks]}
