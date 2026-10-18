from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from data_pipeline.resource_dataset.schema import Resource
from evaluation import core
from evaluation.core import (
    FrozenIdentity,
    assert_frozen_identity,
    build_experiment_identity,
    evaluation_code_hash,
    load_and_validate_assets,
    load_retrieval_benchmark,
    load_verification_benchmark,
    benchmark_hash,
    metadata_matches,
    prepare_retrieval_candidates,
    render_retrieval_sheet,
    render_verification_blind_sheet,
    validate_retrieval_references,
    validate_verification_candidate,
)
from evaluation.s4_lite import _latency_summary, _parser


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = ROOT / "evaluation" / "benchmarks" / "s4_lite_retrieval.yaml"
VERIFICATION_PATH = ROOT / "evaluation" / "benchmarks" / "s4_lite_verification.jsonl"


def synthetic_verification_payload(label: str = "SUPPORTED") -> dict:
    """Self-authored schema fixture, NOT an empirical/human Gold annotation.

    The annotation_status value exercises the historical schema contract only.
    This object is never exported as a benchmark or used for reported metrics.
    """
    resource_id = "github:9000000000001"
    claim = {
        "claim_id": core.claim_id_for(
            resource_id, "uses_framework", resource_id, "uses", "PyTorch", "rule_v1"
        ),
        "resource_id": resource_id, "claim_type": "uses_framework",
        "subject": resource_id, "predicate": "uses", "object": "PyTorch",
        "text": "Synthetic example uses PyTorch.", "generation_method": "rule_v1",
    }
    return {
        "benchmark_version": "1.0.0", "case_id": "synthetic_schema_001",
        "source_candidate_id": "synthetic_candidate_001",
        "source_query_case": "synthetic_query_001",
        "evaluation_reference_time": "2026-09-16T00:00:00Z",
        "resource_id": resource_id, "repository": "example/synthetic-schema-fixture",
        "claim": claim,
        "evidence": [{
            "chunk_id": "ch_0000000000000001", "evidence_id": "ev_0000000000000001",
            "resource_id": resource_id, "source_id": resource_id + ":readme",
            "text": "Synthetic engineering fixture: this example uses PyTorch.",
            "source_url": "https://github.com/example/synthetic-schema-fixture/blob/main/README.md",
            "evidence_source_type": "github_readme", "source_type": "community",
            "source_reliability": 0.7, "source_reliability_method": "heuristic_v0",
            "retrieval_score": 0.5,
        }],
        "gold_label": label,
        "human_rationale": "Synthetic enum/schema exercise, not an empirical human annotation.",
        "annotation_status": "human_confirmed",
    }


def make_resource(
    numeric_id: int,
    *,
    name: str,
    description: str,
    topics: list[str],
) -> Resource:
    full_name = f"example/{name}"
    return Resource(
        resource_id=f"github:{numeric_id}",
        resource_type="github_project",
        name=name,
        canonical_name=name,
        full_name=full_name,
        description=description,
        url=f"https://github.com/{full_name}",
        canonical_url=f"https://github.com/{full_name}",
        source_platform="github",
        source_id=f"github:{numeric_id}",
        source_type="community",
        source_reliability=0.7,
        source_reliability_method="heuristic_v0",
        owner_name="example",
        owner_type="Organization",
        organization_name="example",
        language="Python",
        topics=topics,
        license="MIT",
        metadata_completeness=1.0,
        domain_relevance_score=1.0,
        data_quality_score=1.0,
        raw_source_id=full_name,
        retrieved_at="2026-09-16T00:00:00Z",
        record_version="v0.1.0",
    )


def frozen_identity(**updates: object) -> FrozenIdentity:
    values = {
        "git_tag": core.EXPECTED_TAG,
        "git_tag_target": core.EXPECTED_TAG_TARGET,
        "raw_run_id": core.EXPECTED_RAW_RUN_ID,
        "curation_manifest_id": core.EXPECTED_CURATION_ID,
        "curation_manifest_version": core.EXPECTED_CURATION_VERSION,
        "curation_manifest_hash": core.EXPECTED_CURATION_HASH,
        "resource_build_id": core.EXPECTED_RESOURCE_BUILD_ID,
        "evidence_build_id": core.EXPECTED_EVIDENCE_BUILD_ID,
        "resource_count": 200,
        "document_count": 400,
        "chunk_count": 5902,
        "excluded_resources_absent": True,
        "orphan_documents": 0,
        "orphan_chunks": 0,
        "identity_matches_promoted_canonical": True,
        "data_root": "data/frozen",
    }
    values.update(updates)
    return FrozenIdentity.model_validate(values)


def test_cli_requires_explicit_data_root() -> None:
    with pytest.raises(SystemExit):
        _parser().parse_args(["preflight"])
    parsed = _parser().parse_args(["preflight", "--data-root", "data/frozen"])
    assert parsed.data_root == Path("data/frozen")


def test_missing_explicit_root_does_not_fallback(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="explicit --data-root"):
        load_and_validate_assets(tmp_path / "missing", project_root=ROOT)


def test_frozen_identity_accepts_exact_contract_and_rejects_one_mismatch() -> None:
    assert_frozen_identity(frozen_identity())
    with pytest.raises(ValueError, match="resource_count"):
        assert_frozen_identity(frozen_identity(resource_count=199))


def test_draft_benchmark_has_actual_structured_intent_schema() -> None:
    benchmark = load_retrieval_benchmark(BENCHMARK_PATH)
    assert len(benchmark.cases) == 10
    assert all(case.gold_intent.language == [] for case in benchmark.cases)
    assert all(case.acceptable_resource_ids for case in benchmark.cases)
    assert benchmark.benchmark_status == "human_confirmed"


def test_duplicate_retrieval_case_id_is_rejected() -> None:
    payload = core.read_yaml(BENCHMARK_PATH)
    payload["cases"][1]["case_id"] = payload["cases"][0]["case_id"]
    with pytest.raises(ValueError, match="duplicate retrieval case_id"):
        core.RetrievalBenchmark.model_validate(payload)


def test_confirmed_benchmark_allows_formal_metrics_command() -> None:
    benchmark = load_retrieval_benchmark(BENCHMARK_PATH)
    known = {
        resource_id
        for case in benchmark.cases
        for resource_id in case.acceptable_resource_ids
    }
    validate_retrieval_references(benchmark, known, require_confirmed=True)
    parsed = _parser().parse_args(["evaluate", "--data-root", "data/frozen"])
    assert parsed.command == "evaluate"


@pytest.mark.parametrize("label", ["SUPPORTED", "UNKNOWN", "CONTRADICTED"])
def test_synthetic_verification_schema_accepts_label_enum(label: str) -> None:
    case = core.VerificationGoldCase.model_validate(synthetic_verification_payload(label))
    assert case.gold_label == label


def test_undistributed_historical_gold_fails_explicitly() -> None:
    assert not VERIFICATION_PATH.exists()
    with pytest.raises(FileNotFoundError, match="not distributed with this showcase"):
        load_verification_benchmark(VERIFICATION_PATH)


def test_formal_evaluation_cannot_substitute_data_for_missing_gold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from evaluation import s4_lite

    def reject_data_access(*args: object, **kwargs: object) -> None:
        pytest.fail("Missing historical Gold must fail before corpus access")

    monkeypatch.setattr(s4_lite, "load_and_validate_assets", reject_data_access)
    args = _parser().parse_args(["evaluate", "--data-root", "data/not-distributed"])
    with pytest.raises(FileNotFoundError, match="not distributed with this showcase"):
        s4_lite.run(args, root=ROOT)


def test_verification_gold_rejects_noncanonical_claim_identity() -> None:
    payload = synthetic_verification_payload()
    payload["claim"]["claim_id"] = "claim_0000000000000000"
    with pytest.raises(ValueError, match="canonical Claim identity"):
        core.VerificationGoldCase.model_validate(payload)


def test_benchmark_hash_includes_verification_gold() -> None:
    benchmark = load_retrieval_benchmark(BENCHMARK_PATH)
    cases = [synthetic_verification_payload()]
    original = benchmark_hash(benchmark, cases)
    changed = [dict(item) for item in cases]
    changed[0] = {**changed[0], "human_rationale": "changed only for hash test"}
    assert original != benchmark_hash(benchmark, changed)


def test_benchmark_hash_includes_abstention_cases() -> None:
    benchmark = load_retrieval_benchmark(BENCHMARK_PATH)
    verification = [synthetic_verification_payload()]
    abstention = core.load_abstention_benchmark(
        ROOT / "evaluation" / "benchmarks" / "s4_lite_abstention.yaml"
    )
    original = benchmark_hash(benchmark, verification, abstention)
    changed = abstention.model_copy(deep=True)
    changed.cases[0].description = "changed only for hash test"
    assert original != benchmark_hash(benchmark, verification, changed)


def test_runtime_summary_reports_average_and_median() -> None:
    rows = [
        {
            "query_latency_ms": value,
            "core_latency_ms": value + 1,
            "generation_latency_ms": value + 2,
            "total_latency_ms": value + 3,
        }
        for value in (1.0, 3.0, 5.0, 7.0, 9.0)
    ]
    summary = _latency_summary(rows)
    assert summary["query"] == {"average_ms": 5.0, "median_ms": 5.0}
    assert summary["total"] == {"average_ms": 8.0, "median_ms": 8.0}


def test_unknown_resource_reference_is_rejected() -> None:
    benchmark = load_retrieval_benchmark(BENCHMARK_PATH)
    changed = benchmark.model_copy(deep=True)
    changed.cases[0].acceptable_resource_ids.append("github:999999999")
    with pytest.raises(ValueError, match="unknown Resource IDs"):
        validate_retrieval_references(changed, {"github:1"})


def test_code_hash_is_deterministic_and_identifies_uncommitted_source(tmp_path: Path) -> None:
    package = tmp_path / "evaluation"
    package.mkdir()
    (package / "a.py").write_text("x = 1\n", encoding="utf-8")
    first = evaluation_code_hash(tmp_path)
    assert first == evaluation_code_hash(tmp_path)
    (package / "a.py").write_text("x = 2\n", encoding="utf-8")
    assert first != evaluation_code_hash(tmp_path)


def test_experiment_identity_marks_dirty_commit_as_base_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(core, "git_state", lambda root: ("a" * 40, True, core.EXPECTED_TAG_TARGET))
    monkeypatch.setattr(core, "evaluation_code_hash", lambda root: "sha256:" + "1" * 64)
    monkeypatch.setattr(core, "evaluation_config_hash", lambda root: "sha256:" + "2" * 64)
    identity = build_experiment_identity(
        project_root=ROOT,
        benchmark=load_retrieval_benchmark(BENCHMARK_PATH),
        frozen=frozen_identity(),
    )
    assert identity.git_dirty is True
    assert identity.git_commit_role == "base_commit_only"
    assert identity.evaluation_code_hash == "sha256:" + "1" * 64
    assert "base commit" in identity.source_identity_note


def test_metadata_matching_uses_traceable_terms_and_fields() -> None:
    resource = make_resource(
        1, name="vision-course", description="A computer vision tutorial",
        topics=["computer-vision"],
    )
    terms, fields = metadata_matches(resource, ["computer vision"])
    assert terms == ["computer vision"]
    assert fields == ["description", "topics"]


def test_retrieval_candidates_record_both_sources_without_creating_gold() -> None:
    benchmark = load_retrieval_benchmark(BENCHMARK_PATH).model_copy(deep=True)
    benchmark.candidate_pool = {"retriever_limit": 1, "metadata_supplement_limit": 2}
    resources = [
        make_resource(1, name="ml-one", description="machine learning fundamentals", topics=["machine-learning"]),
        make_resource(2, name="ml-two", description="machine learning examples", topics=["machine-learning"]),
        make_resource(3, name="other", description="unrelated material", topics=[]),
    ]
    cases = prepare_retrieval_candidates(benchmark, resources)
    first_case = cases[0]
    assert first_case["candidates"][0]["candidate_sources"] == [
        "retriever_top10", "metadata_match",
    ]
    assert any(
        item["candidate_sources"] == ["metadata_match"]
        for item in first_case["candidates"]
    )
    assert all(item["acceptable"] is None for item in first_case["candidates"])
    sheet = render_retrieval_sheet([first_case])
    assert "retriever_top10 + metadata_match" in sheet
    assert "Current ranking must not be copied into Gold" in sheet


def test_retrieval_candidate_generation_is_deterministic() -> None:
    benchmark = load_retrieval_benchmark(BENCHMARK_PATH)
    resources = [
        make_resource(2, name="two", description="pytorch deep learning", topics=["pytorch"]),
        make_resource(1, name="one", description="machine learning", topics=["machine-learning"]),
    ]
    assert prepare_retrieval_candidates(benchmark, resources) == prepare_retrieval_candidates(
        benchmark, list(reversed(resources))
    )


def test_verification_candidate_requires_exact_same_resource_canonical_chunk() -> None:
    chunk = SimpleNamespace(
        chunk_id="ch_1234567890abcdef",
        evidence_id="ev_1234567890abcdef",
        resource_id="github:1",
        source_id="github:1:readme",
        text="This project uses PyTorch.",
    )
    candidate = {
        "resource_id": "github:1",
        "claim": {"resource_id": "github:1"},
        "evidence": [{
            "chunk_id": chunk.chunk_id,
            "evidence_id": chunk.evidence_id,
            "resource_id": chunk.resource_id,
            "source_id": chunk.source_id,
            "text": chunk.text,
        }],
    }
    validate_verification_candidate(candidate, {chunk.chunk_id: chunk})
    candidate["evidence"][0]["resource_id"] = "github:2"
    with pytest.raises(ValueError, match="cross-resource"):
        validate_verification_candidate(candidate, {chunk.chunk_id: chunk})


def test_unknown_chunk_and_changed_evidence_text_are_rejected() -> None:
    chunk = SimpleNamespace(
        chunk_id="ch_1234567890abcdef", evidence_id="ev_1234567890abcdef",
        resource_id="github:1", source_id="github:1:readme", text="Canonical text",
    )
    candidate = {
        "resource_id": "github:1", "claim": {"resource_id": "github:1"},
        "evidence": [{
            "chunk_id": chunk.chunk_id, "evidence_id": chunk.evidence_id,
            "resource_id": chunk.resource_id, "source_id": chunk.source_id,
            "text": "Changed text",
        }],
    }
    with pytest.raises(ValueError, match="differs from canonical"):
        validate_verification_candidate(candidate, {chunk.chunk_id: chunk})
    with pytest.raises(ValueError, match="unknown canonical chunk_id"):
        validate_verification_candidate(candidate, {})


def test_blind_sheet_prioritizes_reading_and_hides_system_prediction() -> None:
    case = {
        "case_id": "verification_candidate_001",
        "source_query_case": "retrieval_001",
        "resource_id": "github:1",
        "repository": "example/repo",
        "claim": {
            "claim_id": "claim_1234567890abcdef",
            "claim_type": "covers_topic",
            "text": "repo covers Machine Learning.",
        },
        "evidence": [{
            "evidence_id": "ev_1234567890abcdef",
            "chunk_id": "ch_1234567890abcdef",
            "source_id": "github:1:readme",
            "source_url": "https://github.com/example/repo/blob/main/README.md",
            "evidence_source_type": "github_readme",
            "text": "A clear human-readable Evidence excerpt.",
        }],
        "system_verification_label": "SUPPORTED",
        "system_verification_rationale": "SECRET_SYSTEM_RATIONALE",
    }
    sheet = render_verification_blind_sheet([case])
    assert "A clear human-readable Evidence excerpt." in sheet
    assert "[ ] SUPPORTED" in sheet and "[ ] UNKNOWN" in sheet
    assert "SECRET_SYSTEM_RATIONALE" not in sheet
    assert "system_verification_label" not in sheet
    assert sheet.index("**Evidence:**") < sheet.index("<details>")
