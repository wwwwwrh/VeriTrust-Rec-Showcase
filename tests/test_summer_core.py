from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from data_pipeline.evidence.schema import EvidenceChunk, EvidenceDocument
from data_pipeline.resource_dataset.schema import Resource
from recommender.abstention.policy import decide_candidate
from recommender.config import SummerConfig, load_config
from recommender.data import EvidenceCorpus, load_evidence_corpus, load_resources
from recommender.errors import CandidateErrorCode, GlobalErrorCode, PipelineError
from recommender.kg.graph import KnowledgeGraph
from recommender.llm import LocalLLMClient, MockLLMClient
from recommender.pipeline import run_core_pipeline
from recommender.retrieval.evidence import EvidenceRetriever
from recommender.retrieval.retriever import TfidfRetriever
from recommender.schemas import (
    Candidate,
    Claim,
    ConfidenceBreakdown,
    RetrievedEvidence,
    StructuredIntent,
    VerificationResult,
)
from recommender.uncertainty.scorer import estimate_confidence
from recommender.verification.claim_extractor import claim_id_for, generate_claims
from recommender.verification.verifier import summarize_verification, verify_claim


BUILD_ID = "evidence_build_1234567890abcdef"
NOW = datetime(2026, 9, 12, tzinfo=timezone.utc)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def make_resource(index: int, **overrides) -> Resource:
    data = {
        "resource_id": f"github:{1000 + index}",
        "resource_type": "github_project",
        "name": f"resource-{index}",
        "canonical_name": f"resource-{index}",
        "full_name": f"owner/resource-{index}",
        "description": "A machine learning tutorial with practical examples",
        "url": f"https://github.com/owner/resource-{index}",
        "canonical_url": f"https://github.com/owner/resource-{index}",
        "source_platform": "github",
        "source_id": f"github:{1000 + index}",
        "source_type": "community",
        "source_reliability": 0.7,
        "source_reliability_method": "heuristic_v0",
        "owner_name": "owner",
        "owner_type": "Organization",
        "organization_name": "owner",
        "language": "Python",
        "topics": ["machine-learning", "deep-learning"],
        "frameworks": [],
        "skills": [],
        "prerequisites": [],
        "difficulty": "unknown",
        "learning_style": [],
        "tasks": [],
        "stars": 100,
        "forks": 10,
        "watchers": 20,
        "open_issues": 1,
        "license": "MIT",
        "is_archived": False,
        "is_fork": False,
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2026-08-01T00:00:00Z",
        "pushed_at": "2026-08-15T00:00:00Z",
        "activity_score": 0.8,
        "freshness_score": 0.9,
        "metadata_completeness": 1.0,
        "domain_relevance_score": 1.0,
        "data_quality_score": 1.0,
        "raw_source_id": f"owner/resource-{index}",
        "retrieved_at": "2026-09-01T00:00:00Z",
        "record_version": "v0.1.0",
    }
    data.update(overrides)
    return Resource.model_validate(data)


def provenance_for(resources: list[Resource]) -> list[dict]:
    output = []
    for resource in resources:
        for field in ("topics", "frameworks", "language", "owner_name"):
            value = getattr(resource, field)
            if value:
                output.append({
                    "resource_id": resource.resource_id,
                    "field": field,
                    "source_url": resource.url,
                    "source_hash": "a" * 64,
                    "value": value,
                })
    return output


def metadata_text(
    *, topics: list[str] | None = None, license_id: str | None = "MIT",
    archived: bool = False, pushed_at: str | None = "2026-08-15T00:00:00Z",
) -> str:
    values = {
        "repository_id": 1000,
        "topics": topics or ["machine-learning", "deep-learning"],
        "license_spdx_id": license_id,
        "is_archived": archived,
        "pushed_at": pushed_at,
    }
    return "\n".join(f"{key}: {json.dumps(value)}" for key, value in values.items())


def make_document(resource: Resource, kind: str, text: str) -> EvidenceDocument:
    suffix = "metadata" if kind == "repo_metadata" else "readme"
    evidence_id = "ev_" + _sha(resource.resource_id + suffix)[:16]
    return EvidenceDocument(
        evidence_id=evidence_id,
        document_key=f"{resource.resource_id}:{suffix}",
        evidence_build_id=BUILD_ID,
        resource_id=resource.resource_id,
        evidence_source_type=kind,
        source_type=resource.source_type,
        source_id=f"{resource.resource_id}:{suffix}",
        raw_source_id=resource.raw_source_id,
        source_url=resource.url,
        raw_source_hash="b" * 64,
        title=f"{resource.name} {suffix}",
        text=text,
        text_format="structured_metadata" if suffix == "metadata" else "markdown",
        content_hash="sha256:" + _sha(text),
        retrieved_at=resource.retrieved_at,
        source_reliability=resource.source_reliability,
        source_reliability_method="heuristic_v0",
        extraction_method="api_metadata_serialization" if suffix == "metadata" else "raw_snapshot",
        run_id="run1",
        collector_version="0.1.0",
        snapshot_schema_version="1.0.0",
        evidence_schema_version="0.1.0",
        normalizer_version="v0.1.0",
        markdown_parser_version="v0.1.0",
        metadata={
            "raw_snapshot_path": "data/raw/example.json",
            "raw_provenance": {},
            "resource_discovery_provenance": {},
            "resource_record_version": "v0.1.0",
            "normalization_warnings": [],
        },
    )


def make_chunk(resource: Resource, kind: str, text: str, index: int = 0) -> EvidenceChunk:
    document = make_document(resource, kind, text)
    return EvidenceChunk(
        chunk_id="ch_" + _sha(document.evidence_id + str(index) + text)[:16],
        evidence_id=document.evidence_id,
        evidence_build_id=BUILD_ID,
        resource_id=resource.resource_id,
        evidence_source_type=kind,
        source_type=resource.source_type,
        source_id=document.source_id,
        raw_source_id=resource.raw_source_id,
        source_url=resource.url,
        raw_source_hash="b" * 64,
        section_path=["__metadata__" if kind == "repo_metadata" else "Overview"],
        section_title="__metadata__" if kind == "repo_metadata" else "Overview",
        section_level=0 if kind == "repo_metadata" else 1,
        chunk_index=index,
        source_start_line=1,
        source_end_line=max(1, text.count("\n") + 1),
        source_start_char=0,
        source_end_char=len(text),
        text=text,
        char_count=len(text),
        content_hash="sha256:" + _sha(text),
        contains_code=False,
        contains_table=False,
        retrieved_at=resource.retrieved_at,
        source_reliability=resource.source_reliability,
        source_reliability_method="heuristic_v0",
        run_id="run1",
        collector_version="0.1.0",
        snapshot_schema_version="1.0.0",
        evidence_schema_version="0.1.0",
        normalizer_version="v0.1.0",
        markdown_parser_version="v0.1.0",
        chunker_version="v0.1.1",
        metadata={},
    )


def make_corpus(resources: list[Resource]) -> EvidenceCorpus:
    documents: list[EvidenceDocument] = []
    chunks: list[EvidenceChunk] = []
    for resource in resources:
        meta = metadata_text(topics=resource.topics, license_id=resource.license)
        readme = "This practical machine learning tutorial is implemented using PyTorch."
        documents.extend([make_document(resource, "repo_metadata", meta), make_document(resource, "github_readme", readme)])
        chunks.extend([make_chunk(resource, "repo_metadata", meta), make_chunk(resource, "github_readme", readme)])
    return EvidenceCorpus(documents, chunks, {"evidence_build_id": BUILD_ID})


@pytest.fixture
def summer_config() -> SummerConfig:
    config = load_config()
    return config.model_copy(update={
        "logging": config.logging.model_copy(update={"save_sessions": False}),
    })


def candidate(resource: Resource, score: float = 0.8) -> Candidate:
    return Candidate(resource_id=resource.resource_id, title=resource.name, url=resource.url, retrieval_score=score)


def claim(resource: Resource, claim_type: str, obj: str) -> Claim:
    predicates = {
        "uses_framework": "uses", "covers_topic": "covers",
        "is_maintained": "is", "has_license": "has",
    }
    claim_id = claim_id_for(resource.resource_id, claim_type, resource.resource_id, predicates[claim_type], obj)
    return Claim(
        claim_id=claim_id, resource_id=resource.resource_id, claim_type=claim_type,
        subject=resource.resource_id, predicate=predicates[claim_type], object=obj,
        text=f"{resource.name} {predicates[claim_type]} {obj}",
    )


def runtime_evidence(chunk: EvidenceChunk, score: float = 0.8) -> RetrievedEvidence:
    return EvidenceRetriever._to_runtime(chunk, score)


def test_runtime_schemas_forbid_unknown_fields_and_bad_identity():
    with pytest.raises(ValidationError):
        StructuredIntent(topics=[], invented=True)
    with pytest.raises(ValidationError):
        Candidate(resource_id="owner/repo", title="x", url="https://example.com", retrieval_score=0.5)


def test_structured_intent_normalizes_and_deduplicates_lists():
    intent = StructuredIntent(topics=[" Machine Learning ", "machine learning", ""])
    assert intent.topics == ["Machine Learning"]


def test_resource_loader_supports_jsonl_and_parquet(tmp_path: Path):
    resource = make_resource(1)
    jsonl = tmp_path / "resources.jsonl"
    jsonl.write_text(json.dumps(resource.model_dump(mode="json")) + "\n", encoding="utf-8")
    parquet = tmp_path / "resources.parquet"
    pq.write_table(pa.Table.from_pylist([resource.model_dump(mode="json")]), parquet)
    assert load_resources(jsonl)[0].resource_id == resource.resource_id
    assert load_resources(parquet)[0].resource_id == resource.resource_id


def test_resource_loader_rejects_duplicate_ids(tmp_path: Path):
    resource = make_resource(1).model_dump(mode="json")
    path = tmp_path / "resources.jsonl"
    path.write_text("\n".join((json.dumps(resource), json.dumps(resource))) + "\n", encoding="utf-8")
    with pytest.raises(PipelineError) as caught:
        load_resources(path)
    assert caught.value.code == GlobalErrorCode.RESOURCE_DATASET_INVALID


def test_tfidf_is_positive_top_k_and_deterministic():
    resources = [
        make_resource(1, description="PyTorch transformer tutorial"),
        make_resource(2, description="TensorFlow computer vision"),
        make_resource(3, description="database systems"),
    ]
    intent = StructuredIntent(topics=["transformer"], framework_preferences=["PyTorch"])
    retriever = TfidfRetriever(resources)
    first = retriever.retrieve(intent, top_k=1)
    second = retriever.retrieve(intent, top_k=1)
    assert first == second and len(first) == 1
    assert first[0].resource_id == resources[0].resource_id
    assert first[0].retrieval_score > 0


def test_tfidf_does_not_pad_zero_score_results():
    resources = [make_resource(1, description="machine learning"), make_resource(2, description="database")]
    results = TfidfRetriever(resources).retrieve(StructuredIntent(topics=["machine"]), top_k=5)
    assert len(results) == 1


def test_kg_semantics_and_provenance_are_correct():
    resource = make_resource(1, frameworks=["PyTorch"])
    kg = KnowledgeGraph.build([resource], provenance_for([resource]))
    assert kg.has_fact(resource.resource_id, "covers", "machine-learning")
    assert kg.has_fact(resource.resource_id, "uses", "PyTorch")
    assert kg.has_fact(resource.resource_id, "written_in", "Python")
    assert all(item.source_reference.startswith("resource_provenance:") for item in kg.facts)
    assert not any(item.object_type == "Topic" and item.predicate == "uses" for item in kg.facts)


def test_empty_frameworks_do_not_create_uses_edges():
    resource = make_resource(1, frameworks=[])
    kg = KnowledgeGraph.build([resource], provenance_for([resource]))
    assert not any(item.predicate == "uses" for item in kg.facts)


def test_claims_are_bound_deterministic_and_never_general():
    resource = make_resource(1)
    kg = KnowledgeGraph.build([resource], provenance_for([resource]))
    intent = StructuredIntent(topics=["machine-learning"], framework_preferences=["PyTorch"])
    first = generate_claims(candidate(resource), intent, resource, kg)
    second = generate_claims(candidate(resource), intent, resource, kg)
    assert first == second
    assert all(item.resource_id == resource.resource_id for item in first)
    assert {item.claim_type for item in first} == {"uses_framework", "covers_topic", "is_maintained", "has_license"}


def test_evidence_retrieval_is_same_resource_and_preserves_real_ids():
    left, right = make_resource(1), make_resource(2)
    chunks = [
        make_chunk(left, "github_readme", "We are using PyTorch for all models."),
        make_chunk(right, "github_readme", "We are using PyTorch for all models."),
    ]
    result = EvidenceRetriever(chunks).retrieve(claim(left, "uses_framework", "PyTorch"), top_k=3)
    assert result and all(item.resource_id == left.resource_id for item in result)
    assert result[0].chunk_id == chunks[0].chunk_id
    assert result[0].evidence_id == chunks[0].evidence_id


def test_cross_resource_evidence_is_rejected(summer_config: SummerConfig):
    left, right = make_resource(1), make_resource(2)
    foreign = runtime_evidence(make_chunk(right, "github_readme", "using PyTorch"))
    with pytest.raises(PipelineError) as caught:
        verify_claim(
            claim(left, "uses_framework", "PyTorch"), [foreign],
            config=summer_config.verification, evaluation_time=NOW,
        )
    assert caught.value.code == CandidateErrorCode.EVIDENCE_ATTRIBUTION_ERROR


def test_framework_supported_and_mention_only_unknown(summer_config: SummerConfig):
    resource = make_resource(1)
    item = claim(resource, "uses_framework", "PyTorch")
    supported = runtime_evidence(make_chunk(resource, "github_readme", "The models are implemented using PyTorch."))
    mention = runtime_evidence(make_chunk(resource, "github_readme", "PyTorch"))
    assert verify_claim(item, [supported], config=summer_config.verification, evaluation_time=NOW).label == "SUPPORTED"
    assert verify_claim(item, [mention], config=summer_config.verification, evaluation_time=NOW).label == "UNKNOWN"


def test_other_framework_does_not_contradict_target(summer_config: SummerConfig):
    resource = make_resource(1)
    evidence = runtime_evidence(make_chunk(resource, "github_readme", "This project uses TensorFlow."))
    result = verify_claim(
        claim(resource, "uses_framework", "PyTorch"), [evidence],
        config=summer_config.verification, evaluation_time=NOW,
    )
    assert result.label == "UNKNOWN"


def test_explicit_framework_negation_is_contradicted(summer_config: SummerConfig):
    resource = make_resource(1)
    evidence = runtime_evidence(make_chunk(resource, "github_readme", "This project does not use PyTorch."))
    result = verify_claim(
        claim(resource, "uses_framework", "PyTorch"), [evidence],
        config=summer_config.verification, evaluation_time=NOW,
    )
    assert result.label == "CONTRADICTED"


def test_topic_metadata_support_references_the_metadata_chunk(summer_config: SummerConfig):
    resource = make_resource(1)
    metadata = make_chunk(resource, "repo_metadata", metadata_text(topics=["transformers"]))
    unrelated = make_chunk(resource, "github_readme", "Welcome to the repository.")
    result = verify_claim(
        claim(resource, "covers_topic", "transformers"),
        [runtime_evidence(unrelated), runtime_evidence(metadata)],
        config=summer_config.verification, evaluation_time=NOW,
    )
    assert result.label == "SUPPORTED"
    assert result.evidence_chunk_ids == [metadata.chunk_id]


def test_maintenance_uses_same_repo_metadata_chunk(summer_config: SummerConfig):
    resource = make_resource(1)
    metadata = make_chunk(resource, "repo_metadata", metadata_text(archived=False))
    unrelated = make_chunk(resource, "github_readme", "A popular project.")
    result = verify_claim(
        claim(resource, "is_maintained", "actively_maintained"),
        [runtime_evidence(unrelated), runtime_evidence(metadata)],
        config=summer_config.verification, evaluation_time=NOW,
    )
    assert result.label == "SUPPORTED"
    assert result.evidence_chunk_ids == [metadata.chunk_id]


def test_archived_metadata_contradicts_maintenance(summer_config: SummerConfig):
    resource = make_resource(1)
    metadata = runtime_evidence(make_chunk(resource, "repo_metadata", metadata_text(archived=True)))
    result = verify_claim(
        claim(resource, "is_maintained", "actively_maintained"), [metadata],
        config=summer_config.verification, evaluation_time=NOW,
    )
    assert result.label == "CONTRADICTED"


@pytest.mark.parametrize("license_id", ["MIT", "Apache-2.0", "BSD-3-Clause"])
def test_any_recognized_open_source_spdx_supports_has_license(license_id: str, summer_config: SummerConfig):
    resource = make_resource(1)
    evidence = runtime_evidence(make_chunk(resource, "repo_metadata", metadata_text(license_id=license_id)))
    result = verify_claim(
        claim(resource, "has_license", "open_source_license"), [evidence],
        config=summer_config.verification, evaluation_time=NOW,
    )
    assert result.label == "SUPPORTED"


def test_missing_or_different_unrecognized_license_is_unknown_not_contradicted(summer_config: SummerConfig):
    resource = make_resource(1)
    for value in (None, "NOASSERTION", "Custom-Internal"):
        evidence = runtime_evidence(make_chunk(resource, "repo_metadata", metadata_text(license_id=value)))
        result = verify_claim(
            claim(resource, "has_license", "open_source_license"), [evidence],
            config=summer_config.verification, evaluation_time=NOW,
        )
        assert result.label == "UNKNOWN"


def test_summary_coverage_and_unknown_denominator(summer_config: SummerConfig):
    resource = make_resource(1)
    kg = KnowledgeGraph.build([resource], provenance_for([resource]))
    claims = [claim(resource, "covers_topic", "machine-learning"), claim(resource, "uses_framework", "PyTorch")]
    evidence = runtime_evidence(make_chunk(resource, "repo_metadata", metadata_text()))
    results = [
        VerificationResult(claim_id=claims[0].claim_id, resource_id=resource.resource_id, label="SUPPORTED", score=0.9,
                           evidence_chunk_ids=[evidence.chunk_id], rationale="supported"),
        VerificationResult(claim_id=claims[1].claim_id, resource_id=resource.resource_id, label="UNKNOWN", score=0,
                           evidence_chunk_ids=[], rationale="unknown"),
    ]
    summary = summarize_verification(claims, {claims[0].claim_id: [evidence], claims[1].claim_id: []}, results, kg)
    assert summary.evidence_coverage == 0.5
    assert summary.verification_score == 0.5


def test_source_reliability_is_unique_by_evidence_source_and_only_used_evidence():
    resource = make_resource(1)
    kg = KnowledgeGraph.build([resource], provenance_for([resource]))
    first = make_chunk(resource, "github_readme", "using PyTorch", 0)
    second = make_chunk(resource, "github_readme", "PyTorch dependency", 1)
    unused = make_chunk(resource, "repo_metadata", metadata_text(), 0).model_copy(update={"source_reliability": 0.9})
    evidence = [runtime_evidence(first), runtime_evidence(second), runtime_evidence(unused)]
    item = claim(resource, "uses_framework", "PyTorch")
    result = VerificationResult(
        claim_id=item.claim_id, resource_id=resource.resource_id, label="SUPPORTED", score=0.9,
        evidence_chunk_ids=[first.chunk_id, second.chunk_id], rationale="supported",
    )
    summary = summarize_verification([item], {item.claim_id: evidence}, [result], kg)
    assert summary.unique_verification_sources == 1
    assert summary.source_reliability == 0.7


def test_confidence_formula_is_direct_and_unknown_lowers_score(summer_config: SummerConfig):
    result = estimate_confidence(
        retrieval_score=1, evidence_coverage=1, verification_score=0.5,
        kg_support=0.5, source_reliability=0.7, config=summer_config.confidence,
    )
    assert result.confidence == pytest.approx(0.76)
    assert result.method == "prototype_heuristic_v1"


def test_abstention_semantics(summer_config: SummerConfig):
    resource = make_resource(1)
    cand = candidate(resource)
    high = ConfidenceBreakdown(confidence=0.85, bucket="High", components={}, weights={})
    contradicted = VerificationResult(
        claim_id="claim_1234567890abcdef", resource_id=resource.resource_id,
        label="CONTRADICTED", score=0.99, evidence_chunk_ids=[], rationale="explicit",
    )
    assert decide_candidate(
        cand, high, evidence_coverage=1, verifications=[contradicted], local_errors=[],
        abstention=summer_config.abstention, ranking=summer_config.ranking,
    ).abstain
    assert not decide_candidate(
        cand, high, evidence_coverage=1, verifications=[], local_errors=[],
        abstention=summer_config.abstention, ranking=summer_config.ranking,
    ).abstain


def test_evidence_manifest_compatibility_ignores_git_commit(tmp_path: Path, summer_config: SummerConfig):
    resource = make_resource(1)
    text = metadata_text()
    document = make_document(resource, "repo_metadata", text)
    chunk = make_chunk(resource, "repo_metadata", text)
    documents_path, chunks_path = tmp_path / "documents.jsonl", tmp_path / "chunks.jsonl"
    documents_path.write_text(json.dumps(document.model_dump(mode="json")) + "\n", encoding="utf-8")
    chunks_path.write_text(json.dumps(chunk.model_dump(mode="json")) + "\n", encoding="utf-8")
    resource_manifest = tmp_path / "resource_manifest.json"
    resource_manifest.write_text(json.dumps({"build_id": "build_phase2"}), encoding="utf-8")
    manifest = {
        "manifest_role": "atomic_build_commit_marker",
        "evidence_build_id": BUILD_ID,
        "input_resource_build_id": "build_phase2",
        "document_count": 1,
        "chunk_count": 1,
        "evidence_schema_version": "0.1.0",
        "normalizer_version": "v0.1.0",
        "markdown_parser_version": "v0.1.0",
        "chunker_version": "v0.1.1",
        "effective_config_hash": "c" * 64,
        "code_version": "old-version",
        "git_commit": "not-current-head",
    }
    manifest_path = tmp_path / "evidence_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    corpus = load_evidence_corpus(
        documents_path=documents_path, chunks_path=chunks_path,
        evidence_manifest_path=manifest_path, resource_manifest_path=resource_manifest,
        resources=[resource], compatibility=summer_config.evidence_compatibility,
    )
    assert corpus.manifest["git_commit"] == "not-current-head"


def test_incompatible_resource_build_is_rejected(tmp_path: Path, summer_config: SummerConfig):
    resource = make_resource(1)
    manifest_path = tmp_path / "evidence.json"
    manifest_path.write_text(json.dumps({
        "manifest_role": "atomic_build_commit_marker", "evidence_build_id": BUILD_ID,
        "input_resource_build_id": "old", "evidence_schema_version": "0.1.0",
        "normalizer_version": "v0.1.0", "markdown_parser_version": "v0.1.0",
        "chunker_version": "v0.1.1", "effective_config_hash": "c" * 64,
    }), encoding="utf-8")
    resource_manifest = tmp_path / "resource.json"
    resource_manifest.write_text(json.dumps({"build_id": "current"}), encoding="utf-8")
    with pytest.raises(PipelineError) as caught:
        load_evidence_corpus(
            documents_path=tmp_path / "missing-docs", chunks_path=tmp_path / "missing-chunks",
            evidence_manifest_path=manifest_path, resource_manifest_path=resource_manifest,
            resources=[resource], compatibility=summer_config.evidence_compatibility,
        )
    assert caught.value.code == GlobalErrorCode.EVIDENCE_BUILD_INCOMPATIBLE


def test_mock_llm_is_local_deterministic_interface():
    client = MockLLMClient(["{}"])
    assert isinstance(client, LocalLLMClient)
    assert client.generate("prompt") == "{}"
    assert client.calls == [{"prompt": "prompt", "temperature": 0.0, "max_tokens": 512}]


def test_core_pipeline_is_offline_resource_bound_and_deterministic(summer_config: SummerConfig):
    resources = [make_resource(1), make_resource(2, description="database systems", topics=["database"])]
    corpus = make_corpus(resources)
    intent = StructuredIntent(topics=["machine-learning"], framework_preferences=["PyTorch"])
    kwargs = {
        "config": summer_config,
        "resources": resources,
        "provenance_records": provenance_for(resources),
        "evidence_corpus": corpus,
        "evaluation_time": NOW,
    }
    first = run_core_pipeline(intent, **kwargs)
    second = run_core_pipeline(intent, **kwargs)
    assert first == second
    assert first.status == "OK"
    assert first.ranked_resource_ids
    for assessment in first.assessments:
        assert all(item.resource_id == assessment.candidate.resource_id for item in assessment.claims)
        assert all(
            item.resource_id == assessment.candidate.resource_id
            for values in assessment.evidence_by_claim.values() for item in values
        )


def test_core_pipeline_orders_claim_evidence_verification_before_decision(monkeypatch, summer_config: SummerConfig):
    import recommender.pipeline as pipeline_module

    resource = make_resource(1)
    events: list[str] = []
    original_candidate = TfidfRetriever.retrieve
    original_claim = pipeline_module.generate_claims
    original_evidence = EvidenceRetriever.retrieve
    original_verify = pipeline_module.verify_claim
    original_confidence = pipeline_module.estimate_confidence
    original_decision = pipeline_module.decide_candidate

    def candidate_wrapper(self, *args, **kwargs):
        events.append("candidate")
        return original_candidate(self, *args, **kwargs)

    def claim_wrapper(*args, **kwargs):
        events.append("claim")
        return original_claim(*args, **kwargs)

    def evidence_wrapper(self, *args, **kwargs):
        events.append("evidence")
        return original_evidence(self, *args, **kwargs)

    def verify_wrapper(*args, **kwargs):
        events.append("verification")
        return original_verify(*args, **kwargs)

    def confidence_wrapper(*args, **kwargs):
        events.append("confidence")
        return original_confidence(*args, **kwargs)

    def decision_wrapper(*args, **kwargs):
        events.append("decision")
        return original_decision(*args, **kwargs)

    monkeypatch.setattr(TfidfRetriever, "retrieve", candidate_wrapper)
    monkeypatch.setattr(pipeline_module, "generate_claims", claim_wrapper)
    monkeypatch.setattr(EvidenceRetriever, "retrieve", evidence_wrapper)
    monkeypatch.setattr(pipeline_module, "verify_claim", verify_wrapper)
    monkeypatch.setattr(pipeline_module, "estimate_confidence", confidence_wrapper)
    monkeypatch.setattr(pipeline_module, "decide_candidate", decision_wrapper)
    result = run_core_pipeline(
        StructuredIntent(topics=["machine-learning"], framework_preferences=["PyTorch"]),
        config=summer_config, resources=[resource], provenance_records=provenance_for([resource]),
        evidence_corpus=make_corpus([resource]), evaluation_time=NOW,
    )
    assert result.status == "OK"
    assert events[0:3] == ["candidate", "claim", "evidence"]
    assert max(index for index, event in enumerate(events) if event == "verification") < events.index("confidence")
    assert events.index("confidence") < events.index("decision")


def test_no_candidate_is_terminal_abstention_not_error(summer_config: SummerConfig):
    resource = make_resource(1, description="database", topics=["database"], language="Go")
    result = run_core_pipeline(
        StructuredIntent(topics=["quantum-physics"]), config=summer_config,
        resources=[resource], provenance_records=provenance_for([resource]),
        evidence_corpus=make_corpus([resource]), evaluation_time=NOW,
    )
    assert result.status == "ABSTAINED"
    assert result.error_code == "NO_CANDIDATE"


def test_candidate_local_failure_does_not_become_global(monkeypatch, summer_config: SummerConfig):
    resources = [
        make_resource(1, description="PyTorch machine learning"),
        make_resource(2, description="PyTorch machine learning"),
    ]
    original = generate_claims

    def fail_one(cand, intent, resource, kg):
        if resource.resource_id == resources[0].resource_id:
            raise RuntimeError("local failure")
        return original(cand, intent, resource, kg)

    monkeypatch.setattr("recommender.pipeline.generate_claims", fail_one)
    result = run_core_pipeline(
        StructuredIntent(topics=["machine-learning"], framework_preferences=["PyTorch"]),
        config=summer_config, resources=resources, provenance_records=provenance_for(resources),
        evidence_corpus=make_corpus(resources), evaluation_time=NOW,
    )
    assert result.status != "ERROR"
    assert len(result.assessments) == 2
    assert any("CLAIM_GENERATION_FAILED" in error for item in result.assessments for error in item.errors)


def test_logging_failure_is_warning_not_abstention(monkeypatch, summer_config: SummerConfig, tmp_path: Path):
    resource = make_resource(1)
    config = summer_config.model_copy(update={
        "logging": summer_config.logging.model_copy(update={"save_sessions": True, "output_path": "sessions.jsonl"}),
    })

    def fail_log(*args, **kwargs):
        raise PermissionError("read only")

    monkeypatch.setattr("recommender.pipeline.write_session_log", fail_log)
    result = run_core_pipeline(
        StructuredIntent(topics=["machine-learning"], framework_preferences=["PyTorch"]),
        config=config, resources=[resource], provenance_records=provenance_for([resource]),
        evidence_corpus=make_corpus([resource]), evaluation_time=NOW, root=tmp_path,
    )
    assert result.status == "OK"
    assert result.warnings and result.warnings[0].startswith("LOGGING_FAILED:")


def test_missing_evidence_is_global_critical(tmp_path: Path, summer_config: SummerConfig):
    config = summer_config.model_copy(update={
        "paths": summer_config.paths.model_copy(update={
            "resources": "missing.jsonl",
        }),
    })
    result = run_core_pipeline(StructuredIntent(topics=["machine"]), config=config, root=tmp_path)
    assert result.status == "ERROR"
    assert result.error_code == GlobalErrorCode.RESOURCE_DATASET_MISSING.value
