from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

import recommender.pipeline as pipeline_module
from recommender.audit import write_user_session_log
from recommender.config import load_config, load_prompts, project_root
from recommender.errors import QueryErrorCode, QueryProcessingError
from recommender.final_benchmark import _grounding_counts, summarize
from recommender.generation.builder import (
    build_generation_view,
    build_grounded_result_pack,
)
from recommender.generation.grounding import (
    hydrate_final_explanation,
    validate_final_output,
)
from recommender.generation.template import build_template_explanation
from recommender.llm import MockLLMClient
from recommender.llm.final_generator import generate_final_explanation
from recommender.schemas import (
    Candidate,
    CandidateAssessment,
    CandidateDecision,
    Claim,
    ConfidenceBreakdown,
    FinalLLMOutput,
    QueryParseAudit,
    RetrievedEvidence,
    StructuredIntent,
    SummerPipelineResult,
    SummerQueryResult,
    VerificationResult,
)


RESOURCE_ID = "github:1000"
SUPPORTED_ID = "claim_aaaaaaaaaaaaaaaa"
UNKNOWN_ID = "claim_bbbbbbbbbbbbbbbb"
CHUNK_ID = "ch_aaaaaaaaaaaaaaaa"


def _confidence() -> ConfidenceBreakdown:
    return ConfidenceBreakdown(
        confidence=0.72,
        bucket="Medium",
        components={"retrieval_score": 0.8},
        weights={"w_retrieval": 0.2},
    )


def _pipeline_result() -> SummerPipelineResult:
    candidate = Candidate(
        resource_id=RESOURCE_ID,
        title="resource",
        url="https://github.com/owner/resource",
        retrieval_score=0.8,
    )
    supported = Claim(
        claim_id=SUPPORTED_ID,
        resource_id=RESOURCE_ID,
        claim_type="has_license",
        subject=RESOURCE_ID,
        predicate="has",
        object="open_source_license",
        text="resource has an explicit open-source license.",
    )
    unknown = Claim(
        claim_id=UNKNOWN_ID,
        resource_id=RESOURCE_ID,
        claim_type="covers_topic",
        subject=RESOURCE_ID,
        predicate="covers",
        object="Transformers",
        text="resource covers Transformers.",
    )
    evidence = RetrievedEvidence(
        chunk_id=CHUNK_ID,
        evidence_id="ev_aaaaaaaaaaaaaaaa",
        resource_id=RESOURCE_ID,
        source_id=f"{RESOURCE_ID}:metadata",
        text="license_spdx_id: \"MIT\"\n" + "metadata " * 500,
        source_url="https://github.com/owner/resource",
        evidence_source_type="repo_metadata",
        source_type="community",
        source_reliability=0.7,
        source_reliability_method="heuristic_v0",
        retrieval_score=0.95,
        section_title="__metadata__",
    )
    verifications = [
        VerificationResult(
            claim_id=SUPPORTED_ID,
            resource_id=RESOURCE_ID,
            label="SUPPORTED",
            score=0.95,
            evidence_chunk_ids=[CHUNK_ID],
            rationale="Metadata contains MIT.",
        ),
        VerificationResult(
            claim_id=UNKNOWN_ID,
            resource_id=RESOURCE_ID,
            label="UNKNOWN",
            score=0.0,
            evidence_chunk_ids=[],
            rationale="Coverage was not established.",
        ),
    ]
    decision = CandidateDecision(
        resource_id=RESOURCE_ID,
        retrieval_score=0.8,
        confidence=0.72,
        confidence_bucket="Medium",
        abstain=False,
        final_score=0.768,
    )
    return SummerPipelineResult(
        status="OK",
        intent=StructuredIntent(topics=["Transformers"]),
        assessments=[CandidateAssessment(
            candidate=candidate,
            claims=[supported, unknown],
            evidence_by_claim={SUPPORTED_ID: [evidence], UNKNOWN_ID: []},
            verifications=verifications,
            confidence=_confidence(),
            decision=decision,
        )],
        ranked_resource_ids=[RESOURCE_ID],
    )


def _valid_output() -> dict:
    return {
        "summary": "A grounded result.",
        "items": {RESOURCE_ID: {
            "reason": "The backend retained this relevant resource.",
        }},
    }


def _settings():
    config = load_config()
    prompts = load_prompts(project_root() / config.llm.prompt_path)
    return config, prompts


def test_grounded_pack_uses_backend_order_labels_and_verification_evidence():
    pack = build_grounded_result_pack(_pipeline_result())
    assert [item.resource_id for item in pack.recommendations] == [RESOURCE_ID]
    recommendation = pack.recommendations[0]
    assert [item.claim_id for item in recommendation.supported_claims] == [SUPPORTED_ID]
    assert [item.claim_id for item in recommendation.unknown_claims] == [UNKNOWN_ID]
    assert recommendation.supported_claims[0].evidence[0].chunk_id == CHUNK_ID
    assert recommendation.unknown_claims[0].evidence == []


def test_generation_view_is_bounded_without_mutating_full_evidence():
    config, _ = _settings()
    bounded = config.generation.model_copy(update={
        "max_input_chars": 2500,
        "evidence_excerpt_chars": 120,
    })
    pack = build_grounded_result_pack(_pipeline_result())
    original = pack.recommendations[0].supported_claims[0].evidence[0].excerpt
    view = build_generation_view(pack, output_language="en", config=bounded)
    serialized = json.dumps(view.model_dump(mode="json"), ensure_ascii=False)
    assert len(serialized) <= 2500
    claim = view.recommendations[0].claims[0]
    assert claim.evidence_chunk_ids == [CHUNK_ID]
    assert claim.evidence_excerpts[0].excerpt_truncated is True
    assert pack.recommendations[0].supported_claims[0].evidence[0].excerpt == original


def test_llm_schema_cannot_generate_fact_text_or_deterministic_metadata():
    invalid = _valid_output()
    invalid["generation_method"] = "invented"
    with pytest.raises(ValidationError):
        FinalLLMOutput.model_validate(invalid)

    fact_invalid = _valid_output()
    fact_invalid["items"][RESOURCE_ID]["verified_facts"] = [{
        "claim_id": SUPPORTED_ID,
        "text": "invented fact",
        "evidence_chunk_ids": [CHUNK_ID],
    }]
    with pytest.raises(ValidationError):
        FinalLLMOutput.model_validate(fact_invalid)


def test_grounding_and_hydration_resolve_fact_text_from_canonical_claim():
    pack = build_grounded_result_pack(_pipeline_result())
    output = FinalLLMOutput.model_validate(_valid_output())
    assert validate_final_output(output, pack).is_valid
    explanation = hydrate_final_explanation(
        output, pack, output_language="en", generation_method="qwen_grounded_v1",
    )
    fact = explanation.items[0].verified_facts[0]
    assert fact.text == "resource has an explicit open-source license."
    assert fact.evidence_chunk_ids == [CHUNK_ID]
    assert explanation.generation_method == "qwen_grounded_v1"
    assert "not a calibrated probability" in explanation.confidence_disclaimer


def test_grounding_rejects_resource_attack_while_rank_is_backend_controlled():
    data = _valid_output()
    data["items"]["github:9999"] = data["items"].pop(RESOURCE_ID)
    output = FinalLLMOutput.model_validate(data)
    result = validate_final_output(output, build_grounded_result_pack(_pipeline_result()))
    assert not result.is_valid
    assert result.fallback_required


def test_template_fallback_is_deterministic_and_backend_grounded():
    pack = build_grounded_result_pack(_pipeline_result())
    first = build_template_explanation(pack, output_language="zh")
    second = build_template_explanation(pack, output_language="zh")
    assert first == second
    assert first.generation_method == "template_grounded_v1"
    assert first.items[0].verified_facts[0].text == (
        "resource has an explicit open-source license."
    )


def test_final_generator_success_and_retry_are_grounded():
    config, prompts = _settings()
    client = MockLLMClient([
        "not json",
        json.dumps(_valid_output()),
    ])
    outcome = generate_final_explanation(
        build_grounded_result_pack(_pipeline_result()),
        client,
        output_language="en",
        config=config.generation,
        prompt=prompts.final_generation,
    )
    assert outcome.audit.generation_status == "SUCCESS"
    assert outcome.audit.retry_count == 1
    assert outcome.audit.fallback_used is False
    assert len(client.calls) == 2


def test_transport_timeout_preserves_real_reason_and_uses_fallback():
    config, prompts = _settings()
    client = MockLLMClient([
        QueryProcessingError(QueryErrorCode.LOCAL_LLM_TIMEOUT, "timeout"),
    ])
    outcome = generate_final_explanation(
        build_grounded_result_pack(_pipeline_result()),
        client,
        output_language="en",
        config=config.generation,
        prompt=prompts.final_generation,
    )
    assert outcome.audit.generation_status == "FALLBACK"
    assert outcome.audit.error_code == "FINAL_LLM_TIMEOUT"
    assert outcome.audit.fallback_used is True
    assert all("FINAL_TEMPLATE_FALLBACK" not in item for item in outcome.warnings)
    assert len(client.calls) == 1


def test_generation_disabled_is_non_error_template_fallback():
    config, prompts = _settings()
    client = MockLLMClient([])
    outcome = generate_final_explanation(
        build_grounded_result_pack(_pipeline_result()),
        client,
        output_language="en",
        config=config.generation.model_copy(update={"enabled": False}),
        prompt=prompts.final_generation,
    )
    assert outcome.audit.generation_status == "FALLBACK"
    assert outcome.audit.error_code is None
    assert client.calls == []


def make_query_result(status: str = "OK") -> SummerQueryResult:
    audit = QueryParseAudit(
        backend="mock",
        model="mock",
        prompt_version="summer_query_v2",
        parse_status="SUCCESS",
        retry_count=0,
        latency_ms=1,
        query_hash="sha256:" + "a" * 64,
        query_length=10,
    )
    parsed = {
        "intent": StructuredIntent(topics=["Transformers"]),
        "audit": audit,
    }
    recommendation = _pipeline_result()
    recommendation.status = "OK" if status == "OK" else "ABSTAINED"
    if status != "OK":
        recommendation.ranked_resource_ids = []
        recommendation.error_code = "NO_CANDIDATE"
        recommendation.error_message = "none"
    return SummerQueryResult(
        status=status,
        parsed_query=parsed,
        query_audit=audit,
        recommendation=recommendation,
    )


def test_user_pipeline_success_is_copy_on_request_and_preserves_backend(monkeypatch):
    config, prompts = _settings()
    captured = {}

    def fake_recommend(*args, **kwargs):
        captured["config"] = kwargs["config"]
        return make_query_result("OK")

    monkeypatch.setattr(pipeline_module, "recommend_from_query", fake_recommend)
    final_client = MockLLMClient([json.dumps(_valid_output())])
    result = pipeline_module.run_user_pipeline(
        "query",
        config=config,
        prompts=prompts,
        query_client=MockLLMClient([]),
        final_client=final_client,
        top_k=7,
        output_language="en",
    )
    assert result.status == "OK"
    assert result.generation_audit.generation_status == "SUCCESS"
    assert result.recommendations[0].candidate.resource_id == RESOURCE_ID
    assert result.recommendations[0].explanation.verified_facts[0].text == (
        "resource has an explicit open-source license."
    )
    assert config.retrieval.top_k == 5
    assert captured["config"] is not config
    assert captured["config"].retrieval.top_k == 7
    assert captured["config"].logging.save_sessions is False


def test_user_pipeline_abstention_skips_final_llm(monkeypatch):
    config, prompts = _settings()
    monkeypatch.setattr(
        pipeline_module, "recommend_from_query",
        lambda *args, **kwargs: make_query_result("RECOMMENDATION_ABSTAINED"),
    )
    final_client = MockLLMClient([])
    result = pipeline_module.run_user_pipeline(
        "query", config=config, prompts=prompts, final_client=final_client,
    )
    assert result.status == "RECOMMENDATION_ABSTAINED"
    assert result.error_code == "NO_CANDIDATE"
    assert result.generation_audit.generation_status == "NOT_RUN"
    assert result.recommendations == []
    assert final_client.calls == []


def test_user_pipeline_query_error_skips_s1_result_and_final_llm(monkeypatch):
    config, prompts = _settings()
    audit = QueryParseAudit(
        backend="mock", model="mock", prompt_version="summer_query_v2",
        parse_status="ERROR", retry_count=0, latency_ms=1,
        query_hash="sha256:" + "b" * 64, query_length=0,
    )
    query_error = SummerQueryResult(
        status="QUERY_ERROR",
        query_audit=audit,
        error_code="QUERY_EMPTY",
        error_message="empty",
    )
    monkeypatch.setattr(
        pipeline_module, "recommend_from_query", lambda *args, **kwargs: query_error,
    )
    final_client = MockLLMClient([])
    result = pipeline_module.run_user_pipeline(
        "", config=config, prompts=prompts, final_client=final_client,
    )
    assert result.status == "QUERY_ERROR"
    assert result.generation_audit.generation_status == "NOT_RUN"
    assert result.pipeline_trace.recommendation_status == "NOT_RUN"
    assert final_client.calls == []


def test_ungrounded_final_output_falls_back_without_changing_ranking(monkeypatch):
    config, prompts = _settings()
    monkeypatch.setattr(
        pipeline_module, "recommend_from_query",
        lambda *args, **kwargs: make_query_result("OK"),
    )
    attack = _valid_output()
    attack["items"]["github:9999"] = attack["items"].pop(RESOURCE_ID)
    final_client = MockLLMClient([json.dumps(attack), json.dumps(attack)])
    result = pipeline_module.run_user_pipeline(
        "query", config=config, prompts=prompts, final_client=final_client,
    )
    assert result.status == "OK"
    assert result.generation_audit.generation_status == "FALLBACK"
    assert result.generation_audit.error_code == "FINAL_GROUNDING_FAILED"
    assert [item.candidate.resource_id for item in result.recommendations] == [RESOURCE_ID]
    assert result.final_explanation.generation_method == "template_grounded_v1"


def test_top_k_outside_ui_range_is_rejected_without_mutating_config(monkeypatch):
    config, prompts = _settings()
    called = False

    def fake_recommend(*args, **kwargs):
        nonlocal called
        called = True
        return make_query_result("OK")

    monkeypatch.setattr(pipeline_module, "recommend_from_query", fake_recommend)
    with pytest.raises(ValueError, match="top_k"):
        pipeline_module.run_user_pipeline(
            "query", config=config, prompts=prompts, top_k=11,
        )
    assert not called
    assert config.retrieval.top_k == 5


def test_benchmark_grounding_audit_uses_backend_owned_facts(monkeypatch):
    config, prompts = _settings()
    monkeypatch.setattr(
        pipeline_module, "recommend_from_query",
        lambda *args, **kwargs: make_query_result("OK"),
    )
    result = pipeline_module.run_user_pipeline(
        "query", config=config, prompts=prompts,
        final_client=MockLLMClient([json.dumps(_valid_output())]),
    )
    assert _grounding_counts(result) == {
        "ranking_preserved": True,
        "hallucinated_resources": 0,
        "hallucinated_claims": 0,
        "hallucinated_evidence_ids": 0,
    }


def test_benchmark_summary_separates_abstention_from_generation():
    rows = [{
        "status": "OK",
        "generation_attempted": True,
        "generation_status": "SUCCESS",
        "llm_grounding_passed": True,
        "hallucinated_resources": 0,
        "hallucinated_claims": 0,
        "hallucinated_evidence_ids": 0,
        "ranking_preserved": True,
        "query_latency_ms": 10.0,
        "core_latency_ms": 2.0,
        "generation_latency_ms": 30.0,
        "total_latency_ms": 42.0,
        "generation_input_characters": 1200,
        "prompt_eval_tokens": 300,
        "output_eval_tokens": 80,
    }, {
        "status": "RECOMMENDATION_ABSTAINED",
        "generation_attempted": False,
        "generation_status": "NOT_RUN",
        "llm_grounding_passed": False,
        "hallucinated_resources": 0,
        "hallucinated_claims": 0,
        "hallucinated_evidence_ids": 0,
        "ranking_preserved": True,
        "query_latency_ms": 8.0,
        "core_latency_ms": 1.0,
        "generation_latency_ms": 0.0,
        "total_latency_ms": 9.0,
        "generation_input_characters": 0,
        "prompt_eval_tokens": None,
        "output_eval_tokens": None,
    }]
    summary = summarize(rows, model="qwen3:8b", prompt_version="summer_final_v1")
    assert summary["generation_attempted_count"] == 1
    assert summary["final_json_valid_rate"] == 1.0
    assert summary["grounding_pass_rate"] == 1.0
    assert summary["query_statuses"] == {"OK": 1, "RECOMMENDATION_ABSTAINED": 1}


def test_s3_audit_log_omits_raw_query_and_generated_prose(monkeypatch, tmp_path):
    config, prompts = _settings()
    monkeypatch.setattr(
        pipeline_module, "recommend_from_query",
        lambda *args, **kwargs: make_query_result("OK"),
    )
    result = pipeline_module.run_user_pipeline(
        "private raw query",
        config=config.model_copy(update={
            "logging": config.logging.model_copy(update={"save_sessions": False}),
        }),
        prompts=prompts,
        final_client=MockLLMClient([json.dumps(_valid_output())]),
    )
    target = tmp_path / "session.jsonl"
    write_user_session_log(result, target)
    logged = target.read_text(encoding="utf-8")
    assert "private raw query" not in logged
    assert "The backend retained this relevant resource." not in logged
    assert "A grounded result." not in logged
    assert SUPPORTED_ID not in logged
    assert CHUNK_ID not in logged


def test_s3_logging_failure_is_warning_not_recommendation_failure(monkeypatch):
    config, prompts = _settings()
    monkeypatch.setattr(
        pipeline_module, "recommend_from_query",
        lambda *args, **kwargs: make_query_result("OK"),
    )
    monkeypatch.setattr(
        pipeline_module, "write_user_session_log",
        lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("denied")),
    )
    result = pipeline_module.run_user_pipeline(
        "query", config=config, prompts=prompts,
        final_client=MockLLMClient([json.dumps(_valid_output())]),
    )
    assert result.status == "OK"
    assert result.generation_audit.generation_status == "SUCCESS"
    assert any(warning.startswith("LOGGING_FAILED:") for warning in result.warnings)
