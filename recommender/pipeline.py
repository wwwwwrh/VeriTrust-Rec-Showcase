"""Summer S1 core: intent → candidates → claims → Evidence → decision."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any

from data_pipeline.resource_dataset.schema import Resource
from pydantic import ValidationError

from recommender.abstention.policy import decide_candidate
from recommender.audit import (
    write_query_failure_log,
    write_session_log,
    write_user_session_log,
)
from recommender.config import (
    PromptsConfig,
    SummerConfig,
    load_config,
    load_prompts,
    project_root,
    resolve_project_path,
)
from recommender.data import (
    EvidenceCorpus,
    load_evidence_corpus,
    load_resource_provenance,
    load_resources,
)
from recommender.errors import (
    CandidateErrorCode,
    GlobalErrorCode,
    PipelineError,
    QueryErrorCode,
    QueryProcessingError,
)
from recommender.llm import LocalLLMClient, OllamaLocalLLMClient, parse_query
from recommender.llm.final_generator import generate_final_explanation
from recommender.generation.builder import build_grounded_result_pack
from recommender.kg.graph import KnowledgeGraph
from recommender.retrieval.evidence import EvidenceRetriever
from recommender.retrieval.retriever import TfidfRetriever
from recommender.schemas import (
    Candidate,
    CandidateAssessment,
    CandidateDecision,
    ConfidenceBreakdown,
    QueryParseAudit,
    OutputLanguage,
    StructuredIntent,
    SummerPipelineResult,
    SummerQueryResult,
    GenerationAudit,
    PipelineTrace,
    UserFacingClaim,
    UserFacingEvidence,
    UserFacingRecommendation,
    UserFacingResult,
    VerificationResult,
)
from recommender.uncertainty.scorer import estimate_confidence
from recommender.verification.claim_extractor import generate_claims
from recommender.verification.verifier import VerificationSummary, summarize_verification, verify_claim


_DEFAULT_LLM_CLIENTS: dict[tuple[str, str, float], OllamaLocalLLMClient] = {}
_DEFAULT_LLM_CLIENTS_LOCK = Lock()
_DEFAULT_FINAL_LLM_CLIENTS: dict[tuple[str, str, float], OllamaLocalLLMClient] = {}
_DEFAULT_FINAL_LLM_CLIENTS_LOCK = Lock()


def _default_llm_client(config: SummerConfig) -> OllamaLocalLLMClient:
    """Reuse the localhost client so model validation is not repeated per query."""

    key = (config.llm.base_url, config.llm.model, config.llm.timeout_seconds)
    with _DEFAULT_LLM_CLIENTS_LOCK:
        client = _DEFAULT_LLM_CLIENTS.get(key)
        if client is None:
            client = OllamaLocalLLMClient(config.llm)
            _DEFAULT_LLM_CLIENTS[key] = client
        return client


def _default_final_llm_client(config: SummerConfig) -> OllamaLocalLLMClient:
    """Cache a transport-only client; it stores no per-query generation state."""

    key = (config.llm.base_url, config.llm.model, config.generation.timeout_seconds)
    with _DEFAULT_FINAL_LLM_CLIENTS_LOCK:
        client = _DEFAULT_FINAL_LLM_CLIENTS.get(key)
        if client is None:
            final_llm_config = config.llm.model_copy(update={
                "timeout_seconds": config.generation.timeout_seconds,
            })
            client = OllamaLocalLLMClient(final_llm_config)
            _DEFAULT_FINAL_LLM_CLIENTS[key] = client
        return client


def _error_result(code: GlobalErrorCode, message: str, intent: StructuredIntent | None) -> SummerPipelineResult:
    return SummerPipelineResult(
        status="ERROR",
        intent=intent,
        error_code=code.value,
        error_message=message,
    )


def _zero_confidence(config: SummerConfig) -> ConfidenceBreakdown:
    return estimate_confidence(
        retrieval_score=0,
        evidence_coverage=0,
        verification_score=0,
        kg_support=0.5,
        source_reliability=0,
        config=config.confidence,
    )


def _empty_summary() -> VerificationSummary:
    return VerificationSummary(0, 0, 0, 0, 0.0, 0.0, 0.5, 0.0, 0)


def _local_failure_assessment(
    candidate: Candidate,
    config: SummerConfig,
    code: CandidateErrorCode,
    message: str,
    *,
    claims: list | None = None,
    evidence_by_claim: dict | None = None,
    verifications: list[VerificationResult] | None = None,
) -> CandidateAssessment:
    confidence = _zero_confidence(config)
    errors = [f"{code.value}: {message}"]
    decision = decide_candidate(
        candidate,
        confidence,
        evidence_coverage=0,
        verifications=verifications or [],
        local_errors=errors,
        abstention=config.abstention,
        ranking=config.ranking,
    )
    return CandidateAssessment(
        candidate=candidate,
        claims=claims or [],
        evidence_by_claim=evidence_by_claim or {},
        verifications=verifications or [],
        confidence=confidence,
        decision=decision,
        errors=errors,
    )


def _process_candidate(
    candidate: Candidate,
    *,
    intent: StructuredIntent,
    resource: Resource,
    kg: KnowledgeGraph,
    evidence_retriever: EvidenceRetriever,
    config: SummerConfig,
    evaluation_time: datetime,
) -> CandidateAssessment:
    try:
        claims = generate_claims(candidate, intent, resource, kg)
    except Exception as exc:
        return _local_failure_assessment(
            candidate, config, CandidateErrorCode.CLAIM_GENERATION_FAILED, str(exc),
        )

    evidence_by_claim: dict = {}
    verifications: list[VerificationResult] = []
    for claim in claims:
        try:
            evidence = evidence_retriever.retrieve(claim, top_k=config.evidence.top_k_per_claim)
            evidence_by_claim[claim.claim_id] = evidence
        except Exception as exc:
            return _local_failure_assessment(
                candidate, config, CandidateErrorCode.EVIDENCE_RETRIEVAL_FAILED, str(exc),
                claims=claims, evidence_by_claim=evidence_by_claim, verifications=verifications,
            )
        try:
            verifications.append(verify_claim(
                claim, evidence, config=config.verification, evaluation_time=evaluation_time,
            ))
        except PipelineError as exc:
            code = exc.code if isinstance(exc.code, CandidateErrorCode) else CandidateErrorCode.VERIFICATION_FAILED
            return _local_failure_assessment(
                candidate, config, code, str(exc), claims=claims,
                evidence_by_claim=evidence_by_claim, verifications=verifications,
            )
        except Exception as exc:
            return _local_failure_assessment(
                candidate, config, CandidateErrorCode.VERIFICATION_FAILED, str(exc), claims=claims,
                evidence_by_claim=evidence_by_claim, verifications=verifications,
            )

    try:
        summary = summarize_verification(claims, evidence_by_claim, verifications, kg)
    except Exception as exc:
        return _local_failure_assessment(
            candidate, config, CandidateErrorCode.VERIFICATION_FAILED, str(exc), claims=claims,
            evidence_by_claim=evidence_by_claim, verifications=verifications,
        )
    confidence = estimate_confidence(
        retrieval_score=candidate.retrieval_score,
        evidence_coverage=summary.evidence_coverage,
        verification_score=summary.verification_score,
        kg_support=summary.kg_support,
        source_reliability=summary.source_reliability,
        config=config.confidence,
    )
    decision = decide_candidate(
        candidate,
        confidence,
        evidence_coverage=summary.evidence_coverage,
        verifications=verifications,
        local_errors=[],
        abstention=config.abstention,
        ranking=config.ranking,
    )
    return CandidateAssessment(
        candidate=candidate,
        claims=claims,
        evidence_by_claim=evidence_by_claim,
        verifications=verifications,
        confidence=confidence,
        decision=decision,
    )


def _log_noncritical(
    result: SummerPipelineResult,
    config: SummerConfig,
    root: Path,
    *,
    query_audit: QueryParseAudit | None = None,
) -> SummerPipelineResult:
    if not config.logging.save_sessions:
        return result
    try:
        write_session_log(
            result,
            resolve_project_path(config.logging.output_path, root=root),
            query_audit=query_audit,
        )
    except Exception as exc:
        result.warnings.append(f"LOGGING_FAILED: {type(exc).__name__}: {exc}")
    return result


def run_core_pipeline(
    intent: StructuredIntent | dict[str, Any],
    *,
    config: SummerConfig | None = None,
    resources: list[Resource] | None = None,
    provenance_records: list[dict[str, Any]] | None = None,
    evidence_corpus: EvidenceCorpus | None = None,
    evaluation_time: datetime | None = None,
    root: Path | None = None,
    query_audit: QueryParseAudit | None = None,
) -> SummerPipelineResult:
    root = root or project_root()
    parsed_intent: StructuredIntent | None = None
    try:
        config = config or load_config()
        parsed_intent = StructuredIntent.model_validate(intent)
    except PipelineError as exc:
        return _error_result(GlobalErrorCode.CONFIG_INVALID, str(exc), parsed_intent)
    except ValidationError as exc:
        return _error_result(GlobalErrorCode.INTENT_INVALID, str(exc), None)

    try:
        resources = resources or load_resources(resolve_project_path(config.paths.resources, root=root))
        provenance_records = (
            provenance_records if provenance_records is not None
            else load_resource_provenance(resolve_project_path(config.paths.resource_provenance, root=root))
        )
        evidence_corpus = evidence_corpus or load_evidence_corpus(
            documents_path=resolve_project_path(config.paths.evidence_documents, root=root),
            chunks_path=resolve_project_path(config.paths.evidence_chunks, root=root),
            evidence_manifest_path=resolve_project_path(config.paths.evidence_manifest, root=root),
            resource_manifest_path=resolve_project_path(config.paths.resource_manifest, root=root),
            resources=resources,
            compatibility=config.evidence_compatibility,
        )
        evidence_retriever = EvidenceRetriever(
            evidence_corpus.chunks,
            metadata_priority_bonus=config.evidence.metadata_priority_bonus,
        )
    except PipelineError as exc:
        return _log_noncritical(
            _error_result(exc.code, str(exc), parsed_intent), config, root,
            query_audit=query_audit,
        )  # type: ignore[arg-type]
    except Exception as exc:
        return _log_noncritical(_error_result(
            GlobalErrorCode.EVIDENCE_CORPUS_INVALID, str(exc), parsed_intent,
        ), config, root, query_audit=query_audit)

    try:
        kg = KnowledgeGraph.build(resources, provenance_records)
    except Exception as exc:
        return _log_noncritical(_error_result(
            GlobalErrorCode.KG_BUILD_FAILED, str(exc), parsed_intent,
        ), config, root, query_audit=query_audit)
    try:
        candidates = TfidfRetriever(resources).retrieve(parsed_intent, top_k=config.retrieval.top_k)
    except Exception as exc:
        return _log_noncritical(_error_result(
            GlobalErrorCode.CANDIDATE_RETRIEVAL_FAILED, str(exc), parsed_intent,
        ), config, root, query_audit=query_audit)
    if not candidates:
        return _log_noncritical(SummerPipelineResult(
            status="ABSTAINED",
            intent=parsed_intent,
            error_code="NO_CANDIDATE",
            error_message="no candidate has a positive TF-IDF score",
        ), config, root, query_audit=query_audit)

    resource_by_id = {item.resource_id: item for item in resources}
    at = evaluation_time or datetime.now(timezone.utc)
    assessments = [
        _process_candidate(
            candidate,
            intent=parsed_intent,
            resource=resource_by_id[candidate.resource_id],
            kg=kg,
            evidence_retriever=evidence_retriever,
            config=config,
            evaluation_time=at,
        )
        for candidate in candidates
    ]
    accepted = [item.decision for item in assessments if not item.decision.abstain]
    accepted.sort(key=lambda item: (-item.final_score, item.resource_id))
    result = SummerPipelineResult(
        status="OK" if accepted else "ABSTAINED",
        intent=parsed_intent,
        assessments=assessments,
        ranked_resource_ids=[item.resource_id for item in accepted],
        error_code=None if accepted else "ALL_CANDIDATES_ABSTAINED",
        error_message=None if accepted else "all retrieved candidates were conservatively abstained",
    )
    return _log_noncritical(result, config, root, query_audit=query_audit)


def _query_audit(
    raw_query: str,
    *,
    backend: str,
    model: str,
    prompt_version: str,
    retry_count: int,
    latency_ms: float,
) -> QueryParseAudit:
    return QueryParseAudit(
        backend=backend,
        model=model,
        prompt_version=prompt_version,
        parse_status="ERROR",
        retry_count=retry_count,
        latency_ms=round(max(0.0, latency_ms), 3),
        query_hash="sha256:" + hashlib.sha256(raw_query.strip().encode("utf-8")).hexdigest(),
        query_length=len(raw_query.strip()),
    )


def _log_query_failure_noncritical(
    result: SummerQueryResult,
    *,
    config: SummerConfig | None,
    root: Path,
) -> SummerQueryResult:
    if config is None or not config.logging.save_sessions:
        return result
    try:
        write_query_failure_log(
            result,
            resolve_project_path(config.logging.output_path, root=root),
        )
    except Exception as exc:
        result.warnings.append(f"LOGGING_FAILED: {type(exc).__name__}: {exc}")
    return result


def recommend_from_query(
    raw_query: str,
    *,
    client: LocalLLMClient | None = None,
    config: SummerConfig | None = None,
    prompts: PromptsConfig | None = None,
    resources: list[Resource] | None = None,
    provenance_records: list[dict[str, Any]] | None = None,
    evidence_corpus: EvidenceCorpus | None = None,
    evaluation_time: datetime | None = None,
    root: Path | None = None,
) -> SummerQueryResult:
    """Run S2 query processing, then S1 only when parsing succeeds."""

    root = root or project_root()
    loaded_config = config
    try:
        loaded_config = loaded_config or load_config()
        prompts = prompts or load_prompts(
            resolve_project_path(loaded_config.llm.prompt_path, root=root)
        )
    except Exception as exc:
        audit = _query_audit(
            raw_query,
            backend="ollama",
            model=getattr(getattr(loaded_config, "llm", None), "model", "unconfigured"),
            prompt_version="unconfigured",
            retry_count=0,
            latency_ms=0,
        )
        return SummerQueryResult(
            status="QUERY_ERROR",
            query_audit=audit,
            error_code=QueryErrorCode.LOCAL_LLM_NOT_CONFIGURED.value,
            error_message=str(exc),
        )

    assert loaded_config is not None and prompts is not None
    active_client = client or _default_llm_client(loaded_config)
    try:
        parsed = parse_query(
            raw_query,
            active_client,
            llm_config=loaded_config.llm,
            prompt_config=prompts.query_understanding,
        )
    except QueryProcessingError as exc:
        audit = _query_audit(
            raw_query,
            backend=active_client.backend,
            model=active_client.model,
            prompt_version=prompts.query_understanding.prompt_version,
            retry_count=exc.retry_count,
            latency_ms=exc.latency_ms,
        )
        result = SummerQueryResult(
            status="QUERY_ERROR",
            query_audit=audit,
            error_code=exc.code.value,
            error_message=str(exc),
        )
        return _log_query_failure_noncritical(result, config=loaded_config, root=root)
    recommendation = run_core_pipeline(
        parsed.intent,
        config=loaded_config,
        resources=resources,
        provenance_records=provenance_records,
        evidence_corpus=evidence_corpus,
        evaluation_time=evaluation_time,
        root=root,
        query_audit=parsed.audit,
    )
    if recommendation.status == "OK":
        status = "OK"
    elif recommendation.status == "ABSTAINED":
        status = "RECOMMENDATION_ABSTAINED"
    else:
        status = "RECOMMENDATION_ERROR"
    return SummerQueryResult(
        status=status,
        parsed_query=parsed,
        query_audit=parsed.audit,
        recommendation=recommendation,
        error_code=recommendation.error_code if recommendation.status == "ERROR" else None,
        error_message=recommendation.error_message if recommendation.status == "ERROR" else None,
    )


def _not_run_generation_audit(
    *,
    config: SummerConfig | None,
    prompts: PromptsConfig | None,
) -> GenerationAudit:
    return GenerationAudit(
        backend=config.llm.backend if config else "unconfigured",
        model=config.llm.model if config else "unconfigured",
        prompt_version=(
            prompts.final_generation.prompt_version if prompts else "unconfigured"
        ),
        generation_status="NOT_RUN",
        retry_count=0,
        latency_ms=0,
        grounding_valid=False,
        fallback_used=False,
        resource_count=0,
        input_character_count=0,
    )


def _pipeline_trace(
    result: SummerQueryResult,
    *,
    recommendation_elapsed_ms: float,
    generation_audit: GenerationAudit,
    total_elapsed_ms: float,
) -> PipelineTrace:
    recommendation = result.recommendation
    assessments = recommendation.assessments if recommendation else []
    verifications = [
        item for assessment in assessments for item in assessment.verifications
    ]
    query_latency = result.query_audit.latency_ms
    return PipelineTrace(
        query_status=result.query_audit.parse_status,
        query_latency_ms=query_latency,
        recommendation_status=recommendation.status if recommendation else "NOT_RUN",
        core_latency_ms=round(max(0.0, recommendation_elapsed_ms - query_latency), 3),
        candidate_count=len(assessments),
        claim_count=sum(len(item.claims) for item in assessments),
        retrieved_evidence_count=sum(
            len(evidence)
            for assessment in assessments
            for evidence in assessment.evidence_by_claim.values()
        ),
        supported_count=sum(item.label == "SUPPORTED" for item in verifications),
        unknown_count=sum(item.label == "UNKNOWN" for item in verifications),
        contradicted_count=sum(item.label == "CONTRADICTED" for item in verifications),
        accepted_count=sum(not item.decision.abstain for item in assessments),
        abstained_count=sum(item.decision.abstain for item in assessments),
        generation_status=generation_audit.generation_status,
        generation_latency_ms=generation_audit.latency_ms,
        total_latency_ms=round(max(0.0, total_elapsed_ms), 3),
    )


def _user_recommendations(
    result: SummerPipelineResult,
    *,
    explanation: object | None,
) -> list[UserFacingRecommendation]:
    assessment_by_id = {item.candidate.resource_id: item for item in result.assessments}
    explanation_by_id = {
        item.resource_id: item
        for item in getattr(explanation, "items", [])
    }
    output: list[UserFacingRecommendation] = []
    for rank, resource_id in enumerate(result.ranked_resource_ids, start=1):
        assessment = assessment_by_id[resource_id]
        verification_by_id = {item.claim_id: item for item in assessment.verifications}
        claims: list[UserFacingClaim] = []
        for claim in assessment.claims:
            verification = verification_by_id.get(claim.claim_id)
            if verification is None:
                continue
            used = set(verification.evidence_chunk_ids)
            claims.append(UserFacingClaim(
                claim=claim,
                verification=verification,
                evidence=[UserFacingEvidence(
                    evidence=item,
                    used_for_verification=item.chunk_id in used,
                ) for item in assessment.evidence_by_claim.get(claim.claim_id, [])],
            ))
        output.append(UserFacingRecommendation(
            rank=rank,
            candidate=assessment.candidate,
            confidence=assessment.confidence,
            decision=assessment.decision,
            claims=claims,
            explanation=explanation_by_id.get(resource_id),
        ))
    return output


def _log_user_noncritical(
    result: UserFacingResult,
    *,
    config: SummerConfig | None,
    root: Path,
) -> UserFacingResult:
    if config is None or not config.logging.save_sessions:
        return result
    try:
        write_user_session_log(
            result,
            resolve_project_path(config.logging.output_path, root=root),
        )
    except Exception as exc:
        result.warnings.append(f"LOGGING_FAILED: {type(exc).__name__}: {exc}")
    return result


def run_user_pipeline(
    raw_query: str,
    *,
    output_language: OutputLanguage = "zh",
    top_k: int | None = None,
    query_client: LocalLLMClient | None = None,
    final_client: LocalLLMClient | None = None,
    config: SummerConfig | None = None,
    prompts: PromptsConfig | None = None,
    resources: list[Resource] | None = None,
    provenance_records: list[dict[str, Any]] | None = None,
    evidence_corpus: EvidenceCorpus | None = None,
    evaluation_time: datetime | None = None,
    root: Path | None = None,
) -> UserFacingResult:
    """Run S2 → S1 → grounded S3 explanation without changing S1 decisions."""

    started = perf_counter()
    root = root or project_root()
    loaded_config = config
    loaded_prompts = prompts
    try:
        loaded_config = loaded_config or load_config()
        loaded_prompts = loaded_prompts or load_prompts(
            resolve_project_path(loaded_config.llm.prompt_path, root=root)
        )
    except Exception:
        request_config = loaded_config
    else:
        requested_top_k = top_k if top_k is not None else loaded_config.retrieval.top_k
        if not loaded_config.ui.min_top_k <= requested_top_k <= loaded_config.ui.max_top_k:
            raise ValueError(
                f"top_k must be between {loaded_config.ui.min_top_k} "
                f"and {loaded_config.ui.max_top_k}"
            )
        # Copy-on-request: never mutate a config object cached by Streamlit.
        request_config = loaded_config.model_copy(update={
            "retrieval": loaded_config.retrieval.model_copy(update={
                "top_k": requested_top_k,
            }),
            "logging": loaded_config.logging.model_copy(update={
                "save_sessions": False,
            }),
        })

    recommendation_started = perf_counter()
    query_result = recommend_from_query(
        raw_query,
        client=query_client,
        config=request_config,
        prompts=loaded_prompts,
        resources=resources,
        provenance_records=provenance_records,
        evidence_corpus=evidence_corpus,
        evaluation_time=evaluation_time,
        root=root,
    )
    recommendation_elapsed = (perf_counter() - recommendation_started) * 1000

    if query_result.status != "OK":
        audit = _not_run_generation_audit(
            config=loaded_config, prompts=loaded_prompts,
        )
        nested = query_result.recommendation
        error_code = query_result.error_code or (nested.error_code if nested else None)
        error_message = query_result.error_message or (
            nested.error_message if nested else None
        )
        result = UserFacingResult(
            status=query_result.status,
            intent=query_result.parsed_query.intent if query_result.parsed_query else None,
            query_audit=query_result.query_audit,
            generation_audit=audit,
            pipeline_trace=_pipeline_trace(
                query_result,
                recommendation_elapsed_ms=recommendation_elapsed,
                generation_audit=audit,
                total_elapsed_ms=(perf_counter() - started) * 1000,
            ),
            error_code=error_code,
            error_message=error_message,
            warnings=(query_result.warnings + (nested.warnings if nested else [])),
        )
        return _log_user_noncritical(result, config=loaded_config, root=root)

    assert query_result.recommendation is not None
    active_final_client = final_client or _default_final_llm_client(loaded_config)  # type: ignore[arg-type]
    try:
        pack = build_grounded_result_pack(query_result.recommendation)
        outcome = generate_final_explanation(
            pack,
            active_final_client,
            output_language=output_language,
            config=loaded_config.generation,  # type: ignore[union-attr]
            prompt=loaded_prompts.final_generation,  # type: ignore[union-attr]
        )
    except Exception as exc:
        outcome = None
        audit = GenerationAudit(
            backend=active_final_client.backend,
            model=active_final_client.model,
            prompt_version=(
                loaded_prompts.final_generation.prompt_version
                if loaded_prompts else "unconfigured"
            ),
            generation_status="FAILED",
            retry_count=0,
            latency_ms=0,
            grounding_valid=False,
            fallback_used=False,
            resource_count=len(query_result.recommendation.ranked_resource_ids),
            input_character_count=0,
            error_code="FINAL_GROUNDING_FAILED",
        )
        generation_warnings = [
            f"FINAL_GROUNDING_FAILED: {type(exc).__name__}: {exc}"
        ]
        explanation = None
    else:
        audit = outcome.audit
        generation_warnings = outcome.warnings
        explanation = outcome.explanation

    user_recommendations = _user_recommendations(
        query_result.recommendation, explanation=explanation,
    )
    result = UserFacingResult(
        status="OK",
        intent=query_result.parsed_query.intent if query_result.parsed_query else None,
        query_audit=query_result.query_audit,
        recommendations=user_recommendations,
        final_explanation=explanation,
        generation_audit=audit,
        pipeline_trace=_pipeline_trace(
            query_result,
            recommendation_elapsed_ms=recommendation_elapsed,
            generation_audit=audit,
            total_elapsed_ms=(perf_counter() - started) * 1000,
        ),
        warnings=(
            query_result.warnings
            + query_result.recommendation.warnings
            + generation_warnings
        ),
    )
    return _log_user_noncritical(result, config=loaded_config, root=root)


__all__ = ["recommend_from_query", "run_core_pipeline", "run_user_pipeline"]
