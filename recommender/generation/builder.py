"""Build full and length-bounded views of trustworthy S1 results."""

from __future__ import annotations

import json

from recommender.config import GenerationConfig
from recommender.schemas import (
    CandidateAssessment,
    GenerationClaimView,
    GenerationEvidenceExcerpt,
    GenerationRecommendationView,
    GroundedClaimReference,
    GroundedEvidenceReference,
    GroundedGenerationView,
    GroundedRecommendation,
    GroundedResultPack,
    OutputLanguage,
    SummerPipelineResult,
)


def _truncate(value: str, limit: int) -> tuple[str, bool]:
    if len(value) <= limit:
        return value, False
    candidate = value[:limit].rstrip()
    boundary = max(candidate.rfind("\n"), candidate.rfind(" "))
    if boundary >= max(1, limit // 2):
        candidate = candidate[:boundary].rstrip()
    return candidate + "…", True


def _ground_claims(assessment: CandidateAssessment) -> list[GroundedClaimReference]:
    claim_by_id = {item.claim_id: item for item in assessment.claims}
    if len(claim_by_id) != len(assessment.claims):
        raise ValueError("CandidateAssessment contains duplicate Claim IDs")
    verification_by_id = {item.claim_id: item for item in assessment.verifications}
    if set(verification_by_id) != set(claim_by_id):
        raise ValueError("every Claim must have exactly one VerificationResult")

    output: list[GroundedClaimReference] = []
    for claim in assessment.claims:
        verification = verification_by_id[claim.claim_id]
        if verification.resource_id != assessment.candidate.resource_id:
            raise ValueError("VerificationResult has the wrong resource_id")
        retrieved = assessment.evidence_by_claim.get(claim.claim_id, [])
        retrieved_by_id = {item.chunk_id: item for item in retrieved}
        if len(retrieved_by_id) != len(retrieved):
            raise ValueError("one Claim contains duplicate retrieved chunks")
        used_evidence: list[GroundedEvidenceReference] = []
        for chunk_id in verification.evidence_chunk_ids:
            item = retrieved_by_id.get(chunk_id)
            if item is None:
                raise ValueError(
                    f"VerificationResult references non-retrieved chunk {chunk_id}"
                )
            if item.resource_id != claim.resource_id:
                raise ValueError("grounded pack detected cross-resource Evidence")
            used_evidence.append(GroundedEvidenceReference(
                chunk_id=item.chunk_id,
                evidence_id=item.evidence_id,
                resource_id=item.resource_id,
                source_id=item.source_id,
                source_url=item.source_url,
                evidence_source_type=item.evidence_source_type,
                source_type=item.source_type,
                section_title=item.section_title,
                retrieval_score=item.retrieval_score,
                excerpt=item.text,
                excerpt_truncated=False,
            ))
        output.append(GroundedClaimReference(
            claim_id=claim.claim_id,
            resource_id=claim.resource_id,
            claim_type=claim.claim_type,
            text=claim.text,
            label=verification.label,
            verification_score=verification.score,
            rationale=verification.rationale,
            evidence=used_evidence,
        ))
    return output


def build_grounded_result_pack(result: SummerPipelineResult) -> GroundedResultPack:
    if result.status != "OK" or result.intent is None or not result.ranked_resource_ids:
        raise ValueError("GroundedResultPack can only be built from an OK S1 result")
    assessment_by_id = {item.candidate.resource_id: item for item in result.assessments}
    if len(assessment_by_id) != len(result.assessments):
        raise ValueError("SummerPipelineResult contains duplicate Candidate resources")

    recommendations: list[GroundedRecommendation] = []
    for rank, resource_id in enumerate(result.ranked_resource_ids, start=1):
        assessment = assessment_by_id.get(resource_id)
        if assessment is None:
            raise ValueError(f"ranked resource has no CandidateAssessment: {resource_id}")
        if assessment.decision.abstain:
            raise ValueError("an abstained Candidate cannot appear in ranked_resource_ids")
        if assessment.decision.resource_id != resource_id:
            raise ValueError("CandidateDecision resource does not match ranking")
        grounded_claims = _ground_claims(assessment)
        recommendations.append(GroundedRecommendation(
            rank=rank,
            resource_id=resource_id,
            title=assessment.candidate.title,
            url=assessment.candidate.url,
            retrieval_score=assessment.candidate.retrieval_score,
            confidence=assessment.confidence.confidence,
            confidence_bucket=assessment.confidence.bucket,
            final_score=assessment.decision.final_score,
            supported_claims=[item for item in grounded_claims if item.label == "SUPPORTED"],
            unknown_claims=[item for item in grounded_claims if item.label == "UNKNOWN"],
            contradicted_claims=[item for item in grounded_claims if item.label == "CONTRADICTED"],
        ))
    return GroundedResultPack(intent=result.intent, recommendations=recommendations)


def _view_with_limits(
    pack: GroundedResultPack,
    *,
    output_language: OutputLanguage,
    claim_limit: int,
    evidence_excerpt_limit: int,
    claim_text_chars: int,
    evidence_excerpt_chars: int,
) -> GroundedGenerationView:
    recommendations: list[GenerationRecommendationView] = []
    for recommendation in pack.recommendations:
        all_claims = (
            recommendation.supported_claims
            + recommendation.unknown_claims
            + recommendation.contradicted_claims
        )
        selected = all_claims[:claim_limit]
        omitted = all_claims[claim_limit:]
        claims: list[GenerationClaimView] = []
        for claim in selected:
            claim_text, _ = _truncate(claim.text, claim_text_chars)
            excerpts: list[GenerationEvidenceExcerpt] = []
            for evidence in claim.evidence[:evidence_excerpt_limit]:
                excerpt, truncated = _truncate(evidence.excerpt, evidence_excerpt_chars)
                excerpts.append(GenerationEvidenceExcerpt(
                    chunk_id=evidence.chunk_id,
                    evidence_id=evidence.evidence_id,
                    source_id=evidence.source_id,
                    source_url=evidence.source_url,
                    evidence_source_type=evidence.evidence_source_type,
                    section_title=evidence.section_title,
                    excerpt=excerpt,
                    excerpt_truncated=truncated,
                ))
            claims.append(GenerationClaimView(
                claim_id=claim.claim_id,
                claim_type=claim.claim_type,
                label=claim.label,
                claim_text=claim_text,
                evidence_chunk_ids=[item.chunk_id for item in claim.evidence],
                evidence_excerpts=excerpts,
            ))
        recommendations.append(GenerationRecommendationView(
            rank=recommendation.rank,
            resource_id=recommendation.resource_id,
            title=recommendation.title,
            retrieval_score=recommendation.retrieval_score,
            confidence_bucket=recommendation.confidence_bucket,
            claims=claims,
            omitted_claim_ids=[item.claim_id for item in omitted],
            omitted_evidence_chunk_ids=[
                evidence.chunk_id for claim in omitted for evidence in claim.evidence
            ],
        ))
    return GroundedGenerationView(
        output_language=output_language,
        intent=pack.intent,
        recommendations=recommendations,
    )


def _serialized_length(view: GroundedGenerationView) -> int:
    return len(json.dumps(
        view.model_dump(mode="json"), ensure_ascii=False,
        sort_keys=True, separators=(",", ":"),
    ))


def build_generation_view(
    pack: GroundedResultPack,
    *,
    output_language: OutputLanguage,
    config: GenerationConfig,
) -> GroundedGenerationView:
    """Build a deterministic bounded LLM view without changing the full pack."""

    claim_limit = config.max_claims_per_resource
    evidence_limit = config.max_evidence_excerpts_per_claim
    excerpt_chars = config.evidence_excerpt_chars
    view = _view_with_limits(
        pack,
        output_language=output_language,
        claim_limit=claim_limit,
        evidence_excerpt_limit=evidence_limit,
        claim_text_chars=config.claim_text_chars,
        evidence_excerpt_chars=excerpt_chars,
    )
    while _serialized_length(view) > config.max_input_chars and excerpt_chars > 80:
        excerpt_chars = max(80, excerpt_chars // 2)
        view = _view_with_limits(
            pack,
            output_language=output_language,
            claim_limit=claim_limit,
            evidence_excerpt_limit=evidence_limit,
            claim_text_chars=config.claim_text_chars,
            evidence_excerpt_chars=excerpt_chars,
        )
    if _serialized_length(view) > config.max_input_chars and evidence_limit:
        evidence_limit = 0
        view = _view_with_limits(
            pack,
            output_language=output_language,
            claim_limit=claim_limit,
            evidence_excerpt_limit=0,
            claim_text_chars=config.claim_text_chars,
            evidence_excerpt_chars=80,
        )
    while _serialized_length(view) > config.max_input_chars and claim_limit > 1:
        claim_limit -= 1
        view = _view_with_limits(
            pack,
            output_language=output_language,
            claim_limit=claim_limit,
            evidence_excerpt_limit=evidence_limit,
            claim_text_chars=config.claim_text_chars,
            evidence_excerpt_chars=excerpt_chars,
        )
    if _serialized_length(view) > config.max_input_chars:
        raise ValueError("minimum grounded generation view exceeds max_input_chars")
    return view


__all__ = ["build_generation_view", "build_grounded_result_pack"]
