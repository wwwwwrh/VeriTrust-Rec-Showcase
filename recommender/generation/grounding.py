"""Structural grounding checks and backend-controlled fact hydration."""

from __future__ import annotations

from recommender.schemas import (
    FinalLLMOutput,
    FinalRecommendationExplanation,
    FinalRecommendationItem,
    GenerationMethod,
    GroundedClaimReference,
    GroundedRecommendation,
    GroundedResultPack,
    GroundingValidationResult,
    OutputLanguage,
    ResolvedClaimReference,
)


CONFIDENCE_DISCLAIMERS: dict[str, str] = {
    "zh": "置信度是原型启发式分数，不是经过校准的正确概率。",
    "en": "Confidence is a prototype heuristic score, not a calibrated probability.",
}


def _claims_by_id(recommendation: GroundedRecommendation) -> dict[str, GroundedClaimReference]:
    claims = (
        recommendation.supported_claims
        + recommendation.unknown_claims
        + recommendation.contradicted_claims
    )
    return {item.claim_id: item for item in claims}


def validate_final_output(
    output: FinalLLMOutput,
    pack: GroundedResultPack,
) -> GroundingValidationResult:
    errors: list[str] = []
    expected = [item.resource_id for item in pack.recommendations]
    actual = list(output.items)
    if set(actual) != set(expected) or len(actual) != len(expected):
        errors.append(f"resource set mismatch: expected={expected}, actual={actual}")

    validated = expected if not errors else []
    return GroundingValidationResult(
        is_valid=not errors,
        errors=errors,
        validated_resource_ids=validated if not errors else [],
        fallback_required=bool(errors),
    )


def _resolve(claims: list[GroundedClaimReference]) -> list[ResolvedClaimReference]:
    return [ResolvedClaimReference(
            claim_id=claim.claim_id,
            claim_type=claim.claim_type,
            label=claim.label,
            text=claim.text,
            evidence_chunk_ids=[item.chunk_id for item in claim.evidence],
        ) for claim in claims]


def hydrate_final_explanation(
    output: FinalLLMOutput,
    pack: GroundedResultPack,
    *,
    output_language: OutputLanguage,
    generation_method: GenerationMethod,
) -> FinalRecommendationExplanation:
    validation = validate_final_output(output, pack)
    if not validation.is_valid:
        raise ValueError("cannot hydrate an ungrounded Final LLM output")
    items: list[FinalRecommendationItem] = []
    for recommendation in pack.recommendations:
        generated = output.items[recommendation.resource_id]
        items.append(FinalRecommendationItem(
            resource_id=recommendation.resource_id,
            rank=recommendation.rank,
            reason=generated.reason,
            verified_facts=_resolve(recommendation.supported_claims),
            uncertainty_notes=_resolve(recommendation.unknown_claims),
            contradicted_notes=_resolve(recommendation.contradicted_claims),
        ))
    return FinalRecommendationExplanation(
        output_language=output_language,
        summary=output.summary,
        items=items,
        confidence_disclaimer=CONFIDENCE_DISCLAIMERS[output_language],
        generation_method=generation_method,
    )


__all__ = [
    "CONFIDENCE_DISCLAIMERS", "hydrate_final_explanation", "validate_final_output",
]
