"""Candidate-local abstention and lightweight trust-aware ranking score."""

from __future__ import annotations

from recommender.config import AbstentionConfig, RankingConfig
from recommender.schemas import Candidate, CandidateDecision, ConfidenceBreakdown, VerificationResult


def decide_candidate(
    candidate: Candidate,
    confidence: ConfidenceBreakdown,
    *,
    evidence_coverage: float,
    verifications: list[VerificationResult],
    local_errors: list[str],
    abstention: AbstentionConfig,
    ranking: RankingConfig,
) -> CandidateDecision:
    final_score = round(min(1.0, max(0.0,
        ranking.w_relevance * candidate.retrieval_score
        + ranking.w_confidence * confidence.confidence
    )), 6)
    reason: str | None = None
    if local_errors:
        reason = f"candidate-local critical error: {local_errors[0]}"
    elif abstention.abstain_on_contradiction and any(
        item.label == "CONTRADICTED" for item in verifications
    ):
        reason = "a conservatively verified CONTRADICTED claim is present"
    elif evidence_coverage <= 0:
        reason = "no retrieved evidence covers any claim"
    elif confidence.confidence < abstention.confidence_threshold:
        reason = (
            f"prototype heuristic confidence {confidence.confidence:.3f} "
            f"is below {abstention.confidence_threshold:.3f}"
        )
    return CandidateDecision(
        resource_id=candidate.resource_id,
        retrieval_score=candidate.retrieval_score,
        confidence=confidence.confidence,
        confidence_bucket=confidence.bucket,
        abstain=reason is not None,
        abstention_reason=reason,
        final_score=final_score,
    )


__all__ = ["decide_candidate"]
