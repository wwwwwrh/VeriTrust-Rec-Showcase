"""Five-component, non-calibrated Summer Edition confidence heuristic."""

from __future__ import annotations

from recommender.config import ConfidenceConfig
from recommender.schemas import ConfidenceBreakdown


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def estimate_confidence(
    *,
    retrieval_score: float,
    evidence_coverage: float,
    verification_score: float,
    kg_support: float,
    source_reliability: float,
    config: ConfidenceConfig,
) -> ConfidenceBreakdown:
    components = {
        "retrieval_score": _clamp(retrieval_score),
        "evidence_coverage": _clamp(evidence_coverage),
        "verification_score": _clamp(verification_score),
        "kg_support": _clamp(kg_support),
        "source_reliability": _clamp(source_reliability),
    }
    weights = {
        "w_retrieval": config.w_retrieval,
        "w_coverage": config.w_coverage,
        "w_verification": config.w_verification,
        "w_kg_support": config.w_kg_support,
        "w_source_reliability": config.w_source_reliability,
    }
    confidence = round(_clamp(
        config.w_retrieval * components["retrieval_score"]
        + config.w_coverage * components["evidence_coverage"]
        + config.w_verification * components["verification_score"]
        + config.w_kg_support * components["kg_support"]
        + config.w_source_reliability * components["source_reliability"]
    ), 6)
    bucket = (
        "High" if confidence >= config.high_threshold
        else "Medium" if confidence >= config.low_threshold
        else "Low"
    )
    return ConfidenceBreakdown(
        confidence=confidence,
        bucket=bucket,  # type: ignore[arg-type]
        components=components,
        weights=weights,
    )


__all__ = ["estimate_confidence"]
