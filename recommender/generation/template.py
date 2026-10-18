"""Deterministic grounded explanation fallback."""

from __future__ import annotations

from recommender.schemas import (
    FinalLLMOutput,
    FinalLLMRecommendationItem,
    FinalRecommendationExplanation,
    GroundedResultPack,
    OutputLanguage,
)

from .grounding import hydrate_final_explanation


def build_template_explanation(
    pack: GroundedResultPack,
    *,
    output_language: OutputLanguage,
) -> FinalRecommendationExplanation:
    if output_language == "zh":
        summary = "系统根据已完成的检索、证据验证和可信决策返回以下资源。"
        reason_with_support = "该资源通过了后端可信决策，并有可追溯的已验证声明。"
        reason_without_support = "该资源通过了后端相关性与可信决策；当前没有可作为确定事实展示的声明。"
    else:
        summary = "The system returned these resources after retrieval, evidence verification, and trust-aware decision-making."
        reason_with_support = "This resource passed the backend trust decision and has traceable verified claims."
        reason_without_support = "This resource passed the backend relevance and trust decision; no claim is currently presented as a verified fact."
    items = {item.resource_id: FinalLLMRecommendationItem(
        reason=reason_with_support if item.supported_claims else reason_without_support,
    ) for item in pack.recommendations}
    return hydrate_final_explanation(
        FinalLLMOutput(summary=summary, items=items),
        pack,
        output_language=output_language,
        generation_method="template_grounded_v1",
    )


__all__ = ["build_template_explanation"]
