"""Render-only Streamlit components over the stable UserFacingResult schema."""

from __future__ import annotations

import json

import streamlit as st

from recommender.schemas import StructuredIntent, UserFacingResult

from .i18n import text


LABEL_ICON = {"SUPPORTED": "✅", "UNKNOWN": "❓", "CONTRADICTED": "❌"}


def render_intent(intent: StructuredIntent | None, language: str) -> None:
    if intent is None:
        return
    st.subheader(text(language, "intent"))
    data = intent.model_dump(mode="json")
    columns = st.columns(3)
    for index, (key, value) in enumerate(data.items()):
        shown = ", ".join(value) if isinstance(value, list) else value
        columns[index % 3].markdown(f"**{key}**  \n{shown or '—'}")
    if intent.language:
        st.caption(text(language, "language_notice"))


def render_status(result: UserFacingResult, language: str) -> None:
    if result.status == "QUERY_ERROR":
        st.error(f"{text(language, 'query_error')}: {result.error_code or ''}")
    elif result.status == "RECOMMENDATION_ERROR":
        st.error(f"{text(language, 'recommendation_error')}: {result.error_code or ''}")
    elif result.status == "RECOMMENDATION_ABSTAINED":
        st.warning(f"{text(language, 'no_recommendation')} — {result.error_code or ''}")
    elif result.generation_audit.generation_status == "FALLBACK":
        st.warning(text(language, "generation_fallback"))
    elif result.generation_audit.generation_status == "FAILED":
        st.warning(text(language, "generation_failed"))


def render_recommendations(result: UserFacingResult, language: str) -> None:
    render_status(result, language)
    if result.status != "OK":
        return
    if result.final_explanation:
        st.markdown(result.final_explanation.summary)
    for recommendation in result.recommendations:
        with st.container(border=True):
            candidate = recommendation.candidate
            decision = recommendation.decision
            st.subheader(f"#{recommendation.rank} · {candidate.title}")
            st.markdown(f"[{candidate.url}]({candidate.url})")
            first, second, third = st.columns(3)
            first.metric("Relevance", f"{candidate.retrieval_score:.3f}")
            second.metric("Confidence", f"{recommendation.confidence.confidence:.3f}")
            third.metric("Final Score", f"{decision.final_score:.3f}")
            st.caption(
                f"{recommendation.confidence.bucket} · "
                + text(language, "confidence_notice")
            )
            if recommendation.explanation:
                st.markdown(f"**{text(language, 'why')}**")
                st.write(recommendation.explanation.reason)
                if recommendation.explanation.verified_facts:
                    st.markdown("**✅ SUPPORTED**")
                    for fact in recommendation.explanation.verified_facts:
                        st.write(f"- {fact.text} (`{fact.claim_id}`)")
                if recommendation.explanation.uncertainty_notes:
                    st.markdown("**❓ UNKNOWN**")
                    for fact in recommendation.explanation.uncertainty_notes:
                        st.write(f"- {fact.text} (`{fact.claim_id}`)")
                if recommendation.explanation.contradicted_notes:
                    st.markdown("**❌ CONTRADICTED**")
                    for fact in recommendation.explanation.contradicted_notes:
                        st.write(f"- {fact.text} (`{fact.claim_id}`)")


def render_evidence(result: UserFacingResult, excerpt_chars: int) -> None:
    if not result.recommendations:
        st.info("No accepted recommendation Evidence to display.")
        return
    for recommendation in result.recommendations:
        st.subheader(f"#{recommendation.rank} · {recommendation.candidate.title}")
        for claim_result in recommendation.claims:
            verification = claim_result.verification
            claim = claim_result.claim
            icon = LABEL_ICON[verification.label]
            with st.expander(f"{icon} {verification.label} · {claim.text}"):
                st.markdown(f"**Claim ID:** `{claim.claim_id}`")
                st.write(verification.rationale)
                if not claim_result.evidence:
                    st.caption("No retrieved Evidence for this Claim.")
                for entry in claim_result.evidence:
                    evidence = entry.evidence
                    used = "used by verifier" if entry.used_for_verification else "retrieved only"
                    st.markdown(
                        f"**{evidence.evidence_source_type}** · {used} · "
                        f"score `{evidence.retrieval_score:.3f}`"
                    )
                    st.markdown(f"[Source]({evidence.source_url})")
                    st.caption(
                        f"Evidence `{evidence.evidence_id}` · Chunk `{evidence.chunk_id}` · "
                        f"Section `{evidence.section_title or '—'}`"
                    )
                    excerpt = evidence.text[:excerpt_chars]
                    if len(evidence.text) > excerpt_chars:
                        excerpt += "…"
                    st.code(excerpt, language=None)


def render_trace(result: UserFacingResult, debug: bool) -> None:
    trace = result.pipeline_trace
    st.json(trace.model_dump(mode="json"), expanded=True)
    if debug:
        st.markdown("**Audit-safe debug result**")
        st.code(json.dumps(
            result.model_dump(mode="json"), ensure_ascii=False, indent=2,
        ), language="json")


__all__ = [
    "render_evidence", "render_intent", "render_recommendations",
    "render_status", "render_trace",
]
