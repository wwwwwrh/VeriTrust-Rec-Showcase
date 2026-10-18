"""Conservative rules over same-resource Phase 3 EvidenceChunks."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone

from recommender.config import VerificationConfig
from recommender.errors import CandidateErrorCode, PipelineError
from recommender.kg.graph import KnowledgeGraph
from recommender.retrieval.evidence import normalized_phrase
from recommender.schemas import Claim, RetrievedEvidence, VerificationResult


FRAMEWORK_ALIASES = {
    "pytorch": ("pytorch", "torch"),
    "tensorflow": ("tensorflow",),
    "keras": ("keras",),
    "jax": ("jax",),
    "huggingface": ("huggingface", "hugging face", "transformers"),
    "scikit learn": ("scikit learn", "sklearn"),
}


@dataclass(frozen=True)
class VerificationSummary:
    total_claims: int
    supported: int
    contradicted: int
    unknown: int
    evidence_coverage: float
    verification_score: float
    kg_support: float
    source_reliability: float
    unique_verification_sources: int


def parse_metadata(text: str) -> dict[str, object]:
    output: dict[str, object] = {}
    for line in text.splitlines():
        key, separator, raw_value = line.partition(":")
        if not separator:
            continue
        try:
            output[key.strip()] = json.loads(raw_value.strip())
        except json.JSONDecodeError:
            continue
    return output


def _aliases(value: str) -> tuple[str, ...]:
    key = normalized_phrase(value)
    return FRAMEWORK_ALIASES.get(key, (key,))


def _contains(text: str, values: tuple[str, ...]) -> bool:
    normalized = normalized_phrase(text)
    return any(value and value in normalized for value in values)


def _explicit_relation(text: str, values: tuple[str, ...], positive: tuple[str, ...]) -> bool:
    normalized = normalized_phrase(text)
    for value in values:
        if not value:
            continue
        for signal in positive:
            if re.search(rf"\b{re.escape(signal)}\b.{{0,80}}\b{re.escape(value)}\b", normalized):
                return True
            if re.search(rf"\b{re.escape(value)}\b.{{0,80}}\b{re.escape(signal)}\b", normalized):
                return True
    return False


def _explicit_negation(text: str, values: tuple[str, ...]) -> bool:
    normalized = normalized_phrase(text)
    signals = (
        "does not use", "does not support", "not using", "not based on",
        "no support for", "without support for", "does not cover", "not covered",
    )
    return any(
        re.search(rf"\b{re.escape(signal)}\b.{{0,80}}\b{re.escape(value)}\b", normalized)
        for signal in signals for value in values if value
    )


def _result(
    claim: Claim,
    label: str,
    score: float,
    evidence: list[RetrievedEvidence],
    rationale: str,
) -> VerificationResult:
    return VerificationResult(
        claim_id=claim.claim_id,
        resource_id=claim.resource_id,
        label=label,  # type: ignore[arg-type]
        score=score,
        evidence_chunk_ids=[item.chunk_id for item in evidence],
        rationale=rationale,
    )


def _verify_framework(
    claim: Claim, evidence: list[RetrievedEvidence], config: VerificationConfig,
) -> VerificationResult:
    aliases = _aliases(claim.object)
    related = [item for item in evidence if _contains(item.text, aliases)]
    contradicted = [item for item in related if _explicit_negation(item.text, aliases)]
    if contradicted:
        return _result(claim, "CONTRADICTED", config.contradicted_score, contradicted,
                       "Evidence explicitly denies use or support of the requested framework.")
    positive = ("uses", "using", "based on", "built with", "requires", "install", "dependency")
    supported = [item for item in related if _explicit_relation(item.text, aliases, positive)]
    if supported:
        return _result(claim, "SUPPORTED", config.supported_readme_score, supported,
                       "Evidence explicitly states use of or dependency on the framework.")
    return _result(claim, "UNKNOWN", config.unknown_related_score if related else 0.0, related,
                   "The framework is only mentioned or no sufficient evidence was retrieved.")


def _verify_topic(
    claim: Claim, evidence: list[RetrievedEvidence], config: VerificationConfig,
) -> VerificationResult:
    wanted = normalized_phrase(claim.object)
    metadata_matches: list[RetrievedEvidence] = []
    readme_related: list[RetrievedEvidence] = []
    for item in evidence:
        if item.evidence_source_type == "repo_metadata":
            topics = parse_metadata(item.text).get("topics")
            if isinstance(topics, list) and any(normalized_phrase(str(value)) == wanted for value in topics):
                metadata_matches.append(item)
        elif wanted and wanted in normalized_phrase(item.text):
            readme_related.append(item)
    if metadata_matches:
        return _result(claim, "SUPPORTED", config.supported_metadata_score, metadata_matches,
                       "The canonical repository metadata explicitly lists the topic.")
    contradicted = [item for item in readme_related if _explicit_negation(item.text, (wanted,))]
    if contradicted:
        return _result(claim, "CONTRADICTED", config.contradicted_score, contradicted,
                       "Evidence explicitly denies covering the topic.")
    positive = ("covers", "topic", "tutorial", "guide", "course", "implements", "implementation")
    supported = [item for item in readme_related if _explicit_relation(item.text, (wanted,), positive)]
    if supported:
        return _result(claim, "SUPPORTED", config.supported_readme_score, supported,
                       "README evidence explicitly describes coverage of the topic.")
    return _result(claim, "UNKNOWN", config.unknown_related_score if readme_related else 0.0, readme_related,
                   "No evidence explicitly establishes topic coverage.")


def _parse_utc(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone(timezone.utc) if parsed.tzinfo else None


def _verify_maintenance(
    claim: Claim,
    evidence: list[RetrievedEvidence],
    config: VerificationConfig,
    evaluation_time: datetime,
) -> VerificationResult:
    metadata = [item for item in evidence if item.evidence_source_type == "repo_metadata"]
    for item in metadata:
        values = parse_metadata(item.text)
        if values.get("is_archived") is True:
            return _result(claim, "CONTRADICTED", config.contradicted_score, [item],
                           "Repository metadata explicitly marks the project as archived.")
        pushed_at = _parse_utc(values.get("pushed_at"))
        if values.get("is_archived") is False and pushed_at is not None:
            age_days = max(0.0, (evaluation_time.astimezone(timezone.utc) - pushed_at).total_seconds() / 86400)
            if age_days <= config.maintenance_max_age_days:
                return _result(claim, "SUPPORTED", config.supported_metadata_score, [item],
                               "The same metadata evidence reports a non-archived repository and recent pushed_at.")
            return _result(claim, "UNKNOWN", config.unknown_related_score, [item],
                           "The repository is not archived, but pushed_at is outside the configured window.")
    negative = [
        item for item in evidence
        if item.evidence_source_type == "github_readme"
        and re.search(r"\b(deprecated|unmaintained|no longer maintained|archived project)\b", item.text, re.I)
    ]
    if negative:
        return _result(claim, "CONTRADICTED", config.contradicted_score, negative,
                       "README evidence explicitly describes the project as deprecated or unmaintained.")
    return _result(claim, "UNKNOWN", 0.0, [], "No relevant maintenance metadata was retrieved.")


def _verify_license(
    claim: Claim, evidence: list[RetrievedEvidence], config: VerificationConfig,
) -> VerificationResult:
    allowed = {value.casefold() for value in config.open_source_spdx_ids}
    metadata = [item for item in evidence if item.evidence_source_type == "repo_metadata"]
    related_metadata: list[RetrievedEvidence] = []
    for item in metadata:
        value = parse_metadata(item.text).get("license_spdx_id")
        if value is not None:
            related_metadata.append(item)
        if isinstance(value, str) and value.casefold() in allowed:
            return _result(claim, "SUPPORTED", config.supported_metadata_score, [item],
                           "Repository metadata contains a recognized open-source SPDX license.")
    explicit_negative = [
        item for item in evidence
        if re.search(r"\b(proprietary|all rights reserved|no open.source licen[cs]e)\b", item.text, re.I)
    ]
    if explicit_negative:
        return _result(claim, "CONTRADICTED", config.contradicted_score, explicit_negative,
                       "Evidence explicitly states that no open-source license is granted.")
    # A different valid SPDX identifier is not a contradiction; exact-license comparison is out of scope.
    return _result(claim, "UNKNOWN", config.unknown_related_score if related_metadata else 0.0,
                   related_metadata, "No recognized open-source SPDX license was established.")


def verify_claim(
    claim: Claim,
    evidence: list[RetrievedEvidence],
    *,
    config: VerificationConfig,
    evaluation_time: datetime,
) -> VerificationResult:
    if evaluation_time.tzinfo is None:
        raise ValueError("evaluation_time must include a timezone")
    if any(item.resource_id != claim.resource_id for item in evidence):
        raise PipelineError(
            CandidateErrorCode.EVIDENCE_ATTRIBUTION_ERROR,
            f"claim {claim.claim_id} received cross-resource Evidence",
        )
    if claim.claim_type == "uses_framework":
        return _verify_framework(claim, evidence, config)
    if claim.claim_type == "covers_topic":
        return _verify_topic(claim, evidence, config)
    if claim.claim_type == "is_maintained":
        return _verify_maintenance(claim, evidence, config, evaluation_time)
    return _verify_license(claim, evidence, config)


def summarize_verification(
    claims: list[Claim],
    evidence_by_claim: dict[str, list[RetrievedEvidence]],
    results: list[VerificationResult],
    kg: KnowledgeGraph,
) -> VerificationSummary:
    if len(results) != len(claims):
        raise ValueError("each Claim must have exactly one VerificationResult")
    total = len(claims)
    supported = sum(item.label == "SUPPORTED" for item in results)
    contradicted = sum(item.label == "CONTRADICTED" for item in results)
    unknown = sum(item.label == "UNKNOWN" for item in results)
    covered = sum(bool(evidence_by_claim.get(claim.claim_id)) for claim in claims)

    evidence_lookup = {
        item.chunk_id: item
        for items in evidence_by_claim.values()
        for item in items
    }
    sources: dict[tuple[str, str], float] = {}
    for result in results:
        for chunk_id in result.evidence_chunk_ids:
            item = evidence_lookup.get(chunk_id)
            if item is None:
                raise ValueError(f"VerificationResult references non-retrieved chunk {chunk_id}")
            key = (item.evidence_id, item.source_id)
            previous = sources.setdefault(key, item.source_reliability)
            if previous != item.source_reliability:
                raise ValueError("one Evidence source has inconsistent source reliability")

    applicable = [claim for claim in claims if claim.claim_type in {"uses_framework", "covers_topic"}]
    kg_support = (
        sum(kg.supports_claim(claim) for claim in applicable) / len(applicable)
        if applicable else 0.5
    )
    return VerificationSummary(
        total_claims=total,
        supported=supported,
        contradicted=contradicted,
        unknown=unknown,
        evidence_coverage=round(covered / total, 6) if total else 0.0,
        verification_score=round(supported / total, 6) if total else 0.0,
        kg_support=round(kg_support, 6),
        source_reliability=round(sum(sources.values()) / len(sources), 6) if sources else 0.0,
        unique_verification_sources=len(sources),
    )


__all__ = ["VerificationSummary", "parse_metadata", "summarize_verification", "verify_claim"]
