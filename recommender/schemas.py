"""Strict runtime schemas for the Summer Edition core pipeline."""

from __future__ import annotations

import re
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


Difficulty = Literal["beginner", "intermediate", "advanced", "unknown"]
ClaimType = Literal["uses_framework", "covers_topic", "is_maintained", "has_license"]
VerificationLabel = Literal["SUPPORTED", "CONTRADICTED", "UNKNOWN"]
ConfidenceBucket = Literal["High", "Medium", "Low"]
QueryParseStatus = Literal["SUCCESS", "ERROR"]
SummerQueryStatus = Literal[
    "OK", "QUERY_ERROR", "RECOMMENDATION_ABSTAINED", "RECOMMENDATION_ERROR"
]
GenerationStatus = Literal["NOT_RUN", "SUCCESS", "FALLBACK", "FAILED"]
OutputLanguage = Literal["zh", "en"]
GenerationMethod = Literal["qwen_grounded_v1", "template_grounded_v1"]

RESOURCE_ID_RE = re.compile(r"^github:\d+$")
CLAIM_ID_RE = re.compile(r"^claim_[0-9a-f]{16}$")
FACT_ID_RE = re.compile(r"^kg_[0-9a-f]{16}$")
EVIDENCE_ID_RE = re.compile(r"^ev_[0-9a-f]{16}$")
CHUNK_ID_RE = re.compile(r"^ch_[0-9a-f]{16}$")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _absolute_http_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("URL must be absolute HTTP(S)")
    return value


def _resource_id(value: str) -> str:
    if not RESOURCE_ID_RE.fullmatch(value):
        raise ValueError("resource_id must use github:<numeric_id>")
    return value


def _clean_list(value: object) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("value must be a list")
    output: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            raise ValueError("list items must be strings")
        cleaned = " ".join(item.split())
        if cleaned and cleaned.casefold() not in seen:
            seen.add(cleaned.casefold())
            output.append(cleaned)
    return output


class StructuredIntent(StrictModel):
    topics: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    prerequisites: list[str] = Field(default_factory=list)
    framework_preferences: list[str] = Field(default_factory=list)
    language: list[str] = Field(default_factory=list)
    difficulty: Difficulty = "unknown"
    learning_style: list[str] = Field(default_factory=list)
    time_budget: str | None = None
    constraints: list[str] = Field(default_factory=list)

    @field_validator(
        "topics", "skills", "prerequisites", "framework_preferences",
        "language", "learning_style", "constraints", mode="before",
    )
    @classmethod
    def normalize_lists(cls, value: object) -> list[str]:
        return _clean_list(value)

    @field_validator("time_budget")
    @classmethod
    def normalize_time_budget(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = " ".join(value.split())
        return cleaned or None


class QueryParseAudit(StrictModel):
    backend: str
    model: str
    prompt_version: str
    parse_status: QueryParseStatus
    retry_count: int = Field(ge=0)
    latency_ms: float = Field(ge=0)
    query_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    query_length: int = Field(ge=0)


class ParsedQuery(StrictModel):
    intent: StructuredIntent
    audit: QueryParseAudit


class Candidate(StrictModel):
    resource_id: str
    title: str
    url: str
    retrieval_score: float = Field(ge=0, le=1)
    retrieval_method: Literal["tfidf"] = "tfidf"

    _validate_resource_id = field_validator("resource_id")(_resource_id)
    _validate_url = field_validator("url")(_absolute_http_url)


class Claim(StrictModel):
    claim_id: str
    resource_id: str
    claim_type: ClaimType
    subject: str
    predicate: str
    object: str
    text: str
    generation_method: Literal["rule_v1"] = "rule_v1"

    _validate_resource_id = field_validator("resource_id")(_resource_id)

    @field_validator("claim_id")
    @classmethod
    def validate_claim_id(cls, value: str) -> str:
        if not CLAIM_ID_RE.fullmatch(value):
            raise ValueError("invalid claim_id")
        return value

    @field_validator("subject", "predicate", "object", "text")
    @classmethod
    def validate_nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("claim fields cannot be blank")
        return value


class RetrievedEvidence(StrictModel):
    chunk_id: str
    evidence_id: str
    resource_id: str
    source_id: str
    text: str
    source_url: str
    evidence_source_type: Literal["github_readme", "repo_metadata"]
    source_type: str
    source_reliability: float = Field(ge=0, le=1)
    source_reliability_method: Literal["heuristic_v0"]
    retrieval_score: float = Field(ge=0, le=1)
    section_title: str | None = None

    _validate_resource_id = field_validator("resource_id")(_resource_id)
    _validate_url = field_validator("source_url")(_absolute_http_url)


class VerificationResult(StrictModel):
    claim_id: str
    resource_id: str
    label: VerificationLabel
    score: float = Field(ge=0, le=1)
    method: Literal["rule_v1"] = "rule_v1"
    evidence_chunk_ids: list[str] = Field(default_factory=list)
    rationale: str

    _validate_resource_id = field_validator("resource_id")(_resource_id)


class ConfidenceBreakdown(StrictModel):
    confidence: float = Field(ge=0, le=1)
    bucket: ConfidenceBucket
    components: dict[str, float]
    weights: dict[str, float]
    method: Literal["prototype_heuristic_v1"] = "prototype_heuristic_v1"


class CandidateDecision(StrictModel):
    resource_id: str
    retrieval_score: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    confidence_bucket: ConfidenceBucket
    abstain: bool
    abstention_reason: str | None = None
    final_score: float = Field(ge=0, le=1)

    _validate_resource_id = field_validator("resource_id")(_resource_id)


class KGFact(StrictModel):
    fact_id: str
    resource_id: str
    predicate: Literal["covers", "uses", "written_in", "owned_by"]
    object_type: Literal["Topic", "Framework", "Language", "Owner"]
    object: str
    source_reference: str
    source_url: str
    source_hash: str
    extraction_method: Literal["canonical_resource_field_v1"] = "canonical_resource_field_v1"

    _validate_resource_id = field_validator("resource_id")(_resource_id)
    _validate_url = field_validator("source_url")(_absolute_http_url)

    @field_validator("fact_id")
    @classmethod
    def validate_fact_id(cls, value: str) -> str:
        if not FACT_ID_RE.fullmatch(value):
            raise ValueError("invalid KG fact_id")
        return value


class CandidateAssessment(StrictModel):
    candidate: Candidate
    claims: list[Claim] = Field(default_factory=list)
    evidence_by_claim: dict[str, list[RetrievedEvidence]] = Field(default_factory=dict)
    verifications: list[VerificationResult] = Field(default_factory=list)
    confidence: ConfidenceBreakdown
    decision: CandidateDecision
    errors: list[str] = Field(default_factory=list)


class SummerPipelineResult(StrictModel):
    status: Literal["OK", "ABSTAINED", "ERROR"]
    intent: StructuredIntent | None = None
    assessments: list[CandidateAssessment] = Field(default_factory=list)
    ranked_resource_ids: list[str] = Field(default_factory=list)
    error_code: str | None = None
    error_message: str | None = None
    warnings: list[str] = Field(default_factory=list)


class SummerQueryResult(StrictModel):
    """S2 envelope that keeps query failures separate from S1 abstention."""

    status: SummerQueryStatus
    parsed_query: ParsedQuery | None = None
    query_audit: QueryParseAudit
    recommendation: SummerPipelineResult | None = None
    error_code: str | None = None
    error_message: str | None = None
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_stage_boundary(self) -> "SummerQueryResult":
        if self.status == "QUERY_ERROR":
            if self.parsed_query is not None or self.recommendation is not None:
                raise ValueError("query errors cannot contain an S1 recommendation")
            if self.query_audit.parse_status != "ERROR" or not self.error_code:
                raise ValueError("query errors require failed audit metadata and an error code")
            return self
        if self.parsed_query is None or self.query_audit.parse_status != "SUCCESS":
            raise ValueError("recommendation outcomes require a successfully parsed query")
        if self.recommendation is None:
            raise ValueError("recommendation outcome is missing its S1 result")
        expected = {
            "OK": "OK",
            "RECOMMENDATION_ABSTAINED": "ABSTAINED",
            "RECOMMENDATION_ERROR": "ERROR",
        }[self.status]
        if self.recommendation.status != expected:
            raise ValueError("S2 status does not match the nested S1 result")
        return self


class GroundedEvidenceReference(StrictModel):
    """A verification-used canonical Evidence reference."""

    chunk_id: str
    evidence_id: str
    resource_id: str
    source_id: str
    source_url: str
    evidence_source_type: Literal["github_readme", "repo_metadata"]
    source_type: str
    section_title: str | None = None
    retrieval_score: float = Field(ge=0, le=1)
    excerpt: str
    excerpt_truncated: bool = False

    _validate_resource_id = field_validator("resource_id")(_resource_id)
    _validate_url = field_validator("source_url")(_absolute_http_url)

    @field_validator("chunk_id")
    @classmethod
    def validate_chunk_id(cls, value: str) -> str:
        if not CHUNK_ID_RE.fullmatch(value):
            raise ValueError("invalid chunk_id")
        return value

    @field_validator("evidence_id")
    @classmethod
    def validate_evidence_id(cls, value: str) -> str:
        if not EVIDENCE_ID_RE.fullmatch(value):
            raise ValueError("invalid evidence_id")
        return value


class GroundedClaimReference(StrictModel):
    claim_id: str
    resource_id: str
    claim_type: ClaimType
    text: str
    label: VerificationLabel
    verification_score: float = Field(ge=0, le=1)
    rationale: str
    evidence: list[GroundedEvidenceReference] = Field(default_factory=list)

    _validate_resource_id = field_validator("resource_id")(_resource_id)

    @field_validator("claim_id")
    @classmethod
    def validate_claim_id(cls, value: str) -> str:
        if not CLAIM_ID_RE.fullmatch(value):
            raise ValueError("invalid claim_id")
        return value

    @model_validator(mode="after")
    def validate_evidence_binding(self) -> "GroundedClaimReference":
        if any(item.resource_id != self.resource_id for item in self.evidence):
            raise ValueError("grounded Claim contains cross-resource Evidence")
        if len({item.chunk_id for item in self.evidence}) != len(self.evidence):
            raise ValueError("grounded Claim contains duplicate Evidence chunks")
        return self


class GroundedRecommendation(StrictModel):
    rank: int = Field(ge=1)
    resource_id: str
    title: str
    url: str
    retrieval_score: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    confidence_bucket: ConfidenceBucket
    final_score: float = Field(ge=0, le=1)
    supported_claims: list[GroundedClaimReference] = Field(default_factory=list)
    unknown_claims: list[GroundedClaimReference] = Field(default_factory=list)
    contradicted_claims: list[GroundedClaimReference] = Field(default_factory=list)

    _validate_resource_id = field_validator("resource_id")(_resource_id)
    _validate_url = field_validator("url")(_absolute_http_url)

    @model_validator(mode="after")
    def validate_claim_groups(self) -> "GroundedRecommendation":
        groups = (
            ("SUPPORTED", self.supported_claims),
            ("UNKNOWN", self.unknown_claims),
            ("CONTRADICTED", self.contradicted_claims),
        )
        all_claims: list[GroundedClaimReference] = []
        for label, claims in groups:
            if any(item.label != label for item in claims):
                raise ValueError(f"{label} Claim group contains a different label")
            if any(item.resource_id != self.resource_id for item in claims):
                raise ValueError("grounded recommendation contains cross-resource Claim")
            all_claims.extend(claims)
        if len({item.claim_id for item in all_claims}) != len(all_claims):
            raise ValueError("grounded recommendation contains duplicate Claim IDs")
        return self


class GroundedResultPack(StrictModel):
    pack_version: Literal["summer_grounded_pack_v1"] = "summer_grounded_pack_v1"
    decision_status: Literal["OK"] = "OK"
    intent: StructuredIntent
    recommendations: list[GroundedRecommendation]

    @model_validator(mode="after")
    def validate_ranked_recommendations(self) -> "GroundedResultPack":
        if not self.recommendations:
            raise ValueError("GroundedResultPack requires at least one recommendation")
        expected_ranks = list(range(1, len(self.recommendations) + 1))
        if [item.rank for item in self.recommendations] != expected_ranks:
            raise ValueError("GroundedResultPack ranks must be contiguous and ordered")
        ids = [item.resource_id for item in self.recommendations]
        if len(ids) != len(set(ids)):
            raise ValueError("GroundedResultPack contains duplicate resources")
        return self


class GenerationEvidenceExcerpt(StrictModel):
    chunk_id: str
    evidence_id: str
    source_id: str
    source_url: str
    evidence_source_type: Literal["github_readme", "repo_metadata"]
    section_title: str | None = None
    excerpt: str
    excerpt_truncated: bool


class GenerationClaimView(StrictModel):
    claim_id: str
    claim_type: ClaimType
    label: VerificationLabel
    claim_text: str
    evidence_chunk_ids: list[str] = Field(default_factory=list)
    evidence_excerpts: list[GenerationEvidenceExcerpt] = Field(default_factory=list)


class GenerationRecommendationView(StrictModel):
    rank: int = Field(ge=1)
    resource_id: str
    title: str
    retrieval_score: float = Field(ge=0, le=1)
    confidence_bucket: ConfidenceBucket
    claims: list[GenerationClaimView] = Field(default_factory=list)
    omitted_claim_ids: list[str] = Field(default_factory=list)
    omitted_evidence_chunk_ids: list[str] = Field(default_factory=list)


class GroundedGenerationView(StrictModel):
    view_version: Literal["summer_generation_view_v1"] = "summer_generation_view_v1"
    output_language: OutputLanguage
    intent: StructuredIntent
    recommendations: list[GenerationRecommendationView]


class FinalLLMRecommendationItem(StrictModel):
    reason: str

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("reason cannot be blank")
        return value.strip()


class FinalLLMOutput(StrictModel):
    """The deliberately narrow schema returned by the Final LLM."""

    summary: str
    items: dict[str, FinalLLMRecommendationItem]

    @field_validator("summary")
    @classmethod
    def validate_summary(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("summary cannot be blank")
        return value.strip()


class ResolvedClaimReference(StrictModel):
    claim_id: str
    claim_type: ClaimType
    label: VerificationLabel
    text: str
    evidence_chunk_ids: list[str] = Field(default_factory=list)


class FinalRecommendationItem(StrictModel):
    resource_id: str
    rank: int = Field(ge=1)
    reason: str
    verified_facts: list[ResolvedClaimReference] = Field(default_factory=list)
    uncertainty_notes: list[ResolvedClaimReference] = Field(default_factory=list)
    contradicted_notes: list[ResolvedClaimReference] = Field(default_factory=list)

    _validate_resource_id = field_validator("resource_id")(_resource_id)


class FinalRecommendationExplanation(StrictModel):
    output_language: OutputLanguage
    summary: str
    items: list[FinalRecommendationItem]
    confidence_disclaimer: str
    generation_method: GenerationMethod


class GroundingValidationResult(StrictModel):
    is_valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    validated_resource_ids: list[str] = Field(default_factory=list)
    fallback_required: bool


class GenerationAudit(StrictModel):
    backend: str
    model: str
    prompt_version: str
    generation_status: GenerationStatus
    retry_count: int = Field(ge=0)
    latency_ms: float = Field(ge=0)
    grounding_valid: bool
    fallback_used: bool
    resource_count: int = Field(ge=0)
    input_character_count: int = Field(ge=0)
    prompt_eval_tokens: int | None = Field(default=None, ge=0)
    output_eval_tokens: int | None = Field(default=None, ge=0)
    error_code: str | None = None


class UserFacingEvidence(StrictModel):
    evidence: RetrievedEvidence
    used_for_verification: bool


class UserFacingClaim(StrictModel):
    claim: Claim
    verification: VerificationResult
    evidence: list[UserFacingEvidence] = Field(default_factory=list)


class UserFacingRecommendation(StrictModel):
    rank: int = Field(ge=1)
    candidate: Candidate
    confidence: ConfidenceBreakdown
    decision: CandidateDecision
    claims: list[UserFacingClaim] = Field(default_factory=list)
    explanation: FinalRecommendationItem | None = None


class PipelineTrace(StrictModel):
    query_status: QueryParseStatus
    query_latency_ms: float = Field(ge=0)
    recommendation_status: Literal["NOT_RUN", "OK", "ABSTAINED", "ERROR"]
    core_latency_ms: float = Field(ge=0)
    candidate_count: int = Field(ge=0)
    claim_count: int = Field(ge=0)
    retrieved_evidence_count: int = Field(ge=0)
    supported_count: int = Field(ge=0)
    unknown_count: int = Field(ge=0)
    contradicted_count: int = Field(ge=0)
    accepted_count: int = Field(ge=0)
    abstained_count: int = Field(ge=0)
    generation_status: GenerationStatus
    generation_latency_ms: float = Field(ge=0)
    total_latency_ms: float = Field(ge=0)


class UserFacingResult(StrictModel):
    status: SummerQueryStatus
    intent: StructuredIntent | None = None
    query_audit: QueryParseAudit
    recommendations: list[UserFacingRecommendation] = Field(default_factory=list)
    final_explanation: FinalRecommendationExplanation | None = None
    generation_audit: GenerationAudit
    pipeline_trace: PipelineTrace
    error_code: str | None = None
    error_message: str | None = None
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_stage_boundaries(self) -> "UserFacingResult":
        if self.status != "OK":
            if self.recommendations or self.final_explanation is not None:
                raise ValueError("non-OK user results cannot contain accepted recommendations")
            if self.generation_audit.generation_status != "NOT_RUN":
                raise ValueError("Final Generation must not run for non-OK recommendations")
            return self
        if not self.recommendations:
            raise ValueError("OK user result requires accepted recommendations")
        generation_status = self.generation_audit.generation_status
        if generation_status in {"SUCCESS", "FALLBACK"} and self.final_explanation is None:
            raise ValueError("successful or fallback generation requires an explanation")
        if generation_status == "FAILED" and self.final_explanation is not None:
            raise ValueError("failed generation cannot contain a final explanation")
        if generation_status == "NOT_RUN":
            raise ValueError("Final Generation must be attempted for an OK recommendation")
        return self


__all__ = [
    "Candidate", "CandidateAssessment", "CandidateDecision", "Claim",
    "ConfidenceBreakdown", "FinalLLMOutput",
    "FinalLLMRecommendationItem", "FinalRecommendationExplanation",
    "FinalRecommendationItem", "GenerationAudit", "GenerationClaimView",
    "GenerationEvidenceExcerpt", "GenerationRecommendationView",
    "GenerationStatus", "GroundedClaimReference", "GroundedEvidenceReference",
    "GroundedGenerationView", "GroundedRecommendation", "GroundedResultPack",
    "GroundingValidationResult", "KGFact", "OutputLanguage", "ParsedQuery",
    "PipelineTrace", "QueryParseAudit", "ResolvedClaimReference",
    "RetrievedEvidence", "StructuredIntent", "SummerPipelineResult",
    "SummerQueryResult", "UserFacingClaim", "UserFacingEvidence",
    "UserFacingRecommendation", "UserFacingResult", "VerificationResult",
]
