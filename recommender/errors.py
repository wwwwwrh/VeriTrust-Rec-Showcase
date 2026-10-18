"""Typed runtime errors and their Summer Edition scopes."""

from __future__ import annotations

from enum import StrEnum


class GlobalErrorCode(StrEnum):
    CONFIG_INVALID = "CONFIG_INVALID"
    INTENT_INVALID = "INTENT_INVALID"
    RESOURCE_DATASET_MISSING = "RESOURCE_DATASET_MISSING"
    RESOURCE_DATASET_INVALID = "RESOURCE_DATASET_INVALID"
    EVIDENCE_CORPUS_MISSING = "EVIDENCE_CORPUS_MISSING"
    EVIDENCE_CORPUS_INVALID = "EVIDENCE_CORPUS_INVALID"
    EVIDENCE_BUILD_INCOMPATIBLE = "EVIDENCE_BUILD_INCOMPATIBLE"
    CANDIDATE_RETRIEVAL_FAILED = "CANDIDATE_RETRIEVAL_FAILED"
    KG_BUILD_FAILED = "KG_BUILD_FAILED"


class CandidateErrorCode(StrEnum):
    CLAIM_GENERATION_FAILED = "CLAIM_GENERATION_FAILED"
    EVIDENCE_RETRIEVAL_FAILED = "EVIDENCE_RETRIEVAL_FAILED"
    EVIDENCE_ATTRIBUTION_ERROR = "EVIDENCE_ATTRIBUTION_ERROR"
    VERIFICATION_FAILED = "VERIFICATION_FAILED"


class QueryErrorCode(StrEnum):
    """Technical failures that occur before the S1 recommendation core runs."""

    QUERY_EMPTY = "QUERY_EMPTY"
    LOCAL_LLM_NOT_CONFIGURED = "LOCAL_LLM_NOT_CONFIGURED"
    LOCAL_LLM_UNAVAILABLE = "LOCAL_LLM_UNAVAILABLE"
    LOCAL_LLM_TIMEOUT = "LOCAL_LLM_TIMEOUT"
    LOCAL_LLM_RESPONSE_INVALID = "LOCAL_LLM_RESPONSE_INVALID"
    QUERY_PARSE_FAILED = "QUERY_PARSE_FAILED"


class GenerationErrorCode(StrEnum):
    """Failures in S3 explanation generation, after recommendation decisions."""

    FINAL_LLM_UNAVAILABLE = "FINAL_LLM_UNAVAILABLE"
    FINAL_LLM_TIMEOUT = "FINAL_LLM_TIMEOUT"
    FINAL_GENERATION_INVALID = "FINAL_GENERATION_INVALID"
    FINAL_GROUNDING_FAILED = "FINAL_GROUNDING_FAILED"


class PipelineError(RuntimeError):
    """A typed error whose code determines pipeline scope."""

    def __init__(self, code: GlobalErrorCode | CandidateErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code


class QueryProcessingError(RuntimeError):
    """A typed S2 error carrying audit-safe timing and retry metadata."""

    def __init__(
        self,
        code: QueryErrorCode,
        message: str,
        *,
        retry_count: int = 0,
        latency_ms: float = 0.0,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retry_count = retry_count
        self.latency_ms = latency_ms


class GenerationProcessingError(RuntimeError):
    def __init__(
        self,
        code: GenerationErrorCode,
        message: str,
        *,
        retry_count: int = 0,
        latency_ms: float = 0.0,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retry_count = retry_count
        self.latency_ms = latency_ms


__all__ = [
    "CandidateErrorCode", "GenerationErrorCode", "GenerationProcessingError",
    "GlobalErrorCode", "PipelineError", "QueryErrorCode", "QueryProcessingError",
]
