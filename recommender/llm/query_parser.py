"""Schema-constrained, conservative natural-language query parsing."""

from __future__ import annotations

import copy
import hashlib
import json
from time import perf_counter

from pydantic import ValidationError

from recommender.config import LLMConfig, QueryPromptConfig
from recommender.errors import QueryErrorCode, QueryProcessingError
from recommender.schemas import ParsedQuery, QueryParseAudit, StructuredIntent

from .client import LocalLLMClient


def _query_hash(query: str) -> str:
    return "sha256:" + hashlib.sha256(query.encode("utf-8")).hexdigest()


def structured_intent_response_schema() -> dict:
    """Require a complete object while retaining the canonical S1 field schema."""

    schema = copy.deepcopy(StructuredIntent.model_json_schema())
    schema["required"] = list(schema["properties"])
    descriptions = {
        "topics": "Topics the user explicitly wants to learn; lexically normalize unambiguous AI/ML concepts to canonical English without adding concepts.",
        "skills": "Skills the user explicitly says they already possess; use canonical English technical terminology when unambiguous.",
        "prerequisites": "Requirements the user explicitly places on a learning resource.",
        "framework_preferences": "Only positively requested software frameworks, using canonical product spelling.",
        "language": "Explicit human-language preference for learning materials, never the query language.",
        "difficulty": "Explicit requested difficulty; unknown when absent.",
        "learning_style": "Explicit style using hands_on, theory, project_based, or tutorial; explicit theory, principles, or mathematics requires theory.",
        "time_budget": "Explicit duration or schedule, otherwise null.",
        "constraints": "Explicit residual exclusions or requirements only; never duplicate information captured in a dedicated field.",
    }
    for field, description in descriptions.items():
        schema["properties"][field]["description"] = description
    return schema


def _normalize_missing_sentinel(payload: object) -> object:
    if not isinstance(payload, dict):
        return payload
    normalized = dict(payload)
    value = normalized.get("time_budget")
    if isinstance(value, str) and value.strip().casefold() == "unknown":
        normalized["time_budget"] = None
    return normalized


class QueryParser:
    def __init__(
        self,
        client: LocalLLMClient,
        *,
        llm_config: LLMConfig,
        prompt_config: QueryPromptConfig,
    ) -> None:
        self.client = client
        self.llm_config = llm_config
        self.prompt_config = prompt_config

    def _audit(
        self,
        query: str,
        *,
        status: str,
        retry_count: int,
        latency_ms: float,
    ) -> QueryParseAudit:
        return QueryParseAudit(
            backend=self.client.backend,
            model=self.client.model,
            prompt_version=self.prompt_config.prompt_version,
            parse_status=status,
            retry_count=retry_count,
            latency_ms=round(max(0.0, latency_ms), 3),
            query_hash=_query_hash(query),
            query_length=len(query),
        )

    def parse(self, query: str) -> ParsedQuery:
        started = perf_counter()
        cleaned_query = query.strip()
        if not cleaned_query:
            raise QueryProcessingError(
                QueryErrorCode.QUERY_EMPTY,
                "query cannot be empty",
                latency_ms=(perf_counter() - started) * 1000,
            )

        schema = structured_intent_response_schema()
        last_invalid_reason = "invalid structured response"
        for attempt in range(self.llm_config.max_retries + 1):
            system_prompt = self.prompt_config.system_prompt
            if attempt:
                system_prompt = f"{system_prompt}\n\n{self.prompt_config.retry_instruction}"
            try:
                response = self.client.generate_structured(
                    system_prompt=system_prompt,
                    user_prompt=cleaned_query,
                    schema=schema,
                    temperature=self.llm_config.temperature,
                    max_tokens=self.llm_config.max_tokens,
                    context_length=self.llm_config.context_length,
                    thinking=self.llm_config.thinking,
                )
            except QueryProcessingError as exc:
                latency_ms = (perf_counter() - started) * 1000
                if exc.code == QueryErrorCode.LOCAL_LLM_RESPONSE_INVALID:
                    last_invalid_reason = str(exc)
                    if attempt < self.llm_config.max_retries:
                        continue
                    break
                raise QueryProcessingError(
                    exc.code,
                    str(exc),
                    retry_count=attempt,
                    latency_ms=latency_ms,
                ) from exc

            try:
                raw = json.loads(response)
                normalized = _normalize_missing_sentinel(raw)
                intent = StructuredIntent.model_validate(normalized)
            except (json.JSONDecodeError, ValidationError, TypeError) as exc:
                last_invalid_reason = type(exc).__name__
                if attempt < self.llm_config.max_retries:
                    continue
                break

            latency_ms = (perf_counter() - started) * 1000
            return ParsedQuery(
                intent=intent,
                audit=self._audit(
                    cleaned_query,
                    status="SUCCESS",
                    retry_count=attempt,
                    latency_ms=latency_ms,
                ),
            )

        latency_ms = (perf_counter() - started) * 1000
        raise QueryProcessingError(
            QueryErrorCode.QUERY_PARSE_FAILED,
            f"query could not be parsed after schema validation ({last_invalid_reason})",
            retry_count=self.llm_config.max_retries,
            latency_ms=latency_ms,
        )


def parse_query(
    query: str,
    client: LocalLLMClient,
    *,
    llm_config: LLMConfig,
    prompt_config: QueryPromptConfig,
) -> ParsedQuery:
    return QueryParser(
        client,
        llm_config=llm_config,
        prompt_config=prompt_config,
    ).parse(query)


__all__ = ["QueryParser", "parse_query", "structured_intent_response_schema"]
