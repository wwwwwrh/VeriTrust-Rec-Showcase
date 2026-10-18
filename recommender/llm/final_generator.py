"""Local structured Final Generation with grounding and safe fallback."""

from __future__ import annotations

from dataclasses import dataclass
import copy
import json
from time import perf_counter

from pydantic import ValidationError

from recommender.config import FinalPromptConfig, GenerationConfig
from recommender.errors import GenerationErrorCode, QueryErrorCode, QueryProcessingError
from recommender.generation.builder import build_generation_view
from recommender.generation.grounding import (
    hydrate_final_explanation,
    validate_final_output,
)
from recommender.generation.template import build_template_explanation
from recommender.schemas import (
    FinalLLMOutput,
    FinalRecommendationExplanation,
    GenerationAudit,
    GroundedResultPack,
    GroundingValidationResult,
    OutputLanguage,
)

from .client import LLMGenerationResponse, LocalLLMClient


@dataclass(frozen=True)
class FinalGenerationOutcome:
    explanation: FinalRecommendationExplanation | None
    audit: GenerationAudit
    validation: GroundingValidationResult
    warnings: list[str]


def _error_code(exc: QueryProcessingError) -> GenerationErrorCode:
    if exc.code == QueryErrorCode.LOCAL_LLM_TIMEOUT:
        return GenerationErrorCode.FINAL_LLM_TIMEOUT
    if exc.code in {
        QueryErrorCode.LOCAL_LLM_UNAVAILABLE,
        QueryErrorCode.LOCAL_LLM_NOT_CONFIGURED,
    }:
        return GenerationErrorCode.FINAL_LLM_UNAVAILABLE
    return GenerationErrorCode.FINAL_GENERATION_INVALID


def _response(
    client: LocalLLMClient,
    *,
    system_prompt: str,
    user_prompt: str,
    response_schema: dict,
    config: GenerationConfig,
) -> LLMGenerationResponse:
    method = getattr(client, "generate_structured_with_metadata", None)
    if callable(method):
        return method(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=response_schema,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            context_length=config.context_length,
            thinking=config.thinking,
        )
    return LLMGenerationResponse(text=client.generate_structured(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        schema=response_schema,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        context_length=config.context_length,
        thinking=config.thinking,
    ))


def final_response_schema(pack: GroundedResultPack) -> dict:
    """Constrain obvious ID/label errors before the grounding validator."""

    schema = copy.deepcopy(FinalLLMOutput.model_json_schema())
    base_item_schema = schema["$defs"]["FinalLLMRecommendationItem"]
    item_schemas: dict[str, dict] = {}
    for recommendation in pack.recommendations:
        item_schema = copy.deepcopy(base_item_schema)
        item_schemas[recommendation.resource_id] = item_schema
    schema["properties"]["items"] = {
        "type": "object",
        "properties": item_schemas,
        "required": list(item_schemas),
        "additionalProperties": False,
    }
    return schema


def _fallback(
    pack: GroundedResultPack,
    *,
    output_language: OutputLanguage,
    client: LocalLLMClient,
    prompt: FinalPromptConfig,
    started: float,
    retry_count: int,
    input_chars: int,
    prompt_tokens: int | None,
    output_tokens: int | None,
    error_code: GenerationErrorCode | None,
    warnings: list[str],
) -> FinalGenerationOutcome:
    try:
        explanation = build_template_explanation(pack, output_language=output_language)
        status = "FALLBACK"
        grounding_valid = True
    except Exception as exc:  # pragma: no cover - defensive boundary
        explanation = None
        status = "FAILED"
        grounding_valid = False
        warnings.append(f"FINAL_GENERATION_FAILED: {type(exc).__name__}: {exc}")
        error_code = error_code or GenerationErrorCode.FINAL_GENERATION_INVALID
    return FinalGenerationOutcome(
        explanation=explanation,
        audit=GenerationAudit(
            backend=client.backend,
            model=client.model,
            prompt_version=prompt.prompt_version,
            generation_status=status,
            retry_count=retry_count,
            latency_ms=round((perf_counter() - started) * 1000, 3),
            grounding_valid=grounding_valid,
            fallback_used=explanation is not None,
            resource_count=len(pack.recommendations),
            input_character_count=input_chars,
            prompt_eval_tokens=prompt_tokens,
            output_eval_tokens=output_tokens,
            error_code=error_code.value if error_code else None,
        ),
        validation=GroundingValidationResult(
            is_valid=grounding_valid,
            errors=[] if grounding_valid else ["deterministic template construction failed"],
            validated_resource_ids=(
                [item.resource_id for item in pack.recommendations]
                if grounding_valid else []
            ),
            fallback_required=not grounding_valid,
        ),
        warnings=warnings,
    )


def generate_final_explanation(
    pack: GroundedResultPack,
    client: LocalLLMClient,
    *,
    output_language: OutputLanguage,
    config: GenerationConfig,
    prompt: FinalPromptConfig,
) -> FinalGenerationOutcome:
    started = perf_counter()
    warnings: list[str] = []
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    retry_count = 0
    try:
        view = build_generation_view(
            pack, output_language=output_language, config=config,
        )
        payload = json.dumps(
            view.model_dump(mode="json"), ensure_ascii=False,
            sort_keys=True, separators=(",", ":"),
        )
    except Exception as exc:
        warnings.append(f"FINAL_GENERATION_INVALID: {type(exc).__name__}: {exc}")
        return _fallback(
            pack,
            output_language=output_language,
            client=client,
            prompt=prompt,
            started=started,
            retry_count=0,
            input_chars=0,
            prompt_tokens=None,
            output_tokens=None,
            error_code=GenerationErrorCode.FINAL_GENERATION_INVALID,
            warnings=warnings,
        )
    input_chars = len(payload)
    if not config.enabled:
        return _fallback(
            pack,
            output_language=output_language,
            client=client,
            prompt=prompt,
            started=started,
            retry_count=0,
            input_chars=input_chars,
            prompt_tokens=None,
            output_tokens=None,
            error_code=None,
            warnings=["Final LLM generation is disabled; deterministic template used."],
        )

    last_code = GenerationErrorCode.FINAL_GENERATION_INVALID
    last_validation = GroundingValidationResult(
        is_valid=False,
        errors=["Final Generation did not produce a validated response"],
        fallback_required=True,
    )
    for attempt in range(config.max_retries + 1):
        retry_count = attempt
        system_prompt = prompt.system_prompt
        if attempt:
            correction = "; ".join(last_validation.errors[:5])
            system_prompt = (
                f"{system_prompt}\n\n{prompt.retry_instruction}\n"
                f"Validation errors: {correction}"
            )
        try:
            generated = _response(
                client,
                system_prompt=system_prompt,
                user_prompt=payload,
                response_schema=final_response_schema(pack),
                config=config,
            )
            prompt_tokens = generated.prompt_eval_tokens
            output_tokens = generated.output_eval_tokens
        except QueryProcessingError as exc:
            last_code = _error_code(exc)
            warnings.append(f"{last_code.value}: {exc}")
            break
        except Exception as exc:
            last_code = GenerationErrorCode.FINAL_LLM_UNAVAILABLE
            warnings.append(f"{last_code.value}: {type(exc).__name__}: {exc}")
            break

        try:
            output = FinalLLMOutput.model_validate(json.loads(generated.text))
        except (json.JSONDecodeError, ValidationError, TypeError) as exc:
            last_code = GenerationErrorCode.FINAL_GENERATION_INVALID
            last_validation = GroundingValidationResult(
                is_valid=False,
                errors=[f"schema validation failed: {type(exc).__name__}"],
                fallback_required=True,
            )
            if attempt < config.max_retries:
                continue
            warnings.append(last_validation.errors[0])
            break

        last_validation = validate_final_output(output, pack)
        if not last_validation.is_valid:
            last_code = GenerationErrorCode.FINAL_GROUNDING_FAILED
            if attempt < config.max_retries:
                continue
            warnings.extend(last_validation.errors)
            break
        explanation = hydrate_final_explanation(
            output,
            pack,
            output_language=output_language,
            generation_method="qwen_grounded_v1",
        )
        return FinalGenerationOutcome(
            explanation=explanation,
            audit=GenerationAudit(
                backend=client.backend,
                model=client.model,
                prompt_version=prompt.prompt_version,
                generation_status="SUCCESS",
                retry_count=attempt,
                latency_ms=round((perf_counter() - started) * 1000, 3),
                grounding_valid=True,
                fallback_used=False,
                resource_count=len(pack.recommendations),
                input_character_count=input_chars,
                prompt_eval_tokens=prompt_tokens,
                output_eval_tokens=output_tokens,
            ),
            validation=last_validation,
            warnings=warnings,
        )

    return _fallback(
        pack,
        output_language=output_language,
        client=client,
        prompt=prompt,
        started=started,
        retry_count=retry_count,
        input_chars=input_chars,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        error_code=last_code,
        warnings=warnings,
    )


__all__ = [
    "FinalGenerationOutcome", "final_response_schema", "generate_final_explanation",
]
