"""Local-only LLM interface shared by S2/S3 and its offline test double."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LocalLLMClient(Protocol):
    backend: str
    model: str

    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        ...

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
        temperature: float = 0.0,
        max_tokens: int = 512,
        context_length: int = 4096,
        thinking: bool = False,
    ) -> str:
        ...

    def generate_structured_with_metadata(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
        temperature: float = 0.0,
        max_tokens: int = 512,
        context_length: int = 4096,
        thinking: bool = False,
    ) -> "LLMGenerationResponse":
        ...


@dataclass(frozen=True)
class LLMGenerationResponse:
    text: str
    prompt_eval_tokens: int | None = None
    output_eval_tokens: int | None = None


@dataclass
class MockLLMClient:
    """Deterministic response queue used only by offline tests."""

    responses: list[str | BaseException]
    calls: list[dict[str, object]] = field(default_factory=list)
    backend: str = "mock"
    model: str = "mock"

    def _next_response(self) -> str:
        if not self.responses:
            raise RuntimeError("MockLLMClient response queue is empty")
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    def generate(self, prompt: str, *, temperature: float = 0.0, max_tokens: int = 512) -> str:
        self.calls.append({"prompt": prompt, "temperature": temperature, "max_tokens": max_tokens})
        return self._next_response()

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
        temperature: float = 0.0,
        max_tokens: int = 512,
        context_length: int = 4096,
        thinking: bool = False,
    ) -> str:
        self.calls.append({
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "schema": schema,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "context_length": context_length,
            "thinking": thinking,
        })
        return self._next_response()

    def generate_structured_with_metadata(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
        temperature: float = 0.0,
        max_tokens: int = 512,
        context_length: int = 4096,
        thinking: bool = False,
    ) -> LLMGenerationResponse:
        return LLMGenerationResponse(text=self.generate_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=schema,
            temperature=temperature,
            max_tokens=max_tokens,
            context_length=context_length,
            thinking=thinking,
        ))


__all__ = ["LLMGenerationResponse", "LocalLLMClient", "MockLLMClient"]
