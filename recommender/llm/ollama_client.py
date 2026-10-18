"""Localhost-only Ollama transport shared by Summer S2 and S3."""

from __future__ import annotations

from threading import Lock
from typing import Any

import httpx

from recommender.config import LLMConfig
from recommender.errors import QueryErrorCode, QueryProcessingError

from .client import LLMGenerationResponse


class OllamaLocalLLMClient:
    """Small HTTP client that never falls back to a cloud model or endpoint."""

    backend = "ollama"

    def __init__(
        self,
        config: LLMConfig,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.config = config
        self.model = config.model
        self._http = httpx.Client(
            base_url=config.base_url,
            timeout=config.timeout_seconds,
            transport=transport,
            headers={"Content-Type": "application/json"},
            trust_env=False,
        )
        self._model_validated = False
        self._model_validation_lock = Lock()

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "OllamaLocalLLMClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = self._http.request(method, path, **kwargs)
        except httpx.TimeoutException as exc:
            raise QueryProcessingError(
                QueryErrorCode.LOCAL_LLM_TIMEOUT, "local Ollama request timed out"
            ) from exc
        except httpx.HTTPError as exc:
            raise QueryProcessingError(
                QueryErrorCode.LOCAL_LLM_UNAVAILABLE, "local Ollama is unavailable"
            ) from exc
        if response.status_code == 404:
            raise QueryProcessingError(
                QueryErrorCode.LOCAL_LLM_NOT_CONFIGURED,
                f"configured local model is unavailable: {self.model}",
            )
        if not response.is_success:
            raise QueryProcessingError(
                QueryErrorCode.LOCAL_LLM_UNAVAILABLE,
                f"local Ollama returned HTTP {response.status_code}",
            )
        return response

    def _validate_local_model_once(self) -> None:
        if self._model_validated:
            return
        with self._model_validation_lock:
            if self._model_validated:
                return
            response = self._request("GET", "/api/tags")
            try:
                payload = response.json()
                models = payload["models"]
                matched = next(
                    (
                        item for item in models
                        if self.model in {item.get("name"), item.get("model")}
                        and isinstance(item.get("size"), int)
                        and item["size"] > 0
                    ),
                    None,
                )
            except (AttributeError, TypeError, KeyError, ValueError) as exc:
                raise QueryProcessingError(
                    QueryErrorCode.LOCAL_LLM_RESPONSE_INVALID,
                    "local Ollama model list response is invalid",
                ) from exc
            if matched is None:
                raise QueryProcessingError(
                    QueryErrorCode.LOCAL_LLM_NOT_CONFIGURED,
                    f"configured model is not installed locally: {self.model}",
                )
            self._model_validated = True

    def _chat_with_metadata(self, payload: dict[str, Any]) -> LLMGenerationResponse:
        self._validate_local_model_once()
        response = self._request("POST", "/api/chat", json=payload)
        try:
            body = response.json()
            content = body["message"]["content"]
        except (TypeError, KeyError, ValueError) as exc:
            raise QueryProcessingError(
                QueryErrorCode.LOCAL_LLM_RESPONSE_INVALID,
                "local Ollama chat response is invalid",
            ) from exc
        if not isinstance(content, str) or not content.strip():
            raise QueryProcessingError(
                QueryErrorCode.LOCAL_LLM_RESPONSE_INVALID,
                "local Ollama returned empty structured content",
            )
        prompt_tokens = body.get("prompt_eval_count")
        output_tokens = body.get("eval_count")
        return LLMGenerationResponse(
            text=content,
            prompt_eval_tokens=prompt_tokens if isinstance(prompt_tokens, int) else None,
            output_eval_tokens=output_tokens if isinstance(output_tokens, int) else None,
        )

    def _chat(self, payload: dict[str, Any]) -> str:
        return self._chat_with_metadata(payload).text

    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        return self._chat({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": False,
            "options": {
                "temperature": temperature,
                "num_ctx": self.config.context_length,
                "num_predict": max_tokens,
            },
        })

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
        return self._chat({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "think": thinking,
            "format": schema,
            "options": {
                "temperature": temperature,
                "num_ctx": context_length,
                "num_predict": max_tokens,
            },
        })

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
        return self._chat_with_metadata({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "think": thinking,
            "format": schema,
            "options": {
                "temperature": temperature,
                "num_ctx": context_length,
                "num_predict": max_tokens,
            },
        })


__all__ = ["OllamaLocalLLMClient"]
