"""Configuration loading for the Summer Edition core."""

from __future__ import annotations

from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .errors import GlobalErrorCode, PipelineError


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PathsConfig(ConfigModel):
    resources: str
    resource_manifest: str
    resource_provenance: str
    evidence_documents: str
    evidence_chunks: str
    evidence_manifest: str


class CompatibilityConfig(ConfigModel):
    evidence_schema_versions: list[str]
    normalizer_versions: list[str]
    markdown_parser_versions: list[str]
    chunker_versions: list[str]
    effective_config_hashes: list[str] = Field(default_factory=list)


class RetrievalConfig(ConfigModel):
    top_k: int = Field(default=5, ge=1)


class EvidenceConfig(ConfigModel):
    top_k_per_claim: int = Field(default=3, ge=1)
    metadata_priority_bonus: float = Field(default=0.25, ge=0, le=1)


class VerificationConfig(ConfigModel):
    maintenance_max_age_days: int = Field(default=730, ge=1)
    supported_metadata_score: float = Field(default=0.95, ge=0, le=1)
    supported_readme_score: float = Field(default=0.85, ge=0, le=1)
    contradicted_score: float = Field(default=0.95, ge=0, le=1)
    unknown_related_score: float = Field(default=0.25, ge=0, le=1)
    open_source_spdx_ids: list[str]


class ConfidenceConfig(ConfigModel):
    w_retrieval: float = Field(ge=0, le=1)
    w_coverage: float = Field(ge=0, le=1)
    w_verification: float = Field(ge=0, le=1)
    w_kg_support: float = Field(ge=0, le=1)
    w_source_reliability: float = Field(ge=0, le=1)
    high_threshold: float = Field(default=0.8, ge=0, le=1)
    low_threshold: float = Field(default=0.5, ge=0, le=1)

    @model_validator(mode="after")
    def validate_weights(self) -> "ConfidenceConfig":
        total = sum((self.w_retrieval, self.w_coverage, self.w_verification,
                     self.w_kg_support, self.w_source_reliability))
        if abs(total - 1.0) > 1e-9:
            raise ValueError("confidence weights must sum to 1")
        if self.low_threshold > self.high_threshold:
            raise ValueError("low_threshold cannot exceed high_threshold")
        return self


class RankingConfig(ConfigModel):
    w_relevance: float = Field(ge=0, le=1)
    w_confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_weights(self) -> "RankingConfig":
        if abs(self.w_relevance + self.w_confidence - 1.0) > 1e-9:
            raise ValueError("ranking weights must sum to 1")
        return self


class AbstentionConfig(ConfigModel):
    confidence_threshold: float = Field(default=0.5, ge=0, le=1)
    abstain_on_contradiction: bool = True


class LoggingConfig(ConfigModel):
    save_sessions: bool = True
    output_path: str = "logs/summer/sessions.jsonl"


class LLMConfig(ConfigModel):
    backend: Literal["ollama"] = "ollama"
    model: str = "qwen3:8b"
    base_url: str = "http://localhost:11434"
    timeout_seconds: float = Field(default=60.0, gt=0)
    temperature: float = Field(default=0.0, ge=0, le=2)
    context_length: int = Field(default=4096, ge=512)
    max_tokens: int = Field(default=512, ge=32)
    thinking: Literal[False] = False
    max_retries: int = Field(default=1, ge=0, le=1)
    prompt_path: str = "configs/prompts.yaml"

    @model_validator(mode="after")
    def validate_local_runtime(self) -> "LLMConfig":
        parsed = urlparse(self.base_url)
        if parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("llm.base_url must be a local HTTP loopback endpoint")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("llm.base_url cannot contain credentials, query, or fragment")
        if not self.model.strip():
            raise ValueError("llm.model cannot be blank")
        self.base_url = self.base_url.rstrip("/")
        self.model = self.model.strip()
        return self


class QueryPromptConfig(ConfigModel):
    prompt_version: str
    system_prompt: str
    retry_instruction: str


class FinalPromptConfig(ConfigModel):
    prompt_version: str
    system_prompt: str
    retry_instruction: str


class PromptsConfig(ConfigModel):
    query_understanding: QueryPromptConfig
    final_generation: FinalPromptConfig


class GenerationConfig(ConfigModel):
    enabled: bool = True
    temperature: float = Field(default=0.2, ge=0, le=2)
    context_length: int = Field(default=4096, ge=512)
    max_tokens: int = Field(default=768, ge=128)
    timeout_seconds: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=1, ge=0, le=1)
    thinking: Literal[False] = False
    max_claims_per_resource: int = Field(default=8, ge=1)
    max_evidence_excerpts_per_claim: int = Field(default=2, ge=0)
    claim_text_chars: int = Field(default=280, ge=80)
    evidence_excerpt_chars: int = Field(default=320, ge=80)
    max_input_chars: int = Field(default=7500, ge=2000)


class UIConfig(ConfigModel):
    default_language: Literal["zh", "en"] = "zh"
    default_top_k: int = Field(default=5, ge=1)
    min_top_k: int = Field(default=3, ge=1)
    max_top_k: int = Field(default=10, ge=1)
    debug: bool = False
    evidence_excerpt_chars: int = Field(default=500, ge=100, le=2000)

    @model_validator(mode="after")
    def validate_top_k_range(self) -> "UIConfig":
        if self.min_top_k > self.max_top_k:
            raise ValueError("ui.min_top_k cannot exceed ui.max_top_k")
        if not self.min_top_k <= self.default_top_k <= self.max_top_k:
            raise ValueError("ui.default_top_k must be inside the configured range")
        return self


class SummerConfig(ConfigModel):
    config_version: str
    paths: PathsConfig
    evidence_compatibility: CompatibilityConfig
    retrieval: RetrievalConfig
    evidence: EvidenceConfig
    verification: VerificationConfig
    confidence: ConfidenceConfig
    ranking: RankingConfig
    abstention: AbstentionConfig
    logging: LoggingConfig
    llm: LLMConfig = Field(default_factory=LLMConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    ui: UIConfig = Field(default_factory=UIConfig)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_project_path(value: str, *, root: Path | None = None) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root or project_root()) / path


def load_config(path: Path | None = None) -> SummerConfig:
    target = path or project_root() / "configs" / "summer.yaml"
    try:
        raw = yaml.safe_load(target.read_text(encoding="utf-8"))
        return SummerConfig.model_validate(raw)
    except Exception as exc:
        raise PipelineError(GlobalErrorCode.CONFIG_INVALID, f"invalid Summer config: {exc}") from exc


def load_prompts(path: Path) -> PromptsConfig:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return PromptsConfig.model_validate(raw)
    except Exception as exc:
        raise ValueError(f"invalid prompt config: {exc}") from exc


__all__ = [
    "FinalPromptConfig", "GenerationConfig", "LLMConfig", "PromptsConfig",
    "QueryPromptConfig", "SummerConfig", "UIConfig", "load_config",
    "load_prompts", "project_root", "resolve_project_path",
]
