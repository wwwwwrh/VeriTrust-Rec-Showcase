"""Strict schemas for Evidence Documents and Chunks."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from data_pipeline.resource_dataset.schema import SourceType


EvidenceSourceType = Literal[
    "github_readme",
    "repo_metadata",
]
TextFormat = Literal["markdown", "structured_metadata"]
ExtractionMethod = Literal["raw_snapshot", "api_metadata_serialization"]

HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
RAW_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
EVIDENCE_ID_RE = re.compile(r"^ev_[0-9a-f]{16}$")
CHUNK_ID_RE = re.compile(r"^ch_[0-9a-f]{16}$")
SOURCE_ID_RE = re.compile(r"^github:\d+:(readme|metadata)$")
EVIDENCE_BUILD_ID_RE = re.compile(r"^evidence_build_[0-9a-f]{16}$")


def _valid_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("source_url must be an absolute HTTP(S) URL")
    return value


def _valid_timestamp(value: str | None) -> str | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    return value


class EvidenceDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    document_key: str
    evidence_build_id: str
    resource_id: str
    evidence_source_type: EvidenceSourceType
    source_type: SourceType
    source_id: str
    raw_source_id: str
    source_url: str
    raw_source_hash: str
    title: str
    text: str
    text_format: TextFormat
    content_hash: str
    retrieved_at: str
    published_at: str | None = None
    updated_at: str | None = None
    source_reliability: float = Field(ge=0, le=1)
    source_reliability_method: Literal["heuristic_v0"]
    extraction_method: ExtractionMethod
    run_id: str
    collector_version: str
    snapshot_schema_version: str
    evidence_schema_version: str
    normalizer_version: str
    markdown_parser_version: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("evidence_id")
    @classmethod
    def validate_evidence_id(cls, value: str) -> str:
        if not EVIDENCE_ID_RE.fullmatch(value):
            raise ValueError("invalid evidence_id")
        return value

    @field_validator("evidence_build_id")
    @classmethod
    def validate_build_id(cls, value: str) -> str:
        if not EVIDENCE_BUILD_ID_RE.fullmatch(value):
            raise ValueError("invalid evidence_build_id")
        return value

    @field_validator("document_key", "source_id")
    @classmethod
    def validate_source_id(cls, value: str) -> str:
        if not SOURCE_ID_RE.fullmatch(value):
            raise ValueError("invalid evidence source identity")
        return value

    @field_validator("resource_id")
    @classmethod
    def validate_resource_id(cls, value: str) -> str:
        if not value.startswith("github:") or not value.removeprefix("github:").isdigit():
            raise ValueError("invalid canonical resource_id")
        return value

    @field_validator("source_url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        return _valid_url(value)

    @field_validator("raw_source_hash")
    @classmethod
    def validate_raw_hash(cls, value: str) -> str:
        if not RAW_HASH_RE.fullmatch(value):
            raise ValueError("invalid raw_source_hash")
        return value

    @field_validator("content_hash")
    @classmethod
    def validate_content_hash(cls, value: str) -> str:
        if not HASH_RE.fullmatch(value):
            raise ValueError("invalid content_hash")
        return value

    @field_validator("text")
    @classmethod
    def validate_nonblank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("document text cannot be blank")
        return value

    @field_validator("retrieved_at", "published_at", "updated_at")
    @classmethod
    def validate_timestamp(cls, value: str | None) -> str | None:
        return _valid_timestamp(value)

    @model_validator(mode="after")
    def validate_identity_and_content(self) -> "EvidenceDocument":
        if self.source_id != self.document_key:
            raise ValueError("source_id must equal document_key")
        suffix = "readme" if self.evidence_source_type == "github_readme" else "metadata"
        if self.source_id != f"{self.resource_id}:{suffix}":
            raise ValueError("source_id must be derived from resource_id and evidence type")
        expected_format = "markdown" if suffix == "readme" else "structured_metadata"
        expected_method = "raw_snapshot" if suffix == "readme" else "api_metadata_serialization"
        if self.text_format != expected_format or self.extraction_method != expected_method:
            raise ValueError("text format or extraction method conflicts with evidence type")
        if not self.text:
            raise ValueError("document text cannot be empty")
        provenance_keys = {
            "raw_snapshot_path", "raw_provenance", "resource_discovery_provenance",
            "resource_record_version", "normalization_warnings",
        }
        if not provenance_keys.issubset(self.metadata):
            raise ValueError("document metadata is missing provenance fields")
        return self


class EvidenceChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    evidence_id: str
    evidence_build_id: str
    resource_id: str
    evidence_source_type: EvidenceSourceType
    source_type: SourceType
    source_id: str
    raw_source_id: str
    source_url: str
    raw_source_hash: str
    section_path: list[str]
    section_title: str
    section_level: int = Field(ge=0, le=6)
    chunk_index: int = Field(ge=0)
    source_start_line: int = Field(ge=1)
    source_end_line: int = Field(ge=1)
    source_start_char: int = Field(ge=0)
    source_end_char: int = Field(gt=0)
    text: str
    char_count: int = Field(gt=0)
    content_hash: str
    contains_code: bool
    contains_table: bool
    retrieved_at: str
    source_reliability: float = Field(ge=0, le=1)
    source_reliability_method: Literal["heuristic_v0"]
    run_id: str
    collector_version: str
    snapshot_schema_version: str
    evidence_schema_version: str
    normalizer_version: str
    markdown_parser_version: str
    chunker_version: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("chunk_id")
    @classmethod
    def validate_chunk_id(cls, value: str) -> str:
        if not CHUNK_ID_RE.fullmatch(value):
            raise ValueError("invalid chunk_id")
        return value

    @field_validator("evidence_build_id")
    @classmethod
    def validate_build_id(cls, value: str) -> str:
        if not EVIDENCE_BUILD_ID_RE.fullmatch(value):
            raise ValueError("invalid evidence_build_id")
        return value

    @field_validator("evidence_id")
    @classmethod
    def validate_evidence_id(cls, value: str) -> str:
        if not EVIDENCE_ID_RE.fullmatch(value):
            raise ValueError("invalid evidence_id")
        return value

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, value: str) -> str:
        if not SOURCE_ID_RE.fullmatch(value):
            raise ValueError("invalid evidence source identity")
        return value

    @field_validator("source_url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        return _valid_url(value)

    @field_validator("raw_source_hash")
    @classmethod
    def validate_raw_hash(cls, value: str) -> str:
        if not RAW_HASH_RE.fullmatch(value):
            raise ValueError("invalid raw_source_hash")
        return value

    @field_validator("content_hash")
    @classmethod
    def validate_content_hash(cls, value: str) -> str:
        if not HASH_RE.fullmatch(value):
            raise ValueError("invalid content_hash")
        return value

    @field_validator("text")
    @classmethod
    def validate_nonblank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("chunk text cannot be blank")
        return value

    @field_validator("retrieved_at")
    @classmethod
    def validate_timestamp(cls, value: str) -> str:
        return _valid_timestamp(value)  # type: ignore[return-value]

    @model_validator(mode="after")
    def validate_span_and_identity(self) -> "EvidenceChunk":
        if self.source_end_char <= self.source_start_char:
            raise ValueError("chunk character span must be non-empty")
        if self.source_end_line < self.source_start_line:
            raise ValueError("chunk line span is invalid")
        if self.char_count != len(self.text):
            raise ValueError("char_count must equal len(text)")
        suffix = "readme" if self.evidence_source_type == "github_readme" else "metadata"
        if self.source_id != f"{self.resource_id}:{suffix}":
            raise ValueError("source_id must be derived from resource_id and evidence type")
        return self
