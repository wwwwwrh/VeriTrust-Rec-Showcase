"""Canonical Resource Dataset schema."""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator


ResourceType = Literal["github_project", "official_tutorial", "official_documentation", "university_course"]
SourceType = Literal["official_project", "official_documentation", "university", "major_education_platform", "community", "unknown"]
Difficulty = Literal["beginner", "intermediate", "advanced", "unknown"]


class Resource(BaseModel):
    resource_id: str
    resource_type: ResourceType
    name: str
    canonical_name: str
    full_name: str | None = None
    description: str | None = None
    url: str
    canonical_url: str
    source_platform: str
    source_id: str
    source_type: SourceType
    source_reliability: float = Field(ge=0, le=1)
    source_reliability_method: Literal["heuristic_v0"]
    owner_name: str | None = None
    owner_type: str | None = None
    organization_name: str | None = None
    language: str | None = None
    topics: list[str] = Field(default_factory=list)
    frameworks: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    prerequisites: list[str] = Field(default_factory=list)
    difficulty: Difficulty = "unknown"
    learning_style: list[str] = Field(default_factory=list)
    tasks: list[str] = Field(default_factory=list)
    stars: int | None = Field(default=None, ge=0)
    forks: int | None = Field(default=None, ge=0)
    watchers: int | None = Field(default=None, ge=0)
    open_issues: int | None = Field(default=None, ge=0)
    license: str = "unknown"
    is_archived: bool = False
    is_fork: bool = False
    created_at: str | None = None
    updated_at: str | None = None
    pushed_at: str | None = None
    activity_score: float | None = Field(default=None, ge=0, le=1)
    freshness_score: float | None = Field(default=None, ge=0, le=1)
    metadata_completeness: float = Field(ge=0, le=1)
    domain_relevance_score: float = Field(ge=0, le=1)
    data_quality_score: float = Field(ge=0, le=1)
    raw_source_id: str
    retrieved_at: str
    record_version: str

    @field_validator("resource_id")
    @classmethod
    def validate_resource_id(cls, value: str) -> str:
        if not value.startswith("github:") or not value.removeprefix("github:").isdigit():
            raise ValueError("resource_id must use the stable github:<numeric_id> format")
        return value

    @field_validator("url", "canonical_url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("URL must be absolute HTTP(S)")
        return value

    @field_validator("created_at", "updated_at", "pushed_at", "retrieved_at")
    @classmethod
    def validate_timestamp(cls, value: str | None) -> str | None:
        if value is None:
            return None
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        return value
