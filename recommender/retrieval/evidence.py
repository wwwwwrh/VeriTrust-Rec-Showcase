"""Claim-specific lexical retrieval over canonical Phase 3 EvidenceChunks."""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict

from data_pipeline.evidence.schema import EvidenceChunk

from recommender.schemas import Claim, RetrievedEvidence

from .retriever import tokenize


PREDICATE_ALIASES: dict[str, tuple[str, ...]] = {
    "uses_framework": ("uses", "using", "based on", "built with", "requires", "install", "dependency"),
    "covers_topic": ("covers", "topic", "tutorial", "guide", "course", "implements", "implementation"),
    "is_maintained": ("maintained", "active", "updated", "pushed_at", "is_archived", "deprecated"),
    "has_license": ("license", "license_spdx_id", "open source", "spdx", "copyright"),
}


def normalized_phrase(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value).casefold().replace("_", " ").replace("-", " ")
    return " ".join(normalized.split())


class EvidenceRetriever:
    def __init__(self, chunks: list[EvidenceChunk], *, metadata_priority_bonus: float = 0.25) -> None:
        self.metadata_priority_bonus = metadata_priority_bonus
        self.by_resource: dict[str, list[EvidenceChunk]] = defaultdict(list)
        self.chunk_by_id: dict[str, EvidenceChunk] = {}
        for chunk in sorted(chunks, key=lambda item: (item.resource_id, item.chunk_id)):
            if chunk.chunk_id in self.chunk_by_id:
                raise ValueError(f"duplicate Evidence chunk ID: {chunk.chunk_id}")
            self.by_resource[chunk.resource_id].append(chunk)
            self.chunk_by_id[chunk.chunk_id] = chunk

    @staticmethod
    def _score(claim: Claim, chunk: EvidenceChunk, metadata_priority_bonus: float) -> float:
        text = normalized_phrase(chunk.text)
        text_tokens = set(tokenize(text))
        object_tokens = set(tokenize(normalized_phrase(claim.object)))
        aliases = PREDICATE_ALIASES[claim.claim_type]
        alias_tokens = set(token for alias in aliases for token in tokenize(normalized_phrase(alias)))

        object_coverage = (
            len(object_tokens & text_tokens) / len(object_tokens)
            if object_tokens else 0.0
        )
        alias_coverage = (
            len(alias_tokens & text_tokens) / len(alias_tokens)
            if alias_tokens else 0.0
        )
        object_phrase = normalized_phrase(claim.object)
        exact_object = bool(object_phrase and object_phrase in text)
        section_tokens = set(tokenize(chunk.section_title or ""))
        section_overlap = bool((object_tokens | alias_tokens) & section_tokens)
        score = (
            0.55 * object_coverage
            + 0.25 * alias_coverage
            + 0.15 * float(exact_object)
            + 0.05 * float(section_overlap)
        )
        if claim.claim_type in {"is_maintained", "has_license"} and chunk.evidence_source_type == "repo_metadata":
            score += metadata_priority_bonus
        elif claim.claim_type == "covers_topic" and chunk.evidence_source_type == "repo_metadata":
            score += metadata_priority_bonus / 2
        return min(1.0, score)

    @staticmethod
    def _to_runtime(chunk: EvidenceChunk, score: float) -> RetrievedEvidence:
        return RetrievedEvidence(
            chunk_id=chunk.chunk_id,
            evidence_id=chunk.evidence_id,
            resource_id=chunk.resource_id,
            source_id=chunk.source_id,
            text=chunk.text,
            source_url=chunk.source_url,
            evidence_source_type=chunk.evidence_source_type,
            source_type=chunk.source_type,
            source_reliability=chunk.source_reliability,
            source_reliability_method=chunk.source_reliability_method,
            retrieval_score=round(score, 6),
            section_title=chunk.section_title or None,
        )

    def retrieve(self, claim: Claim, *, top_k: int = 3) -> list[RetrievedEvidence]:
        scored = [
            (self._score(claim, chunk, self.metadata_priority_bonus), chunk)
            for chunk in self.by_resource.get(claim.resource_id, [])
        ]
        scored = [item for item in scored if item[0] > 0]
        scored.sort(key=lambda item: (-item[0], item[1].chunk_id))
        return [self._to_runtime(chunk, score) for score, chunk in scored[:top_k]]


__all__ = ["EvidenceRetriever", "normalized_phrase"]
