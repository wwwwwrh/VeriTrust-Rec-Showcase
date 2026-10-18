"""Deterministic, dependency-free TF-IDF Resource retrieval."""

from __future__ import annotations

import math
import re
from collections import Counter

from data_pipeline.resource_dataset.schema import Resource

from recommender.data import searchable_resource_text
from recommender.schemas import Candidate, StructuredIntent


TOKEN_RE = re.compile(r"[A-Za-z0-9_+#.-]{2,}|[\u4e00-\u9fff]{2,}")


def tokenize(text: str) -> list[str]:
    return [item.casefold().strip(".-") for item in TOKEN_RE.findall(text) if item.strip(".-")]


def intent_to_query(intent: StructuredIntent) -> str:
    parts: list[str] = []
    for values in (
        intent.topics, intent.skills, intent.prerequisites,
        intent.framework_preferences,
    ):
        parts.extend(values)
    if intent.difficulty != "unknown":
        parts.append(intent.difficulty)
    parts.extend(intent.learning_style)
    if intent.time_budget:
        parts.append(intent.time_budget)
    parts.extend(intent.constraints)
    return " ".join(parts)


class TfidfRetriever:
    def __init__(self, resources: list[Resource]) -> None:
        self.resources = sorted(resources, key=lambda item: item.resource_id)
        self._idf: dict[str, float] = {}
        self._doc_vectors: list[dict[str, float]] = []
        self._build_index()

    def _build_index(self) -> None:
        tokenized = [tokenize(searchable_resource_text(item)) for item in self.resources]
        document_frequency: Counter[str] = Counter()
        for tokens in tokenized:
            document_frequency.update(set(tokens))
        count = max(1, len(tokenized))
        self._idf = {
            term: math.log((1 + count) / (1 + frequency)) + 1.0
            for term, frequency in document_frequency.items()
        }
        for tokens in tokenized:
            frequency = Counter(tokens)
            total = max(1, sum(frequency.values()))
            self._doc_vectors.append({
                term: (value / total) * self._idf[term]
                for term, value in frequency.items()
            })

    @staticmethod
    def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
        shared = set(left) & set(right)
        if not shared:
            return 0.0
        numerator = sum(left[item] * right[item] for item in shared)
        left_norm = math.sqrt(sum(value * value for value in left.values()))
        right_norm = math.sqrt(sum(value * value for value in right.values()))
        return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0

    def retrieve(self, intent: StructuredIntent, *, top_k: int = 5) -> list[Candidate]:
        tokens = tokenize(intent_to_query(intent))
        if not tokens:
            return []
        frequency = Counter(tokens)
        total = max(1, sum(frequency.values()))
        query_vector = {
            term: (value / total) * self._idf.get(term, 0.0)
            for term, value in frequency.items()
        }
        scored: list[tuple[float, Resource]] = []
        for resource, vector in zip(self.resources, self._doc_vectors, strict=True):
            score = self._cosine(query_vector, vector)
            if score > 0:
                scored.append((score, resource))
        scored.sort(key=lambda item: (-item[0], item[1].resource_id))
        return [
            Candidate(
                resource_id=resource.resource_id,
                title=resource.name or resource.full_name or resource.resource_id,
                url=resource.url,
                retrieval_score=round(min(1.0, score), 6),
            )
            for score, resource in scored[:top_k]
        ]


__all__ = ["TfidfRetriever", "intent_to_query", "tokenize"]
