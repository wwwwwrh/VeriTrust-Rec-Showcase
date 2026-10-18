"""In-memory KG built only from canonical structured Resource fields."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from typing import Any

from data_pipeline.resource_dataset.schema import Resource

from recommender.schemas import Claim, KGFact


HASH_RE = re.compile(r"^[0-9a-f]{64}$")


def _key(value: str) -> str:
    return " ".join(value.casefold().replace("_", " ").replace("-", " ").split())


def _stable_id(parts: list[str]) -> str:
    payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    return "kg_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class KnowledgeGraph:
    def __init__(self, facts: list[KGFact]) -> None:
        self.facts = sorted(facts, key=lambda item: item.fact_id)
        self._by_resource: dict[str, list[KGFact]] = defaultdict(list)
        for fact in self.facts:
            self._by_resource[fact.resource_id].append(fact)

    @staticmethod
    def _provenance_index(records: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
        index: dict[tuple[str, str], dict[str, Any]] = {}
        for record in records:
            resource_id = record.get("resource_id")
            field = record.get("field")
            if isinstance(resource_id, str) and isinstance(field, str):
                index[(resource_id, field)] = record
        return index

    @classmethod
    def build(cls, resources: list[Resource], provenance_records: list[dict[str, Any]]) -> "KnowledgeGraph":
        provenance = cls._provenance_index(provenance_records)
        facts: list[KGFact] = []

        def add(resource: Resource, field: str, predicate: str, object_type: str, value: str) -> None:
            record = provenance.get((resource.resource_id, field))
            if record is None:
                raise ValueError(f"missing Phase 2 provenance for {resource.resource_id}.{field}")
            source_hash = record.get("source_hash")
            source_url = record.get("source_url")
            if not isinstance(source_hash, str) or not HASH_RE.fullmatch(source_hash):
                raise ValueError(f"invalid Phase 2 source hash for {resource.resource_id}.{field}")
            if not isinstance(source_url, str):
                raise ValueError(f"missing Phase 2 source URL for {resource.resource_id}.{field}")
            reference = f"resource_provenance:{resource.resource_id}:{field}:{source_hash}"
            facts.append(KGFact(
                fact_id=_stable_id([resource.resource_id, predicate, object_type, value, reference]),
                resource_id=resource.resource_id,
                predicate=predicate,  # type: ignore[arg-type]
                object_type=object_type,  # type: ignore[arg-type]
                object=value,
                source_reference=reference,
                source_url=source_url,
                source_hash=source_hash,
            ))

        for resource in sorted(resources, key=lambda item: item.resource_id):
            for topic in resource.topics:
                add(resource, "topics", "covers", "Topic", topic)
            for framework in resource.frameworks:
                add(resource, "frameworks", "uses", "Framework", framework)
            if resource.language:
                add(resource, "language", "written_in", "Language", resource.language)
            if resource.owner_name:
                add(resource, "owner_name", "owned_by", "Owner", resource.owner_name)
        return cls(facts)

    def facts_for_resource(self, resource_id: str) -> list[KGFact]:
        return list(self._by_resource.get(resource_id, []))

    def has_fact(self, resource_id: str, predicate: str, value: str) -> bool:
        wanted = _key(value)
        return any(
            fact.predicate == predicate and _key(fact.object) == wanted
            for fact in self._by_resource.get(resource_id, [])
        )

    def supports_claim(self, claim: Claim) -> bool:
        predicate = {"covers_topic": "covers", "uses_framework": "uses"}.get(claim.claim_type)
        return bool(predicate and self.has_fact(claim.resource_id, predicate, claim.object))


__all__ = ["KnowledgeGraph"]
