"""Deterministic proposal of resource-bound claims before generation."""

from __future__ import annotations

import hashlib
import json

from data_pipeline.resource_dataset.schema import Resource

from recommender.kg.graph import KnowledgeGraph
from recommender.schemas import Candidate, Claim, StructuredIntent


GENERATION_METHOD = "rule_v1"


def claim_id_for(
    resource_id: str,
    claim_type: str,
    subject: str,
    predicate: str,
    object_value: str,
    generation_method: str = GENERATION_METHOD,
) -> str:
    material = [resource_id, claim_type, subject, predicate, object_value, generation_method]
    payload = json.dumps(material, ensure_ascii=False, separators=(",", ":"))
    return "claim_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _claim(resource: Resource, candidate: Candidate, claim_type: str, predicate: str, obj: str, text: str) -> Claim:
    subject = resource.resource_id
    return Claim(
        claim_id=claim_id_for(resource.resource_id, claim_type, subject, predicate, obj),
        resource_id=resource.resource_id,
        claim_type=claim_type,  # type: ignore[arg-type]
        subject=subject,
        predicate=predicate,
        object=obj,
        text=f"{candidate.title} {text}",
    )


def generate_claims(
    candidate: Candidate,
    intent: StructuredIntent,
    resource: Resource,
    kg: KnowledgeGraph,
) -> list[Claim]:
    if candidate.resource_id != resource.resource_id:
        raise ValueError("candidate and Resource identity mismatch")
    claims: list[Claim] = []
    for framework in intent.framework_preferences:
        claims.append(_claim(resource, candidate, "uses_framework", "uses", framework, f"uses {framework}."))
    for topic in intent.topics:
        claims.append(_claim(resource, candidate, "covers_topic", "covers", topic, f"covers {topic}."))
    if resource.pushed_at is not None or resource.updated_at is not None or resource.is_archived:
        claims.append(_claim(
            resource, candidate, "is_maintained", "is", "actively_maintained", "is actively maintained.",
        ))
    if resource.license and resource.license.casefold() not in {"unknown", "noassertion", "other"}:
        claims.append(_claim(
            resource, candidate, "has_license", "has", "open_source_license",
            "has an explicit open-source license.",
        ))

    # KG facts are inputs to proposal/verification, not authority to invent new claim types.
    _ = kg
    unique = {item.claim_id: item for item in claims}
    return [unique[key] for key in sorted(unique)]


__all__ = ["GENERATION_METHOD", "claim_id_for", "generate_claims"]
