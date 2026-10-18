"""Best-effort, non-critical session audit logging."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from recommender.schemas import (
    QueryParseAudit,
    SummerPipelineResult,
    SummerQueryResult,
    UserFacingResult,
)


def write_session_log(
    result: SummerPipelineResult,
    path: Path,
    *,
    query_audit: QueryParseAudit | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "session_id": "summer_" + uuid.uuid4().hex[:16],
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": result.status,
        "intent": result.intent.model_dump(mode="json") if result.intent else None,
        "ranked_resource_ids": result.ranked_resource_ids,
        "error_code": result.error_code,
        "warnings": result.warnings,
        "query_understanding": query_audit.model_dump(mode="json") if query_audit else None,
        "candidates": [
            {
                "resource_id": assessment.candidate.resource_id,
                "claim_ids": [item.claim_id for item in assessment.claims],
                "evidence_chunk_ids": sorted({
                    item.chunk_id
                    for values in assessment.evidence_by_claim.values()
                    for item in values
                }),
                "verification_labels": [item.label for item in assessment.verifications],
                "confidence": assessment.confidence.confidence,
                "abstain": assessment.decision.abstain,
                "errors": assessment.errors,
            }
            for assessment in result.assessments
        ],
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def write_query_failure_log(result: SummerQueryResult, path: Path) -> None:
    """Persist audit-safe query failure metadata without the raw user query."""

    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "session_id": "summer_" + uuid.uuid4().hex[:16],
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": result.status,
        "query_understanding": result.query_audit.model_dump(mode="json"),
        "error_code": result.error_code,
        "error_message": result.error_message,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def write_user_session_log(result: UserFacingResult, path: Path) -> None:
    """Persist one audit-safe S3 record without raw queries or model output."""

    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "session_id": "summer_" + uuid.uuid4().hex[:16],
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": result.status,
        "query_understanding": result.query_audit.model_dump(mode="json"),
        "generation": result.generation_audit.model_dump(mode="json"),
        "pipeline_trace": result.pipeline_trace.model_dump(mode="json"),
        "ranked_resource_ids": [
            item.candidate.resource_id for item in result.recommendations
        ],
        "error_code": result.error_code,
        "warnings": result.warnings,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(
            record, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ) + "\n")


__all__ = ["write_query_failure_log", "write_session_log", "write_user_session_log"]
