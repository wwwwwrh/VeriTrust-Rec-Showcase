"""Run the Summer S3 grounded-generation benchmark against local Ollama.

The command is read-only with respect to canonical project data. Raw per-query
results are written only when an explicit ``--output`` path is supplied.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from recommender.config import load_config, load_prompts, project_root, resolve_project_path
from recommender.data import (
    load_evidence_corpus,
    load_resource_provenance,
    load_resources,
)
from recommender.llm import OllamaLocalLLMClient
from recommender.pipeline import run_user_pipeline
from recommender.schemas import UserFacingResult


DEFAULT_INPUT = Path("tests/fixtures/query_understanding/benchmark.jsonl")


def nearest_rank_p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def _load_cases(path: Path, limit: int | None) -> list[dict[str, Any]]:
    cases = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return cases if limit is None else cases[:limit]


def _grounding_counts(result: UserFacingResult) -> dict[str, int | bool]:
    """Recheck the user-facing output against backend-owned recommendations."""

    recommendation_ids = [item.candidate.resource_id for item in result.recommendations]
    explanation = result.final_explanation
    if explanation is None:
        return {
            # No generated explanation means no generated order to mutate.
            "ranking_preserved": True,
            "hallucinated_resources": 0,
            "hallucinated_claims": 0,
            "hallucinated_evidence_ids": 0,
        }

    explanation_ids = [item.resource_id for item in explanation.items]
    claim_ids_by_resource = {
        item.candidate.resource_id: {claim.claim.claim_id for claim in item.claims}
        for item in result.recommendations
    }
    evidence_ids_by_claim = {
        claim.claim.claim_id: {
            evidence.evidence.chunk_id for evidence in claim.evidence
            if evidence.used_for_verification
        }
        for item in result.recommendations
        for claim in item.claims
    }
    hallucinated_claims = 0
    hallucinated_evidence = 0
    for item in explanation.items:
        known_claims = claim_ids_by_resource.get(item.resource_id, set())
        for fact in (
            item.verified_facts + item.uncertainty_notes + item.contradicted_notes
        ):
            if fact.claim_id not in known_claims:
                hallucinated_claims += 1
            allowed_evidence = evidence_ids_by_claim.get(fact.claim_id, set())
            hallucinated_evidence += sum(
                chunk_id not in allowed_evidence for chunk_id in fact.evidence_chunk_ids
            )
    return {
        "ranking_preserved": recommendation_ids == explanation_ids,
        "hallucinated_resources": len(set(explanation_ids) - set(recommendation_ids)),
        "hallucinated_claims": hallucinated_claims,
        "hallucinated_evidence_ids": hallucinated_evidence,
    }


def summarize(rows: list[dict[str, Any]], *, model: str, prompt_version: str) -> dict[str, Any]:
    generation_rows = [row for row in rows if row["generation_attempted"]]
    successful = [row for row in generation_rows if row["generation_status"] == "SUCCESS"]
    generation_latencies = [row["generation_latency_ms"] for row in generation_rows]
    total_latencies = [row["total_latency_ms"] for row in rows]
    prompt_tokens = [
        row["prompt_eval_tokens"] for row in generation_rows
        if row["prompt_eval_tokens"] is not None
    ]
    output_tokens = [
        row["output_eval_tokens"] for row in generation_rows
        if row["output_eval_tokens"] is not None
    ]
    return {
        "model": model,
        "prompt_version": prompt_version,
        "case_count": len(rows),
        "query_statuses": dict(sorted(Counter(row["status"] for row in rows).items())),
        "generation_attempted_count": len(generation_rows),
        "final_json_valid_rate": (
            len(successful) / len(generation_rows) if generation_rows else None
        ),
        "grounding_pass_rate": (
            sum(row["llm_grounding_passed"] for row in generation_rows)
            / len(generation_rows) if generation_rows else None
        ),
        "hallucinated_resource_count": sum(row["hallucinated_resources"] for row in rows),
        "hallucinated_claim_count": sum(row["hallucinated_claims"] for row in rows),
        "hallucinated_evidence_id_count": sum(
            row["hallucinated_evidence_ids"] for row in rows
        ),
        "ranking_mutation_count": sum(not row["ranking_preserved"] for row in rows),
        "fallback_count": sum(row["generation_status"] == "FALLBACK" for row in rows),
        "average_query_latency_ms": mean(row["query_latency_ms"] for row in rows) if rows else None,
        "average_core_latency_ms": mean(row["core_latency_ms"] for row in rows) if rows else None,
        "average_final_generation_latency_ms": (
            mean(generation_latencies) if generation_latencies else None
        ),
        "p95_final_generation_latency_ms": nearest_rank_p95(generation_latencies),
        "average_total_pipeline_latency_ms": mean(total_latencies) if total_latencies else None,
        "p95_total_pipeline_latency_ms": nearest_rank_p95(total_latencies),
        "max_generation_input_characters": max(
            (row["generation_input_characters"] for row in generation_rows), default=None
        ),
        "average_generation_prompt_tokens": mean(prompt_tokens) if prompt_tokens else None,
        "max_generation_prompt_tokens": max(prompt_tokens, default=None),
        "average_generation_output_tokens": mean(output_tokens) if output_tokens else None,
        "max_generation_output_tokens": max(output_tokens, default=None),
        "language_quality_status": "requires_human_review",
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the local S3 grounded Final Generation benchmark"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--language", choices=("zh", "en"), default="zh")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    root = project_root()
    loaded_config = load_config()
    config = loaded_config.model_copy(update={
        "logging": loaded_config.logging.model_copy(update={"save_sessions": False}),
    })
    prompts = load_prompts(resolve_project_path(config.llm.prompt_path, root=root))
    resources = load_resources(resolve_project_path(config.paths.resources, root=root))
    provenance = load_resource_provenance(
        resolve_project_path(config.paths.resource_provenance, root=root)
    )
    evidence = load_evidence_corpus(
        documents_path=resolve_project_path(config.paths.evidence_documents, root=root),
        chunks_path=resolve_project_path(config.paths.evidence_chunks, root=root),
        evidence_manifest_path=resolve_project_path(config.paths.evidence_manifest, root=root),
        resource_manifest_path=resolve_project_path(config.paths.resource_manifest, root=root),
        resources=resources,
        compatibility=config.evidence_compatibility,
    )
    cases = _load_cases(args.input, args.limit)
    rows: list[dict[str, Any]] = []
    with (
        OllamaLocalLLMClient(config.llm) as query_client,
        OllamaLocalLLMClient(config.llm.model_copy(update={
            "timeout_seconds": config.generation.timeout_seconds,
        })) as final_client,
    ):
        for case in cases:
            result = run_user_pipeline(
                case["query"],
                output_language=args.language,
                query_client=query_client,
                final_client=final_client,
                config=config,
                prompts=prompts,
                resources=resources,
                provenance_records=provenance,
                evidence_corpus=evidence,
                root=root,
            )
            checks = _grounding_counts(result)
            rows.append({
                "case_id": case["case_id"],
                "status": result.status,
                "error_code": result.error_code,
                "recommendation_count": len(result.recommendations),
                "generation_attempted": result.status == "OK",
                "generation_status": result.generation_audit.generation_status,
                "generation_error_code": result.generation_audit.error_code,
                "llm_grounding_passed": (
                    result.generation_audit.generation_status == "SUCCESS"
                    and result.generation_audit.grounding_valid
                ),
                "query_latency_ms": result.pipeline_trace.query_latency_ms,
                "core_latency_ms": result.pipeline_trace.core_latency_ms,
                "generation_latency_ms": result.pipeline_trace.generation_latency_ms,
                "total_latency_ms": result.pipeline_trace.total_latency_ms,
                "generation_input_characters": result.generation_audit.input_character_count,
                "prompt_eval_tokens": result.generation_audit.prompt_eval_tokens,
                "output_eval_tokens": result.generation_audit.output_eval_tokens,
                **checks,
                "language_quality_review": None,
            })

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
    summary = summarize(
        rows, model=config.llm.model,
        prompt_version=prompts.final_generation.prompt_version,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    safety_passed = (
        summary["hallucinated_resource_count"] == 0
        and summary["hallucinated_claim_count"] == 0
        and summary["hallucinated_evidence_id_count"] == 0
        and summary["ranking_mutation_count"] == 0
    )
    return 0 if safety_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
