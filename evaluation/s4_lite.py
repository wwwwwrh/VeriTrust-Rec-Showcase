"""CLI for the frozen S4-Lite v1.0.0 evaluation.

Every command requires an explicit frozen data root. The evaluator never
falls back to canonical paths and never mutates the Resource/Evidence corpus.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

from recommender.config import load_prompts, project_root
from recommender.final_benchmark import _grounding_counts
from recommender.llm import OllamaLocalLLMClient
from recommender.pipeline import run_user_pipeline

from .core import (
    BENCHMARK_STATUS,
    EVALUATION_REFERENCE_TIME,
    build_experiment_identity,
    evaluate_abstention,
    evaluate_retrieval,
    evaluate_verification,
    load_abstention_benchmark,
    load_and_validate_assets,
    load_retrieval_benchmark,
    load_verification_benchmark,
    prepare_retrieval_candidates,
    prepare_verification_candidates,
    render_retrieval_sheet,
    render_verification_blind_sheet,
    validate_retrieval_references,
    validate_verification_candidate,
    validate_verification_gold,
    write_json,
    write_jsonl,
)


DEFAULT_RETRIEVAL = Path("evaluation/benchmarks/s4_lite_retrieval.yaml")
DEFAULT_VERIFICATION = Path("evaluation/benchmarks/s4_lite_verification.jsonl")
DEFAULT_ABSTENTION = Path("evaluation/benchmarks/s4_lite_abstention.yaml")
DEFAULT_RESULTS = Path("evaluation/results")

RUNTIME_QUERY = "我会 Python，想学习 PyTorch 深度学习，希望有代码实践。"
EXPLAINABILITY_QUERIES = {
    "pytorch_deep_learning": RUNTIME_QUERY,
    "llm_rag": "我想学习大语言模型、Transformer 和 RAG，希望有代码示例。",
}


def _resolve(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


def _add_common(child: argparse.ArgumentParser) -> None:
    child.add_argument(
        "--data-root", type=Path, required=True,
        help="Explicit frozen data root; no implicit canonical fallback is allowed",
    )
    child.add_argument("--retrieval-benchmark", type=Path, default=DEFAULT_RETRIEVAL)
    child.add_argument("--verification-benchmark", type=Path, default=DEFAULT_VERIFICATION)
    child.add_argument("--abstention-benchmark", type=Path, default=DEFAULT_ABSTENTION)
    child.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the frozen S4-Lite evaluation")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in (
        "preflight", "prepare-retrieval", "prepare-verification", "evaluate", "runtime",
    ):
        child = subparsers.add_parser(command)
        _add_common(child)
    runtime = subparsers.choices["runtime"]
    runtime.add_argument("--warm-runs", type=int, default=5)
    runtime.add_argument("--output-language", choices=("zh", "en"), default="zh")
    return parser


def _load_inputs(args: argparse.Namespace, root: Path) -> tuple[Any, Any, Any, Any, Any]:
    data_root = _resolve(root, args.data_root)
    retrieval = load_retrieval_benchmark(_resolve(root, args.retrieval_benchmark))
    verification = load_verification_benchmark(_resolve(root, args.verification_benchmark))
    abstention = load_abstention_benchmark(_resolve(root, args.abstention_benchmark))
    assets = load_and_validate_assets(data_root, project_root=root)
    validate_retrieval_references(
        retrieval, {item.resource_id for item in assets.resources}, require_confirmed=True,
    )
    validate_verification_gold(verification, assets)
    identity = build_experiment_identity(
        project_root=root,
        benchmark=retrieval,
        frozen=assets.frozen_identity,
        verification_cases=[item.model_dump(mode="json") for item in verification],
        abstention_benchmark=abstention,
    )
    return retrieval, verification, abstention, assets, identity


def _latency_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    names = ("query_latency_ms", "core_latency_ms", "generation_latency_ms", "total_latency_ms")
    return {
        name.removesuffix("_latency_ms"): {
            "average_ms": round(mean(row[name] for row in rows), 3),
            "median_ms": round(median(row[name] for row in rows), 3),
        }
        for name in names
    }


def _runtime_row(index: int, result: Any) -> dict[str, Any]:
    checks = _grounding_counts(result)
    return {
        "run_index": index,
        "status": result.status,
        "error_code": result.error_code,
        "ranked_resource_ids": [item.candidate.resource_id for item in result.recommendations],
        "generation_status": result.generation_audit.generation_status,
        "grounding_valid": result.generation_audit.grounding_valid,
        "query_latency_ms": result.pipeline_trace.query_latency_ms,
        "core_latency_ms": result.pipeline_trace.core_latency_ms,
        "generation_latency_ms": result.pipeline_trace.generation_latency_ms,
        "total_latency_ms": result.pipeline_trace.total_latency_ms,
        **checks,
    }


def _run_runtime(
    *, assets: Any, root: Path, results_dir: Path, warm_runs: int, output_language: str,
) -> dict[str, Any]:
    if warm_runs != 5:
        raise ValueError("S4-Lite v1.0.0 runtime protocol requires exactly five warm runs")
    prompts = load_prompts(root / assets.config.llm.prompt_path)
    at = datetime.fromisoformat(EVALUATION_REFERENCE_TIME.replace("Z", "+00:00"))
    rows: list[dict[str, Any]] = []
    explainability: dict[str, Any] = {}
    final_config = assets.config.llm.model_copy(update={
        "timeout_seconds": assets.config.generation.timeout_seconds,
    })
    with (
        OllamaLocalLLMClient(assets.config.llm) as query_client,
        OllamaLocalLLMClient(final_config) as final_client,
    ):
        for index in range(1, warm_runs + 2):
            result = run_user_pipeline(
                RUNTIME_QUERY,
                output_language=output_language,
                query_client=query_client,
                final_client=final_client,
                config=assets.config,
                prompts=prompts,
                resources=assets.resources,
                provenance_records=assets.provenance_records,
                evidence_corpus=assets.evidence_corpus,
                evaluation_time=at,
                root=root,
            )
            rows.append(_runtime_row(index, result))
            if index == 1:
                explainability["pytorch_deep_learning"] = result.model_dump(mode="json")

        llm_rag = run_user_pipeline(
            EXPLAINABILITY_QUERIES["llm_rag"],
            output_language=output_language,
            query_client=query_client,
            final_client=final_client,
            config=assets.config,
            prompts=prompts,
            resources=assets.resources,
            provenance_records=assets.provenance_records,
            evidence_corpus=assets.evidence_corpus,
            evaluation_time=at,
            root=root,
        )
        explainability["llm_rag"] = llm_rag.model_dump(mode="json")

    cold = rows[0]
    warm = rows[1:]
    safety = {
        "ranking_mutation_count": sum(not row["ranking_preserved"] for row in rows),
        "hallucinated_resource_count": sum(row["hallucinated_resources"] for row in rows),
        "hallucinated_claim_count": sum(row["hallucinated_claims"] for row in rows),
        "hallucinated_evidence_id_count": sum(row["hallucinated_evidence_ids"] for row in rows),
    }
    output = {
        "protocol": "one_cold_five_warm",
        "model": assets.config.llm.model,
        "query": RUNTIME_QUERY,
        "cold": cold,
        "warm_run_count": len(warm),
        "warm_latency": _latency_summary(warm),
        "safety": safety,
        "all_runs_successful": all(
            row["status"] == "OK" and row["generation_status"] == "SUCCESS"
            for row in rows
        ),
        "runs": rows,
        "explainability_cases": {
            key: {
                "status": value["status"],
                "ranked_resource_ids": [
                    item["candidate"]["resource_id"] for item in value["recommendations"]
                ],
                "generation_status": value["generation_audit"]["generation_status"],
            }
            for key, value in explainability.items()
        },
    }
    write_json(results_dir / "s4_lite_runtime.json", output)
    write_json(results_dir / "s4_lite_explainability.json", explainability)
    return output


def run(args: argparse.Namespace, *, root: Path) -> dict[str, object]:
    retrieval, verification, abstention, assets, identity = _load_inputs(args, root)
    results_dir = _resolve(root, args.results_dir)
    write_json(results_dir / "s4_lite_identity.json", identity.model_dump(mode="json"))
    summary: dict[str, object] = {
        "command": args.command,
        "benchmark_status": BENCHMARK_STATUS,
        "frozen_identity": assets.frozen_identity.model_dump(mode="json"),
        "experiment_identity": identity.model_dump(mode="json"),
    }
    if args.command == "prepare-retrieval":
        cases = prepare_retrieval_candidates(retrieval, assets.resources)
        (results_dir / "retrieval_annotation_sheet.md").parent.mkdir(parents=True, exist_ok=True)
        (results_dir / "retrieval_annotation_sheet.md").write_text(
            render_retrieval_sheet(cases), encoding="utf-8",
        )
        write_json(results_dir / "retrieval_candidate_audit.json", {
            "benchmark_status": BENCHMARK_STATUS, "cases": cases,
        })
        summary["retrieval_case_count"] = len(cases)
    elif args.command == "prepare-verification":
        cases = prepare_verification_candidates(retrieval, assets)
        canonical_chunks = {item.chunk_id: item for item in assets.evidence_corpus.chunks}
        for case in cases:
            validate_verification_candidate(case, canonical_chunks)
        (results_dir / "verification_blind_annotation_sheet.md").parent.mkdir(parents=True, exist_ok=True)
        (results_dir / "verification_blind_annotation_sheet.md").write_text(
            render_verification_blind_sheet(cases), encoding="utf-8",
        )
        write_jsonl(results_dir / "verification_candidate_audit.jsonl", cases)
        summary["verification_candidate_count"] = len(cases)
    elif args.command == "evaluate":
        metrics = {
            "retrieval": evaluate_retrieval(retrieval, assets.resources),
            "verification": evaluate_verification(verification, assets),
            "abstention": evaluate_abstention(abstention, assets),
        }
        write_json(results_dir / "s4_lite_metrics.json", metrics)
        summary["metrics"] = metrics
    elif args.command == "runtime":
        summary["runtime"] = _run_runtime(
            assets=assets,
            root=root,
            results_dir=results_dir,
            warm_runs=args.warm_runs,
            output_language=args.output_language,
        )
    return summary


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = run(args, root=project_root())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
