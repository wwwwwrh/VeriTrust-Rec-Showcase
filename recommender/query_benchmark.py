"""Run the manual-review Summer S2 query benchmark against local Ollama."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean

from recommender.config import load_config, load_prompts, project_root, resolve_project_path
from recommender.errors import QueryProcessingError
from recommender.llm import OllamaLocalLLMClient, parse_query


DEFAULT_INPUT = Path("tests/fixtures/query_understanding/benchmark.jsonl")


def _nearest_rank_p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run qwen3:8b queries and emit predictions for human semantic review"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()

    root = project_root()
    config = load_config(args.config) if args.config else load_config()
    prompts = load_prompts(resolve_project_path(config.llm.prompt_path, root=root))
    cases = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows: list[dict] = []
    with OllamaLocalLLMClient(config.llm) as client:
        for index, case in enumerate(cases):
            try:
                parsed = parse_query(
                    case["query"],
                    client,
                    llm_config=config.llm,
                    prompt_config=prompts.query_understanding,
                )
                rows.append({
                    "case_id": case["case_id"],
                    "query": case["query"],
                    "expected": case["expected"],
                    "prediction": parsed.intent.model_dump(mode="json"),
                    "schema_valid": True,
                    "difficulty_exact": (
                        parsed.intent.difficulty == case["expected"]["difficulty"]
                    ),
                    "latency_ms": parsed.audit.latency_ms,
                    "run_kind": "cold_start" if index == 0 else "warm",
                    "human_field_review": None,
                    "unsupported_inference_count": None,
                    "negative_constraint_correct": None,
                })
            except QueryProcessingError as exc:
                rows.append({
                    "case_id": case["case_id"],
                    "query": case["query"],
                    "expected": case["expected"],
                    "prediction": None,
                    "schema_valid": False,
                    "difficulty_exact": False,
                    "latency_ms": exc.latency_ms,
                    "run_kind": "cold_start" if index == 0 else "warm",
                    "error_code": exc.code.value,
                    "human_field_review": None,
                    "unsupported_inference_count": None,
                    "negative_constraint_correct": None,
                })

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )

    valid = [row for row in rows if row["schema_valid"]]
    warm = [row["latency_ms"] for row in valid if row["run_kind"] == "warm"]
    summary = {
        "model": config.llm.model,
        "prompt_version": prompts.query_understanding.prompt_version,
        "case_count": len(rows),
        "schema_valid_rate": len(valid) / len(rows) if rows else 0,
        "difficulty_exact_rate": (
            sum(bool(row["difficulty_exact"]) for row in rows) / len(rows) if rows else 0
        ),
        "cold_start_latency_ms": rows[0]["latency_ms"] if rows else None,
        "warm_average_latency_ms": mean(warm) if warm else None,
        "warm_p95_latency_ms": _nearest_rank_p95(warm) if len(warm) >= 10 else None,
        "semantic_metrics_status": "requires_human_review",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if len(valid) == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
