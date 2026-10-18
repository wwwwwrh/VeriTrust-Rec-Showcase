"""Command-line entry point for the Summer S2/S3 local pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from recommender.config import load_config
from recommender.pipeline import recommend_from_query, run_user_pipeline


def _human_output(result: object) -> str:
    data = result.model_dump(mode="json")  # type: ignore[attr-defined]
    lines = [
        f"status: {data['status']}",
        f"model: {data['query_audit']['model']}",
        f"prompt_version: {data['query_audit']['prompt_version']}",
        f"parse_latency_ms: {data['query_audit']['latency_ms']}",
    ]
    parsed = data.get("parsed_query")
    if parsed:
        lines.append("structured_intent:")
        lines.append(json.dumps(parsed["intent"], ensure_ascii=False, indent=2))
    recommendation = data.get("recommendation")
    if recommendation:
        lines.extend((
            f"pipeline_status: {recommendation['status']}",
            "ranked_resource_ids: " + json.dumps(
                recommendation["ranked_resource_ids"], ensure_ascii=False
            ),
        ))
    if data.get("error_code"):
        lines.append(f"error_code: {data['error_code']}")
        lines.append(f"error_message: {data['error_message']}")
    return "\n".join(lines)


def _final_human_output(result: object) -> str:
    data = result.model_dump(mode="json")  # type: ignore[attr-defined]
    lines = [
        f"status: {data['status']}",
        f"model: {data['query_audit']['model']}",
        f"prompt_version: {data['query_audit']['prompt_version']}",
        f"generation_status: {data['generation_audit']['generation_status']}",
        f"generation_prompt_version: {data['generation_audit']['prompt_version']}",
    ]
    if data.get("intent"):
        lines.append("structured_intent:")
        lines.append(json.dumps(data["intent"], ensure_ascii=False, indent=2))
    explanation = data.get("final_explanation")
    if explanation:
        lines.append("final_explanation:")
        lines.append(json.dumps(explanation, ensure_ascii=False, indent=2))
    if data.get("recommendations"):
        lines.append("ranked_resource_ids: " + json.dumps([
            item["candidate"]["resource_id"] for item in data["recommendations"]
        ]))
    if data.get("error_code"):
        lines.append(f"error_code: {data['error_code']}")
        lines.append(f"error_message: {data['error_message']}")
    return "\n".join(lines)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Run local Query Understanding, the S1 core, and optional S3 generation"
    )
    parser.add_argument("query", help="Natural-language learning request")
    parser.add_argument("--config", type=Path, help="Optional Summer YAML config")
    parser.add_argument("--json", action="store_true", help="Print the complete result as JSON")
    parser.add_argument("--no-log", action="store_true", help="Disable session audit logging")
    parser.add_argument(
        "--final", action="store_true",
        help="Run S3 grounded Final Generation after the trustworthy core",
    )
    parser.add_argument(
        "--language", choices=("zh", "en"), default="zh",
        help="Final explanation language (only used with --final)",
    )
    parser.add_argument(
        "--top-k", type=int,
        help="Per-request candidate Top-K override (only used with --final)",
    )
    args = parser.parse_args()

    config = load_config(args.config) if args.config else load_config()
    if args.no_log:
        config = config.model_copy(update={
            "logging": config.logging.model_copy(update={"save_sessions": False})
        })
    result = (
        run_user_pipeline(
            args.query,
            config=config,
            output_language=args.language,
            top_k=args.top_k,
        )
        if args.final else recommend_from_query(args.query, config=config)
    )
    if args.json:
        print(result.model_dump_json(indent=2))
    else:
        print(_final_human_output(result) if args.final else _human_output(result))
    if result.status in {"OK", "RECOMMENDATION_ABSTAINED"}:
        return 0
    if result.status == "QUERY_ERROR":
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
