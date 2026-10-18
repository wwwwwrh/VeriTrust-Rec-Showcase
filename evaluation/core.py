"""Core utilities for the read-only S4-Lite evaluation snapshot.

This module deliberately reuses the accepted Summer S1--S3 interfaces. It
does not modify ranking, verification, confidence, abstention, or canonical
data. Formal metrics require the frozen human-confirmed v1.0.0 benchmarks.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from data_pipeline.resource_dataset.schema import Resource
from recommender.config import SummerConfig, load_config, load_prompts
from recommender.data import (
    EvidenceCorpus,
    load_evidence_corpus,
    load_resource_provenance,
    load_resources,
)
from recommender.pipeline import run_core_pipeline
from recommender.retrieval.retriever import TfidfRetriever
from recommender.schemas import Claim, RetrievedEvidence, StructuredIntent, VerificationLabel
from recommender.verification.claim_extractor import claim_id_for
from recommender.verification.verifier import verify_claim


EVALUATOR_VERSION = "s4_lite_evaluator_v1.0.0"
BENCHMARK_STATUS = "human_confirmed"
EVALUATION_REFERENCE_TIME = "2026-09-16T00:00:00Z"

EXPECTED_TAG = "summer-s3.5-resource-200"
EXPECTED_TAG_TARGET = "693ab9b63fcccbf351a2590583d3e934833a3c18"
EXPECTED_RAW_RUN_ID = "a2aa04a6225e406faeb28cfcfe5d6c29"
EXPECTED_CURATION_ID = "summer_s35_stage_c_final_v1"
EXPECTED_CURATION_VERSION = "1.0.0"
EXPECTED_CURATION_HASH = (
    "sha256:0346f405c22dc339e0d3abfc043ae05df6a4373b43969bae11428c2def9ac9cb"
)
EXPECTED_RESOURCE_BUILD_ID = "build_60c4f8a08dad8614"
EXPECTED_EVIDENCE_BUILD_ID = "evidence_build_10deed019a096c1a"
EXPECTED_RESOURCE_COUNT = 200
EXPECTED_DOCUMENT_COUNT = 400
EXPECTED_CHUNK_COUNT = 5902
EXPECTED_EXCLUSIONS = {
    "github:132750724",
    "github:21737465",
    "github:266417772",
    "github:39464018",
    "github:60493101",
}


class EvaluationModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RetrievalGoldCase(EvaluationModel):
    case_id: str
    category: str
    query: str
    gold_intent: StructuredIntent
    metadata_match_terms: list[str]
    acceptable_resource_ids: list[str] = Field(default_factory=list)
    annotation_status: Literal["human_confirmed"]
    annotation_note: str

    @field_validator("metadata_match_terms")
    @classmethod
    def validate_terms(cls, value: list[str]) -> list[str]:
        if not value or any(not item.strip() for item in value):
            raise ValueError("metadata_match_terms must be non-empty")
        return value


class RetrievalBenchmark(EvaluationModel):
    benchmark_version: Literal["1.0.0"]
    benchmark_status: Literal["human_confirmed"]
    evaluation_reference_time: str
    retrieval_top_k: int = Field(ge=3)
    candidate_pool: dict[str, int]
    cases: list[RetrievalGoldCase]

    @model_validator(mode="after")
    def validate_cases(self) -> "RetrievalBenchmark":
        ids = [item.case_id for item in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate retrieval case_id")
        if len(self.cases) != 10:
            raise ValueError("S4-Lite draft requires exactly 10 retrieval cases")
        if any(not item.acceptable_resource_ids for item in self.cases):
            raise ValueError("confirmed benchmark requires a non-empty acceptable set")
        for item in self.cases:
            if len(item.acceptable_resource_ids) != len(set(item.acceptable_resource_ids)):
                raise ValueError(f"duplicate acceptable Resource ID in {item.case_id}")
        if self.evaluation_reference_time != EVALUATION_REFERENCE_TIME:
            raise ValueError("unexpected evaluation reference time")
        return self


class VerificationGoldCase(EvaluationModel):
    benchmark_version: Literal["1.0.0"]
    case_id: str
    source_candidate_id: str
    source_query_case: str
    evaluation_reference_time: Literal["2026-09-16T00:00:00Z"]
    resource_id: str
    repository: str
    claim: Claim
    evidence: list[RetrievedEvidence]
    gold_label: VerificationLabel
    human_rationale: str
    annotation_status: Literal["human_confirmed"]

    @model_validator(mode="after")
    def validate_binding(self) -> "VerificationGoldCase":
        if self.claim.resource_id != self.resource_id:
            raise ValueError("Verification Gold Claim is bound to a different Resource")
        expected_claim_id = claim_id_for(
            self.claim.resource_id,
            self.claim.claim_type,
            self.claim.subject,
            self.claim.predicate,
            self.claim.object,
            self.claim.generation_method,
        )
        if self.claim.claim_id != expected_claim_id:
            raise ValueError("Verification Gold claim_id is not the canonical Claim identity")
        if not self.evidence:
            raise ValueError("Verification Gold requires at least one Evidence record")
        if any(item.resource_id != self.resource_id for item in self.evidence):
            raise ValueError("Verification Gold contains cross-resource Evidence")
        chunk_ids = [item.chunk_id for item in self.evidence]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("Verification Gold contains duplicate Evidence chunks")
        if not self.human_rationale.strip():
            raise ValueError("Verification Gold requires a human rationale")
        return self


class AbstentionCase(EvaluationModel):
    case_id: str
    kind: Literal["full_corpus", "controlled_resource_subset"]
    description: str
    intent: StructuredIntent
    resource_id: str | None = None
    expected_status: Literal["OK", "ABSTAINED"]
    expected_error_code: str | None = None

    @model_validator(mode="after")
    def validate_controlled_case(self) -> "AbstentionCase":
        if self.kind == "controlled_resource_subset" and not self.resource_id:
            raise ValueError("controlled abstention case requires resource_id")
        if self.kind == "full_corpus" and self.resource_id is not None:
            raise ValueError("full-corpus abstention case cannot specify resource_id")
        return self


class AbstentionBenchmark(EvaluationModel):
    benchmark_version: Literal["1.0.0"]
    cases: list[AbstentionCase]

    @model_validator(mode="after")
    def validate_cases(self) -> "AbstentionBenchmark":
        if len(self.cases) != 3:
            raise ValueError("S4-Lite requires exactly three abstention cases")
        ids = [item.case_id for item in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate abstention case_id")
        return self


class FrozenIdentity(EvaluationModel):
    git_tag: str
    git_tag_target: str
    raw_run_id: str
    curation_manifest_id: str
    curation_manifest_version: str
    curation_manifest_hash: str
    resource_build_id: str
    evidence_build_id: str
    resource_count: int
    document_count: int
    chunk_count: int
    excluded_resources_absent: bool
    orphan_documents: int
    orphan_chunks: int
    identity_matches_promoted_canonical: bool
    data_root: str


class ExperimentIdentity(EvaluationModel):
    git_commit: str
    git_dirty: bool
    git_commit_role: Literal["base_commit_only"] = "base_commit_only"
    evaluator_version: str
    evaluation_code_hash: str
    evaluation_config_hash: str
    benchmark_hash: str
    benchmark_status: Literal["human_confirmed"]
    source_identity_note: str
    query_prompt_version: str
    final_prompt_version: str
    model: str
    evaluation_reference_time: str
    frozen_dataset: FrozenIdentity


@dataclass(frozen=True)
class EvaluationAssets:
    data_root: Path
    config: SummerConfig
    resources: list[Resource]
    provenance_records: list[dict[str, Any]]
    evidence_corpus: EvidenceCorpus
    resource_manifest: dict[str, Any]
    evidence_manifest: dict[str, Any]
    frozen_identity: FrozenIdentity


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_prefixed(value: bytes | str) -> str:
    payload = value.encode("utf-8") if isinstance(value, str) else value
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def read_yaml(path: Path) -> Any:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if value is None:
        raise ValueError(f"YAML is empty: {path}")
    return value


def load_retrieval_benchmark(path: Path) -> RetrievalBenchmark:
    return RetrievalBenchmark.model_validate(read_yaml(path))


def load_verification_benchmark(path: Path) -> list[VerificationGoldCase]:
    if not path.is_file():
        raise FileNotFoundError(
            "Historical Verification Gold is not distributed with this showcase. "
            "An authorized benchmark and the original frozen experiment archive are required; "
            "synthetic engineering fixtures cannot reproduce historical metrics."
        )
    cases = [
        VerificationGoldCase.model_validate(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [item.case_id for item in cases]
    if len(cases) != 15:
        raise ValueError("S4-Lite requires exactly 15 Verification Gold cases")
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate Verification Gold case_id")
    return cases


def load_abstention_benchmark(path: Path) -> AbstentionBenchmark:
    return AbstentionBenchmark.model_validate(read_yaml(path))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _git_output(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def git_state(root: Path) -> tuple[str, bool, str]:
    commit = _git_output(root, "rev-parse", "HEAD")
    dirty = bool(_git_output(root, "status", "--porcelain", "--untracked-files=all"))
    tag_target = _git_output(root, "rev-list", "-n", "1", EXPECTED_TAG)
    return commit, dirty, tag_target


def evaluation_code_hash(root: Path) -> str:
    parts: list[str] = []
    evaluation_root = root / "evaluation"
    for path in sorted(evaluation_root.rglob("*.py"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root).as_posix()
        normalized = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
        parts.append(relative + "\n" + normalized)
    if not parts:
        raise ValueError("evaluation source files are missing")
    return sha256_prefixed("\n--FILE--\n".join(parts))


def evaluation_config_hash(root: Path) -> str:
    material = {
        "evaluator_version": EVALUATOR_VERSION,
        "benchmark_status": BENCHMARK_STATUS,
        "evaluation_reference_time": EVALUATION_REFERENCE_TIME,
        "expected_identity": {
            "raw_run_id": EXPECTED_RAW_RUN_ID,
            "curation_id": EXPECTED_CURATION_ID,
            "curation_version": EXPECTED_CURATION_VERSION,
            "curation_hash": EXPECTED_CURATION_HASH,
            "resource_build_id": EXPECTED_RESOURCE_BUILD_ID,
            "evidence_build_id": EXPECTED_EVIDENCE_BUILD_ID,
            "counts": [EXPECTED_RESOURCE_COUNT, EXPECTED_DOCUMENT_COUNT, EXPECTED_CHUNK_COUNT],
        },
        "summer_config": read_yaml(root / "configs" / "summer.yaml"),
        "prompts": read_yaml(root / "configs" / "prompts.yaml"),
        "candidate_generation": {
            "retriever_limit": 10,
            "metadata_supplement_limit": 5,
            "verification_pool_size": 24,
            "verification_per_claim_type": 6,
        },
    }
    return sha256_prefixed(canonical_json(material))


def benchmark_hash(
    benchmark: RetrievalBenchmark,
    verification_cases: list[dict[str, Any]] | None = None,
    abstention_benchmark: AbstentionBenchmark | None = None,
) -> str:
    material: dict[str, Any] = {"retrieval": benchmark.model_dump(mode="json")}
    if verification_cases is not None:
        material["verification"] = verification_cases
    if abstention_benchmark is not None:
        material["abstention"] = abstention_benchmark.model_dump(mode="json")
    return sha256_prefixed(canonical_json(material))


def evaluate_retrieval(
    benchmark: RetrievalBenchmark, resources: list[Resource],
) -> dict[str, Any]:
    validate_retrieval_references(
        benchmark, {item.resource_id for item in resources}, require_confirmed=True,
    )
    retriever = TfidfRetriever(resources)
    rows: list[dict[str, Any]] = []
    for case in benchmark.cases:
        candidates = retriever.retrieve(case.gold_intent, top_k=benchmark.retrieval_top_k)
        acceptable = set(case.acceptable_resource_ids)
        first_rank = next(
            (rank for rank, item in enumerate(candidates, start=1) if item.resource_id in acceptable),
            None,
        )
        rows.append({
            "case_id": case.case_id,
            "category": case.category,
            "query": case.query,
            "top_k": [
                {
                    "resource_id": item.resource_id,
                    "rank": rank,
                    "score": item.retrieval_score,
                    "acceptable": item.resource_id in acceptable,
                }
                for rank, item in enumerate(candidates, start=1)
            ],
            "top1_reasonable": bool(candidates and candidates[0].resource_id in acceptable),
            "top3_hit": any(item.resource_id in acceptable for item in candidates[:3]),
            "first_acceptable_rank": first_rank,
            "reciprocal_rank_at_10": 1.0 / first_rank if first_rank else 0.0,
        })
    count = len(rows)
    return {
        "benchmark_version": benchmark.benchmark_version,
        "case_count": count,
        "top1_reasonable_count": sum(item["top1_reasonable"] for item in rows),
        "top1_reasonable_rate": sum(item["top1_reasonable"] for item in rows) / count,
        "top3_hit_count": sum(item["top3_hit"] for item in rows),
        "top3_hit_rate": sum(item["top3_hit"] for item in rows) / count,
        "mrr_at_10": sum(item["reciprocal_rank_at_10"] for item in rows) / count,
        "cases": rows,
    }


def _macro_f1(confusion: dict[str, dict[str, int]], labels: list[str]) -> float:
    scores: list[float] = []
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[gold][label] for gold in labels if gold != label)
        fn = sum(confusion[label][predicted] for predicted in labels if predicted != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores)


def validate_verification_gold(
    cases: list[VerificationGoldCase], assets: EvaluationAssets,
) -> None:
    resource_by_id = {item.resource_id: item for item in assets.resources}
    chunk_by_id = {item.chunk_id: item for item in assets.evidence_corpus.chunks}
    for case in cases:
        resource = resource_by_id.get(case.resource_id)
        if resource is None:
            raise ValueError(f"unknown Verification Gold resource_id: {case.resource_id}")
        if (resource.full_name or resource.name) != case.repository:
            raise ValueError(f"Verification Gold repository name mismatch: {case.case_id}")
        validate_verification_candidate(case.model_dump(mode="json"), chunk_by_id)


def evaluate_verification(
    cases: list[VerificationGoldCase], assets: EvaluationAssets,
) -> dict[str, Any]:
    validate_verification_gold(cases, assets)
    labels = ["SUPPORTED", "UNKNOWN", "CONTRADICTED"]
    confusion = {gold: {predicted: 0 for predicted in labels} for gold in labels}
    rows: list[dict[str, Any]] = []
    evaluation_time = datetime.fromisoformat(EVALUATION_REFERENCE_TIME.replace("Z", "+00:00"))
    for case in cases:
        result = verify_claim(
            case.claim,
            case.evidence,
            config=assets.config.verification,
            evaluation_time=evaluation_time,
        )
        confusion[case.gold_label][result.label] += 1
        rows.append({
            "case_id": case.case_id,
            "resource_id": case.resource_id,
            "claim_id": case.claim.claim_id,
            "claim_type": case.claim.claim_type,
            "gold_label": case.gold_label,
            "predicted_label": result.label,
            "correct": result.label == case.gold_label,
            "prediction_rationale": result.rationale,
            "human_rationale": case.human_rationale,
            "evidence_chunk_ids": [item.chunk_id for item in case.evidence],
        })
    gold_counts = Counter(item.gold_label for item in cases)
    predicted_counts = Counter(item["predicted_label"] for item in rows)
    support_sufficient = all(gold_counts[label] >= 2 for label in labels)
    correct = sum(item["correct"] for item in rows)
    return {
        "benchmark_version": "1.0.0",
        "case_count": len(rows),
        "correct_count": correct,
        "accuracy": correct / len(rows),
        "gold_label_counts": {label: gold_counts[label] for label in labels},
        "predicted_label_counts": {label: predicted_counts[label] for label in labels},
        "confusion_matrix": confusion,
        "macro_f1": _macro_f1(confusion, labels) if support_sufficient else None,
        "macro_f1_reported": support_sufficient,
        "cases": rows,
    }


def evaluate_abstention(
    benchmark: AbstentionBenchmark, assets: EvaluationAssets,
) -> dict[str, Any]:
    resource_by_id = {item.resource_id: item for item in assets.resources}
    rows: list[dict[str, Any]] = []
    evaluation_time = datetime.fromisoformat(EVALUATION_REFERENCE_TIME.replace("Z", "+00:00"))
    for case in benchmark.cases:
        resources = assets.resources
        provenance = assets.provenance_records
        evidence = assets.evidence_corpus
        if case.kind == "controlled_resource_subset":
            resource = resource_by_id.get(case.resource_id or "")
            if resource is None:
                raise ValueError(f"unknown controlled Resource: {case.resource_id}")
            resources = [resource]
            provenance = [
                item for item in assets.provenance_records
                if item.get("resource_id") == resource.resource_id
            ]
            evidence = EvidenceCorpus(
                documents=[
                    item for item in assets.evidence_corpus.documents
                    if item.resource_id == resource.resource_id
                ],
                chunks=[
                    item for item in assets.evidence_corpus.chunks
                    if item.resource_id == resource.resource_id
                ],
                manifest=assets.evidence_corpus.manifest,
            )
        result = run_core_pipeline(
            case.intent,
            config=assets.config,
            resources=resources,
            provenance_records=provenance,
            evidence_corpus=evidence,
            evaluation_time=evaluation_time,
            root=Path.cwd(),
        )
        passed = (
            result.status == case.expected_status
            and result.error_code == case.expected_error_code
        )
        rows.append({
            "case_id": case.case_id,
            "kind": case.kind,
            "description": case.description,
            "expected_status": case.expected_status,
            "actual_status": result.status,
            "expected_error_code": case.expected_error_code,
            "actual_error_code": result.error_code,
            "ranked_resource_ids": result.ranked_resource_ids,
            "passed": passed,
        })
    return {
        "benchmark_version": benchmark.benchmark_version,
        "case_count": len(rows),
        "passed_count": sum(item["passed"] for item in rows),
        "all_passed": all(item["passed"] for item in rows),
        "cases": rows,
    }


def _paths_for_data_root(config: SummerConfig, data_root: Path) -> SummerConfig:
    paths = config.paths.model_copy(update={
        "resources": str(data_root / "processed" / "resources.jsonl"),
        "resource_manifest": str(data_root / "metadata" / "build_manifest.json"),
        "resource_provenance": str(data_root / "processed" / "resource_provenance.jsonl"),
        "evidence_documents": str(data_root / "evidence" / "documents.jsonl"),
        "evidence_chunks": str(data_root / "evidence" / "chunks.jsonl"),
        "evidence_manifest": str(data_root / "metadata" / "evidence_build_manifest.json"),
    })
    return config.model_copy(update={
        "paths": paths,
        "logging": config.logging.model_copy(update={"save_sessions": False}),
    })


def _manifest_raw_run(resource_manifest: dict[str, Any], evidence_manifest: dict[str, Any]) -> str:
    evidence_value = evidence_manifest.get("raw_run_id")
    resource_input = str(resource_manifest.get("input_run", "")).replace("\\", "/").rstrip("/")
    resource_value = resource_input.rsplit("/", 1)[-1]
    if evidence_value != resource_value:
        raise ValueError("Resource and Evidence manifests reference different raw runs")
    return str(evidence_value)


def assert_frozen_identity(identity: FrozenIdentity) -> None:
    expected = {
        "git_tag": EXPECTED_TAG,
        "git_tag_target": EXPECTED_TAG_TARGET,
        "raw_run_id": EXPECTED_RAW_RUN_ID,
        "curation_manifest_id": EXPECTED_CURATION_ID,
        "curation_manifest_version": EXPECTED_CURATION_VERSION,
        "curation_manifest_hash": EXPECTED_CURATION_HASH,
        "resource_build_id": EXPECTED_RESOURCE_BUILD_ID,
        "evidence_build_id": EXPECTED_EVIDENCE_BUILD_ID,
        "resource_count": EXPECTED_RESOURCE_COUNT,
        "document_count": EXPECTED_DOCUMENT_COUNT,
        "chunk_count": EXPECTED_CHUNK_COUNT,
        "excluded_resources_absent": True,
        "orphan_documents": 0,
        "orphan_chunks": 0,
        "identity_matches_promoted_canonical": True,
    }
    actual = identity.model_dump(mode="json")
    mismatches = [
        f"{field}: expected {wanted!r}, got {actual.get(field)!r}"
        for field, wanted in expected.items() if actual.get(field) != wanted
    ]
    if mismatches:
        raise ValueError("frozen dataset identity mismatch: " + "; ".join(mismatches))


def load_and_validate_assets(data_root: Path, *, project_root: Path) -> EvaluationAssets:
    """Load one explicitly supplied frozen root; there is deliberately no fallback."""

    data_root = data_root.resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"explicit --data-root does not exist: {data_root}")
    base_config = load_config(project_root / "configs" / "summer.yaml")
    config = _paths_for_data_root(base_config, data_root)
    resource_manifest = _read_json(data_root / "metadata" / "build_manifest.json")
    evidence_manifest = _read_json(data_root / "metadata" / "evidence_build_manifest.json")
    resources = load_resources(data_root / "processed" / "resources.jsonl")
    provenance = load_resource_provenance(data_root / "processed" / "resource_provenance.jsonl")
    evidence = load_evidence_corpus(
        documents_path=data_root / "evidence" / "documents.jsonl",
        chunks_path=data_root / "evidence" / "chunks.jsonl",
        evidence_manifest_path=data_root / "metadata" / "evidence_build_manifest.json",
        resource_manifest_path=data_root / "metadata" / "build_manifest.json",
        resources=resources,
        compatibility=config.evidence_compatibility,
    )
    _, _, tag_target = git_state(project_root)
    resource_ids = {item.resource_id for item in resources}
    document_ids = {item.evidence_id for item in evidence.documents}
    orphan_documents = sum(item.resource_id not in resource_ids for item in evidence.documents)
    orphan_chunks = sum(
        item.resource_id not in resource_ids or item.evidence_id not in document_ids
        for item in evidence.chunks
    )
    curation = resource_manifest.get("curation") or {}
    curation_hash = resource_manifest.get("curation_manifest_hash") or curation.get("canonical_hash")
    identity = FrozenIdentity(
        git_tag=EXPECTED_TAG,
        git_tag_target=tag_target,
        raw_run_id=_manifest_raw_run(resource_manifest, evidence_manifest),
        curation_manifest_id=(
            resource_manifest.get("curation_manifest_id") or curation.get("manifest_id")
        ),
        curation_manifest_version=(
            resource_manifest.get("curation_manifest_version") or curation.get("version")
        ),
        curation_manifest_hash=curation_hash,
        resource_build_id=resource_manifest.get("build_id"),
        evidence_build_id=evidence_manifest.get("evidence_build_id"),
        resource_count=len(resources),
        document_count=len(evidence.documents),
        chunk_count=len(evidence.chunks),
        excluded_resources_absent=not bool(resource_ids & EXPECTED_EXCLUSIONS),
        orphan_documents=orphan_documents,
        orphan_chunks=orphan_chunks,
        identity_matches_promoted_canonical=True,
        data_root=data_root.relative_to(project_root).as_posix()
        if data_root.is_relative_to(project_root) else str(data_root),
    )
    assert_frozen_identity(identity)
    if evidence_manifest.get("input_resource_build_id") != identity.resource_build_id:
        raise ValueError("Evidence manifest does not reference the frozen Resource build")
    if resource_manifest.get("final_resources") != EXPECTED_RESOURCE_COUNT:
        raise ValueError("Resource manifest final_resources does not match the frozen count")
    if evidence_manifest.get("document_count") != EXPECTED_DOCUMENT_COUNT:
        raise ValueError("Evidence manifest document_count does not match the frozen count")
    if evidence_manifest.get("chunk_count") != EXPECTED_CHUNK_COUNT:
        raise ValueError("Evidence manifest chunk_count does not match the frozen count")
    return EvaluationAssets(
        data_root=data_root,
        config=config,
        resources=resources,
        provenance_records=provenance,
        evidence_corpus=evidence,
        resource_manifest=resource_manifest,
        evidence_manifest=evidence_manifest,
        frozen_identity=identity,
    )


def build_experiment_identity(
    *, project_root: Path, benchmark: RetrievalBenchmark, frozen: FrozenIdentity,
    verification_cases: list[dict[str, Any]] | None = None,
    abstention_benchmark: AbstentionBenchmark | None = None,
) -> ExperimentIdentity:
    commit, dirty, _ = git_state(project_root)
    prompts = load_prompts(project_root / "configs" / "prompts.yaml")
    config = load_config(project_root / "configs" / "summer.yaml")
    return ExperimentIdentity(
        git_commit=commit,
        git_dirty=dirty,
        evaluator_version=EVALUATOR_VERSION,
        evaluation_code_hash=evaluation_code_hash(project_root),
        evaluation_config_hash=evaluation_config_hash(project_root),
        benchmark_hash=benchmark_hash(
            benchmark, verification_cases, abstention_benchmark,
        ),
        benchmark_status=BENCHMARK_STATUS,
        source_identity_note=(
            "git_commit identifies only the base commit while git_dirty is true; "
            "evaluation_code_hash identifies the current uncommitted evaluator source."
        ),
        query_prompt_version=prompts.query_understanding.prompt_version,
        final_prompt_version=prompts.final_generation.prompt_version,
        model=config.llm.model,
        evaluation_reference_time=EVALUATION_REFERENCE_TIME,
        frozen_dataset=frozen,
    )


def _normalized(value: str) -> str:
    return " ".join(re.sub(r"[-_]", " ", value.casefold()).split())


def _phrase_matches(text: str, term: str) -> bool:
    normalized_text = _normalized(text)
    normalized_term = _normalized(term)
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(normalized_term)}(?![a-z0-9])", normalized_text))


def metadata_matches(resource: Resource, terms: list[str]) -> tuple[list[str], list[str]]:
    fields = {
        "name": resource.name,
        "full_name": resource.full_name or "",
        "description": resource.description or "",
        "topics": " ".join(resource.topics),
        "frameworks": " ".join(resource.frameworks),
        "skills": " ".join(resource.skills),
    }
    matched_terms: set[str] = set()
    matched_fields: set[str] = set()
    for field, text in fields.items():
        for term in terms:
            if _phrase_matches(text, term):
                matched_terms.add(term)
                matched_fields.add(field)
    return sorted(matched_terms, key=str.casefold), sorted(matched_fields)


def prepare_retrieval_candidates(
    benchmark: RetrievalBenchmark, resources: list[Resource],
) -> list[dict[str, Any]]:
    retriever = TfidfRetriever(resources)
    resource_by_id = {item.resource_id: item for item in resources}
    output: list[dict[str, Any]] = []
    for case in benchmark.cases:
        top = retriever.retrieve(case.gold_intent, top_k=benchmark.candidate_pool["retriever_limit"])
        top_by_id = {item.resource_id: item for item in top}
        metadata: list[tuple[int, str, Resource, list[str], list[str]]] = []
        match_by_id: dict[str, tuple[list[str], list[str]]] = {}
        for resource in resources:
            terms, fields = metadata_matches(resource, case.metadata_match_terms)
            if terms:
                match_by_id[resource.resource_id] = (terms, fields)
                metadata.append((-len(fields), resource.resource_id, resource, terms, fields))
        metadata.sort(key=lambda item: (item[0], item[1]))
        selected_ids = [item.resource_id for item in top]
        supplement_limit = benchmark.candidate_pool["metadata_supplement_limit"]
        supplements = 0
        for _, resource_id, _, _, _ in metadata:
            if resource_id in top_by_id:
                continue
            selected_ids.append(resource_id)
            supplements += 1
            if supplements >= supplement_limit:
                break
        candidates: list[dict[str, Any]] = []
        for resource_id in selected_ids:
            resource = resource_by_id[resource_id]
            sources: list[str] = []
            if resource_id in top_by_id:
                sources.append("retriever_top10")
            terms, fields = match_by_id.get(resource_id, ([], []))
            if terms:
                sources.append("metadata_match")
            runtime_candidate = top_by_id.get(resource_id)
            candidates.append({
                "resource_id": resource_id,
                "repository": resource.full_name or resource.name,
                "description": resource.description,
                "topics": resource.topics,
                "candidate_sources": sources,
                "metadata_matched_terms": terms,
                "metadata_matched_fields": fields,
                "retriever_rank": (
                    next(index for index, item in enumerate(top, start=1) if item.resource_id == resource_id)
                    if runtime_candidate else None
                ),
                "retrieval_score": runtime_candidate.retrieval_score if runtime_candidate else None,
                "acceptable": None,
            })
        output.append({
            "case_id": case.case_id,
            "category": case.category,
            "query": case.query,
            "gold_intent": case.gold_intent.model_dump(mode="json"),
            "candidates": candidates,
        })
    return output


def render_retrieval_sheet(cases: list[dict[str, Any]]) -> str:
    lines = [
        "# S4-Lite Retrieval Gold Annotation Sheet",
        "",
        "> Draft candidate pools only. Current ranking must not be copied into Gold. ",
        "> Confirm the StructuredIntent and independently select acceptable resources.",
        "",
    ]
    for case in cases:
        lines.extend((
            f"## {case['case_id']} — {case['category']}", "",
            f"**Query:** {case['query']}", "",
            "**Draft StructuredIntent:**", "", "```json",
            json.dumps(case["gold_intent"], ensure_ascii=False, indent=2), "```", "",
        ))
        for candidate in case["candidates"]:
            source = " + ".join(candidate["candidate_sources"])
            topics = ", ".join(candidate["topics"]) or "(none)"
            lines.extend((
                f"- [ ] **{candidate['repository']}** — `{candidate['resource_id']}`",
                f"  - Candidate source: `{source}`",
                f"  - Description: {candidate['description'] or '(missing)' }",
                f"  - Topics: {topics}",
            ))
        lines.extend(("", "**Intent confirmed:** [ ]", "", "**Annotation note:**", "", ""))
    return "\n".join(lines).rstrip() + "\n"


def _select_verification_candidates(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_type[record["claim"]["claim_type"]].append(record)
    chosen: list[dict[str, Any]] = []
    globally_seen_resources: set[str] = set()
    label_cycle = ("CONTRADICTED", "UNKNOWN", "SUPPORTED")
    for claim_type in ("uses_framework", "covers_topic", "is_maintained", "has_license"):
        candidates = by_type.get(claim_type, [])
        selected: list[dict[str, Any]] = []
        remaining = list(candidates)
        while remaining and len(selected) < 6:
            preferred_label = label_cycle[len(selected) % len(label_cycle)]
            matching = [item for item in remaining if item["system_verification_label"] == preferred_label]
            pool = matching or remaining
            pool.sort(key=lambda item: (
                item["resource_id"] in globally_seen_resources,
                item["evidence_character_count"],
                item["resource_id"],
                item["claim"]["claim_id"],
            ))
            item = pool[0]
            selected.append(item)
            globally_seen_resources.add(item["resource_id"])
            remaining.remove(item)
        if len(selected) != 6:
            raise ValueError(f"not enough real Evidence candidates for {claim_type}: {len(selected)}")
        chosen.extend(selected)
    for index, item in enumerate(chosen, start=1):
        item["case_id"] = f"verification_candidate_{index:03d}"
    return chosen


def prepare_verification_candidates(
    benchmark: RetrievalBenchmark, assets: EvaluationAssets,
) -> list[dict[str, Any]]:
    resource_by_id = {item.resource_id: item for item in assets.resources}
    unique: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
    evaluation_time = datetime.fromisoformat(EVALUATION_REFERENCE_TIME.replace("Z", "+00:00"))
    for case in benchmark.cases:
        result = run_core_pipeline(
            case.gold_intent,
            config=assets.config,
            resources=assets.resources,
            provenance_records=assets.provenance_records,
            evidence_corpus=assets.evidence_corpus,
            evaluation_time=evaluation_time,
            root=Path.cwd(),
        )
        if result.status == "ERROR":
            raise ValueError(f"core pipeline failed for {case.case_id}: {result.error_code}")
        for assessment in result.assessments:
            verification_by_claim = {item.claim_id: item for item in assessment.verifications}
            for claim in assessment.claims:
                evidence = assessment.evidence_by_claim.get(claim.claim_id, [])
                if not evidence:
                    continue
                verification = verification_by_claim[claim.claim_id]
                key = (claim.claim_id, tuple(item.chunk_id for item in evidence))
                if key in unique:
                    continue
                resource = resource_by_id[claim.resource_id]
                unique[key] = {
                    "case_id": "pending",
                    "source_query_case": case.case_id,
                    "resource_id": claim.resource_id,
                    "repository": resource.full_name or resource.name,
                    "claim": claim.model_dump(mode="json"),
                    "evidence": [item.model_dump(mode="json") for item in evidence],
                    "evidence_character_count": sum(len(item.text) for item in evidence),
                    "system_verification_label": verification.label,
                    "system_verification_score": verification.score,
                    "system_verification_rationale": verification.rationale,
                    "system_verification_evidence_chunk_ids": verification.evidence_chunk_ids,
                    "selection_status": "candidate_only_not_gold",
                }
    return _select_verification_candidates(list(unique.values()))


def _blockquote(text: str) -> list[str]:
    return ["> " + line if line else ">" for line in text.splitlines()]


def render_verification_blind_sheet(cases: list[dict[str, Any]]) -> str:
    lines = [
        "# S4-Lite Verification Blind Annotation Sheet", "",
        "> Select exactly 15 cases for the final benchmark. Judge only the Claim and displayed Evidence.",
        "> The current system prediction is intentionally hidden and must not determine Gold.", "",
    ]
    for case in cases:
        lines.extend((
            f"## {case['case_id']}", "",
            f"**Resource:** {case['repository']}", "",
            f"**Claim:** {case['claim']['text']}", "",
            "**Evidence:**", "",
        ))
        for index, evidence in enumerate(case["evidence"], start=1):
            lines.extend((
                f"### Evidence {index}", "",
                f"Source: `{evidence['evidence_source_type']}` — {evidence['source_url']}", "",
                *_blockquote(evidence["text"]), "",
            ))
        lines.extend((
            "**Include in final 15:** [ ]", "",
            "**Gold label:**", "",
            "- [ ] SUPPORTED", "- [ ] UNKNOWN", "- [ ] CONTRADICTED", "",
            "**Human rationale:**", "", "", "<details>",
            "<summary>Internal provenance IDs</summary>", "",
            f"- Resource ID: `{case['resource_id']}`",
            f"- Claim ID: `{case['claim']['claim_id']}`",
            f"- Claim type: `{case['claim']['claim_type']}`",
            f"- Source query case: `{case['source_query_case']}`",
        ))
        for evidence in case["evidence"]:
            lines.extend((
                f"- Evidence ID: `{evidence['evidence_id']}`",
                f"- Chunk ID: `{evidence['chunk_id']}`",
                f"- Source ID: `{evidence['source_id']}`",
            ))
        lines.extend(("", "</details>", "", ""))
    return "\n".join(lines).rstrip() + "\n"


def validate_retrieval_references(
    benchmark: RetrievalBenchmark,
    canonical_resource_ids: set[str],
    *,
    require_confirmed: bool = False,
) -> None:
    """Validate references without treating a draft as formal Gold."""

    referenced = {
        resource_id
        for case in benchmark.cases
        for resource_id in case.acceptable_resource_ids
    }
    unknown = sorted(referenced - canonical_resource_ids)
    if unknown:
        raise ValueError(f"benchmark references unknown Resource IDs: {unknown}")
    if require_confirmed and benchmark.benchmark_status != "human_confirmed":
        raise ValueError("formal metrics require a human-confirmed benchmark")


def validate_verification_candidate(
    candidate: dict[str, Any], canonical_chunks: dict[str, Any],
) -> None:
    """Check that candidate Evidence is an exact same-resource canonical record."""

    resource_id = candidate["resource_id"]
    claim = candidate["claim"]
    if claim["resource_id"] != resource_id:
        raise ValueError("Claim and candidate Resource identities differ")
    for evidence in candidate["evidence"]:
        if evidence["resource_id"] != resource_id:
            raise ValueError("cross-resource Claim/Evidence candidate")
        chunk = canonical_chunks.get(evidence["chunk_id"])
        if chunk is None:
            raise ValueError(f"unknown canonical chunk_id: {evidence['chunk_id']}")
        expected = {
            "resource_id": chunk.resource_id,
            "evidence_id": chunk.evidence_id,
            "source_id": chunk.source_id,
            "text": chunk.text,
        }
        actual = {field: evidence[field] for field in expected}
        if actual != expected:
            raise ValueError(f"Evidence candidate differs from canonical chunk {chunk.chunk_id}")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )


__all__ = [
    "AbstentionBenchmark", "BENCHMARK_STATUS", "EVALUATOR_VERSION", "EvaluationAssets",
    "ExperimentIdentity", "FrozenIdentity", "RetrievalBenchmark", "VerificationGoldCase",
    "assert_frozen_identity",
    "benchmark_hash", "build_experiment_identity", "canonical_json", "evaluation_code_hash",
    "evaluate_abstention", "evaluate_retrieval", "evaluate_verification",
    "evaluation_config_hash", "load_abstention_benchmark", "load_and_validate_assets",
    "load_retrieval_benchmark", "load_verification_benchmark",
    "metadata_matches", "prepare_retrieval_candidates", "prepare_verification_candidates",
    "render_retrieval_sheet", "render_verification_blind_sheet",
    "validate_retrieval_references", "validate_verification_candidate",
    "validate_verification_gold", "write_json", "write_jsonl",
]
