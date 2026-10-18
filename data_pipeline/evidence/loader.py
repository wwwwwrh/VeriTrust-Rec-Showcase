"""Safe loaders for the canonical Evidence Corpus."""

from __future__ import annotations

import json
from pathlib import Path

from .config import project_root
from .schema import EvidenceChunk, EvidenceDocument


def _manifest(path: Path | None = None) -> dict:
    target = path or project_root() / "data" / "metadata" / "evidence_build_manifest.json"
    if not target.is_file():
        raise FileNotFoundError("canonical Evidence Corpus manifest is missing")
    manifest = json.loads(target.read_text(encoding="utf-8"))
    if manifest.get("manifest_role") != "atomic_build_commit_marker":
        raise ValueError("invalid Evidence Corpus commit marker")
    return manifest


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_evidence_documents(
    path: Path | None = None, *, manifest_path: Path | None = None,
) -> list[EvidenceDocument]:
    manifest = _manifest(manifest_path)
    target = path or project_root() / "data" / "evidence" / "documents.jsonl"
    documents = [EvidenceDocument.model_validate(item) for item in _records(target)]
    if len(documents) != manifest.get("document_count") or any(
        item.evidence_build_id != manifest.get("evidence_build_id") for item in documents
    ):
        raise ValueError("documents do not match the committed Evidence Corpus manifest")
    return documents


def load_evidence_chunks(
    path: Path | None = None, *, manifest_path: Path | None = None,
) -> list[EvidenceChunk]:
    manifest = _manifest(manifest_path)
    target = path or project_root() / "data" / "evidence" / "chunks.jsonl"
    chunks = [EvidenceChunk.model_validate(item) for item in _records(target)]
    if len(chunks) != manifest.get("chunk_count") or any(
        item.evidence_build_id != manifest.get("evidence_build_id") for item in chunks
    ):
        raise ValueError("chunks do not match the committed Evidence Corpus manifest")
    return chunks


def get_documents_for_resource(resource_id: str) -> list[EvidenceDocument]:
    return [item for item in load_evidence_documents() if item.resource_id == resource_id]


def get_chunks_for_resource(resource_id: str) -> list[EvidenceChunk]:
    return [item for item in load_evidence_chunks() if item.resource_id == resource_id]
