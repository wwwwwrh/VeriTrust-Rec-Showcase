"""Phase 3 provenance-aware Evidence Corpus construction."""

from .loader import (
    get_chunks_for_resource,
    get_documents_for_resource,
    load_evidence_chunks,
    load_evidence_documents,
)
from .schema import EvidenceChunk, EvidenceDocument

__all__ = [
    "EvidenceChunk",
    "EvidenceDocument",
    "load_evidence_documents",
    "load_evidence_chunks",
    "get_documents_for_resource",
    "get_chunks_for_resource",
]
