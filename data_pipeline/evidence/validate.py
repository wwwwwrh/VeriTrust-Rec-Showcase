"""CLI and API for validating the committed Evidence Corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import project_root
from .core import load_resources, validate_records
from .loader import load_evidence_chunks, load_evidence_documents


def validate_canonical(resources_path: Path | None = None, output_root: Path | None = None) -> dict:
    root = (output_root or project_root()).resolve()
    manifest_path = root / "data" / "metadata" / "evidence_build_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    resources = load_resources(resources_path or root / manifest["resources_path"])
    documents = load_evidence_documents(root / "data" / "evidence" / "documents.jsonl", manifest_path=manifest_path)
    chunks = load_evidence_chunks(root / "data" / "evidence" / "chunks.jsonl", manifest_path=manifest_path)
    return {"evidence_build_id": manifest["evidence_build_id"], **validate_records(documents, chunks, manifest, resources)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the canonical Evidence Corpus")
    parser.add_argument("--resources", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(validate_canonical(args.resources, args.output_root), indent=2))


if __name__ == "__main__":
    main()
