"""CLI for building the Phase 3 Evidence Corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .core import build_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the deterministic Phase 3 Evidence Corpus")
    parser.add_argument("--resources", type=Path, required=True, help="Explicit Phase 2 resources.parquet")
    parser.add_argument("--input-run", type=Path, required=True, help="Explicit Phase 1 raw run directory")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None, help="Isolated project-like output root")
    parser.add_argument("--resource-manifest", type=Path, default=None, help="Explicit Phase 2 build manifest")
    args = parser.parse_args()
    config = load_config(args.config, output_root=args.output_root, resource_manifest=args.resource_manifest)
    result = build_corpus(args.resources, args.input_run, config, output_root=args.output_root)
    print(json.dumps({key: result[key] for key in ("evidence_build_id", "documents", "chunks", "valid")}, indent=2))


if __name__ == "__main__":
    main()
