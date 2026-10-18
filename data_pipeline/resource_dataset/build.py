"""Build the canonical Resource Dataset from one explicit Phase 1 raw run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .core import build_dataset
from .curation import load_curation_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-run", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-root", type=Path, help="Isolated project-like output root.")
    parser.add_argument("--curation-manifest", type=Path, help="Explicit exclude-only curation manifest.")
    args = parser.parse_args()
    curation = load_curation_manifest(args.curation_manifest) if args.curation_manifest else None
    print(json.dumps(
        build_dataset(args.input_run, load_config(args.config, output_root=args.output_root), curation=curation),
        ensure_ascii=False, indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
