"""Validate one isolated S3.5 stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_pipeline.resource_dataset.curation import load_curation_manifest

from .core import validate_expansion


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-run", type=Path, required=True)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--baseline-resources", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--require-evidence", action="store_true")
    parser.add_argument("--curation-manifest", type=Path)
    args = parser.parse_args()
    curation = load_curation_manifest(args.curation_manifest) if args.curation_manifest else None
    result = validate_expansion(
        args.raw_run, args.resources, baseline_resources=args.baseline_resources,
        output_root=args.output_root, require_evidence=args.require_evidence, curation=curation,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
