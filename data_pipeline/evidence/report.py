"""Regenerate the human-readable report from committed quality metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import project_root
from .core import _report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    root = (args.output_root or project_root()).resolve()
    quality_path = root / "data" / "reports" / "evidence_corpus_quality.json"
    manifest_path = root / "data" / "metadata" / "evidence_build_manifest.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    target = root / "data" / "reports" / "evidence_corpus_report.md"
    target.write_text(_report(quality, manifest), encoding="utf-8")
    print(target.relative_to(root).as_posix())


if __name__ == "__main__":
    main()
