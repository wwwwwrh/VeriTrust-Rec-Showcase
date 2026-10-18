"""Print the current Resource Dataset quality report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-root", type=Path, help="Isolated project-like output root.")
    args = parser.parse_args()
    config = load_config(args.config, output_root=args.output_root)
    report = config.path("reports_root") / "resource_dataset_quality.json"
    print(json.dumps(json.loads(report.read_text(encoding="utf-8")), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
