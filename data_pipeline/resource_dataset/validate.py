"""Validate the existing canonical Resource Dataset outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .core import validate_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-root", type=Path, help="Isolated project-like output root.")
    args = parser.parse_args()
    print(json.dumps(validate_dataset(load_config(args.config, output_root=args.output_root)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
