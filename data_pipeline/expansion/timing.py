"""Run one S3.5 stage command and record its wall-clock duration."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path


LABELS = {
    "collection": "collection_seconds",
    "phase2": "resource_build_seconds",
    "phase3": "evidence_build_seconds",
    "phase2-validation": "resource_validation_seconds",
    "phase3-validation": "evidence_validation_seconds",
    "expansion-validation": "expansion_validation_seconds",
    "smoke": "smoke_seconds",
}


def record_duration(output_root: Path, label: str, seconds: float) -> Path:
    target = output_root.resolve() / "data" / "metadata" / "s3_5_performance.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    values = json.loads(target.read_text(encoding="utf-8")) if target.is_file() else {}
    values[LABELS[label]] = round(float(seconds), 3)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(values, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, target)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--label", choices=sorted(LABELS), required=True)
    args, command = parser.parse_known_args()
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("a command is required after --")
    if command[0] in {"-m", "-c"}:
        command = [sys.executable, *command]
    started = time.perf_counter()
    result = subprocess.run(command, check=False)
    record_duration(args.output_root, args.label, time.perf_counter() - started)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
