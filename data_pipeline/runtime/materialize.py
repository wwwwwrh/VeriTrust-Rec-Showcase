"""Restore one explicitly selected frozen build to canonical runtime paths.

This is deliberately a materialization operation, not a builder and not a
fallback selector. It never discovers a source directory and never overwrites
canonical data whose bytes or build identity differ from the accepted source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EXPECTED_RESOURCE_BUILD_ID = "build_60c4f8a08dad8614"
EXPECTED_EVIDENCE_BUILD_ID = "evidence_build_10deed019a096c1a"
EXPECTED_CURATION_HASH = (
    "sha256:0346f405c22dc339e0d3abfc043ae05df6a4373b43969bae11428c2def9ac9cb"
)
EXPECTED_RESOURCES = 200
EXPECTED_DOCUMENTS = 400
EXPECTED_CHUNKS = 5902


@dataclass(frozen=True)
class ExpectedIdentity:
    resource_build_id: str = EXPECTED_RESOURCE_BUILD_ID
    evidence_build_id: str = EXPECTED_EVIDENCE_BUILD_ID
    curation_hash: str = EXPECTED_CURATION_HASH
    resources: int = EXPECTED_RESOURCES
    documents: int = EXPECTED_DOCUMENTS
    chunks: int = EXPECTED_CHUNKS


RUNTIME_FILES = {
    "resources": Path("processed/resources.jsonl"),
    "resource_provenance": Path("processed/resource_provenance.jsonl"),
    "resource_manifest": Path("metadata/build_manifest.json"),
    "documents": Path("evidence/documents.jsonl"),
    "chunks": Path("evidence/chunks.jsonl"),
    "evidence_manifest": Path("metadata/evidence_build_manifest.json"),
}

CANONICAL_FILES = {
    name: Path("data") / relative for name, relative in RUNTIME_FILES.items()
}
RESOURCE_FILES = ("resources", "resource_provenance", "resource_manifest")
EVIDENCE_FILES = ("documents", "chunks", "evidence_manifest")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read JSONL {path}: {exc}") from exc
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object at {path}:{number}")
        records.append(value)
    return records


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _manifest_curation_hash(manifest: dict[str, Any]) -> Any:
    embedded = manifest.get("curation") or {}
    return manifest.get("curation_manifest_hash") or embedded.get("canonical_hash")


def validate_frozen_root(
    source_root: Path, expected: ExpectedIdentity = ExpectedIdentity(),
) -> dict[str, Any]:
    source_root = source_root.resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"explicit frozen source root is missing: {source_root}")
    paths = {name: source_root / relative for name, relative in RUNTIME_FILES.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("frozen source is incomplete: " + ", ".join(missing))

    resource_manifest = _read_json(paths["resource_manifest"])
    evidence_manifest = _read_json(paths["evidence_manifest"])
    checks = {
        "Resource Build ID": (resource_manifest.get("build_id"), expected.resource_build_id),
        "Evidence Build ID": (
            evidence_manifest.get("evidence_build_id"), expected.evidence_build_id,
        ),
        "curation hash": (_manifest_curation_hash(resource_manifest), expected.curation_hash),
        "Evidence input Resource Build ID": (
            evidence_manifest.get("input_resource_build_id"), expected.resource_build_id,
        ),
        "Resource manifest count": (
            resource_manifest.get("final_resources"), expected.resources,
        ),
        "Evidence manifest Document count": (
            evidence_manifest.get("document_count"), expected.documents,
        ),
        "Evidence manifest Chunk count": (
            evidence_manifest.get("chunk_count"), expected.chunks,
        ),
    }
    mismatches = [
        f"{label}: expected {wanted!r}, got {actual!r}"
        for label, (actual, wanted) in checks.items() if actual != wanted
    ]
    if mismatches:
        raise ValueError("frozen runtime identity mismatch: " + "; ".join(mismatches))

    resources = _read_jsonl(paths["resources"])
    provenance = _read_jsonl(paths["resource_provenance"])
    documents = _read_jsonl(paths["documents"])
    chunks = _read_jsonl(paths["chunks"])
    actual_counts = (len(resources), len(documents), len(chunks))
    expected_counts = (expected.resources, expected.documents, expected.chunks)
    if actual_counts != expected_counts:
        raise ValueError(
            "frozen runtime record counts mismatch: "
            f"expected {expected_counts}, got {actual_counts}"
        )

    resource_ids = [item.get("resource_id") for item in resources]
    document_ids = [item.get("evidence_id") for item in documents]
    chunk_ids = [item.get("chunk_id") for item in chunks]
    if None in resource_ids or len(resource_ids) != len(set(resource_ids)):
        raise ValueError("Resource records contain missing or duplicate resource_id")
    if None in document_ids or len(document_ids) != len(set(document_ids)):
        raise ValueError("Evidence Documents contain missing or duplicate evidence_id")
    if None in chunk_ids or len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Evidence Chunks contain missing or duplicate chunk_id")
    resource_id_set = set(resource_ids)
    document_id_set = set(document_ids)
    if any(item.get("resource_id") not in resource_id_set for item in provenance):
        raise ValueError("Resource provenance contains an unknown resource_id")
    if any(
        item.get("resource_id") not in resource_id_set
        or item.get("evidence_build_id") != expected.evidence_build_id
        for item in documents
    ):
        raise ValueError("Evidence Documents contain unknown Resources or build IDs")
    if any(
        item.get("resource_id") not in resource_id_set
        or item.get("evidence_id") not in document_id_set
        or item.get("evidence_build_id") != expected.evidence_build_id
        for item in chunks
    ):
        raise ValueError("Evidence Chunks contain orphan identities or build IDs")

    return {
        "source_root": str(source_root),
        "resource_build_id": expected.resource_build_id,
        "evidence_build_id": expected.evidence_build_id,
        "curation_hash": expected.curation_hash,
        "counts": {
            "resources": len(resources),
            "documents": len(documents),
            "chunks": len(chunks),
        },
        "paths": paths,
        "hashes": {name: _hash_file(path) for name, path in paths.items()},
    }


def _check_existing_targets(
    source_paths: dict[str, Path], target_paths: dict[str, Path],
) -> tuple[list[str], list[str]]:
    matching: list[str] = []
    missing: list[str] = []
    for name, target in target_paths.items():
        if not target.exists():
            missing.append(name)
            continue
        if not target.is_file() or _hash_file(target) != _hash_file(source_paths[name]):
            raise FileExistsError(
                f"Canonical dataset conflict: {target} differs from the accepted frozen source"
            )
        matching.append(name)
    return matching, missing


def _copy_missing_atomically(
    source_paths: dict[str, Path], target_paths: dict[str, Path], missing: list[str],
) -> None:
    manifest_names = {"resource_manifest", "evidence_manifest"}
    order = [name for name in missing if name not in manifest_names]
    order.extend(name for name in missing if name in manifest_names)
    staged: dict[str, Path] = {}
    try:
        for name in order:
            target = target_paths[name]
            target.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(
                prefix=f".{target.name}.", suffix=".tmp", dir=target.parent,
            )
            os.close(descriptor)
            temporary_path = Path(temporary)
            shutil.copyfile(source_paths[name], temporary_path)
            if _hash_file(temporary_path) != _hash_file(source_paths[name]):
                raise OSError(f"copy verification failed for {name}")
            staged[name] = temporary_path
        for name in order:
            os.replace(staged.pop(name), target_paths[name])
    finally:
        for temporary_path in staged.values():
            temporary_path.unlink(missing_ok=True)


def materialize_runtime_data(
    source_root: Path,
    *,
    project_root: Path,
    expected: ExpectedIdentity = ExpectedIdentity(),
    components: str = "all",
) -> dict[str, Any]:
    if components not in {"all", "resources", "evidence"}:
        raise ValueError("components must be one of: all, resources, evidence")
    audit = validate_frozen_root(source_root, expected)
    all_source_paths: dict[str, Path] = audit.pop("paths")
    project_root = project_root.resolve()
    selected_names = (
        tuple(CANONICAL_FILES)
        if components == "all"
        else RESOURCE_FILES if components == "resources" else EVIDENCE_FILES
    )
    source_paths = {name: all_source_paths[name] for name in selected_names}
    target_paths = {name: project_root / CANONICAL_FILES[name] for name in selected_names}
    matching, missing = _check_existing_targets(source_paths, target_paths)
    _copy_missing_atomically(source_paths, target_paths, missing)

    # Re-read every canonical runtime artifact and require byte-for-byte identity.
    remaining = [
        name for name, target in target_paths.items()
        if not target.is_file() or _hash_file(target) != audit["hashes"][name]
    ]
    if remaining:
        raise OSError(f"canonical materialization verification failed: {remaining}")
    if components == "all":
        validate_frozen_root(project_root / "data", expected)
    return {
        **audit,
        "target_root": str(project_root / "data"),
        "status": "already_materialized" if not missing else "materialized",
        "components": components,
        "copied": missing,
        "already_matching": matching,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize the accepted frozen Summer runtime data explicitly"
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--components", choices=("all", "resources", "evidence"), default="all",
        help="Explicit runtime component set to materialize; defaults to all",
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2],
        help="Canonical project root; defaults to this repository",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = materialize_runtime_data(
        args.source_root, project_root=args.project_root, components=args.components,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
