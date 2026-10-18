from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from data_pipeline.runtime.materialize import (
    ExpectedIdentity,
    RUNTIME_FILES,
    materialize_runtime_data,
    validate_frozen_root,
)


TEST_IDENTITY = ExpectedIdentity(
    resource_build_id="build_0000000000000001",
    evidence_build_id="evidence_build_0000000000000002",
    curation_hash="sha256:" + "3" * 64,
    resources=2,
    documents=2,
    chunks=2,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value) + "\n" for value in values), encoding="utf-8")


def make_frozen(root: Path) -> Path:
    data = root / "data"
    resources = [{"resource_id": "github:1"}, {"resource_id": "github:2"}]
    documents = [
        {
            "evidence_id": f"ev_{index:016x}", "resource_id": f"github:{index}",
            "evidence_build_id": TEST_IDENTITY.evidence_build_id,
        }
        for index in (1, 2)
    ]
    chunks = [
        {
            "chunk_id": f"ch_{index:016x}", "evidence_id": f"ev_{index:016x}",
            "resource_id": f"github:{index}",
            "evidence_build_id": TEST_IDENTITY.evidence_build_id,
        }
        for index in (1, 2)
    ]
    _write_jsonl(data / RUNTIME_FILES["resources"], resources)
    _write_jsonl(data / RUNTIME_FILES["resource_provenance"], [
        {"resource_id": "github:1"}, {"resource_id": "github:2"},
    ])
    _write_json(data / RUNTIME_FILES["resource_manifest"], {
        "build_id": TEST_IDENTITY.resource_build_id,
        "final_resources": 2,
        "curation_manifest_hash": TEST_IDENTITY.curation_hash,
    })
    _write_jsonl(data / RUNTIME_FILES["documents"], documents)
    _write_jsonl(data / RUNTIME_FILES["chunks"], chunks)
    _write_json(data / RUNTIME_FILES["evidence_manifest"], {
        "evidence_build_id": TEST_IDENTITY.evidence_build_id,
        "input_resource_build_id": TEST_IDENTITY.resource_build_id,
        "document_count": 2,
        "chunk_count": 2,
    })
    return data


def test_rejects_wrong_resource_build_id(tmp_path: Path) -> None:
    source = make_frozen(tmp_path / "source")
    manifest = source / RUNTIME_FILES["resource_manifest"]
    value = json.loads(manifest.read_text())
    value["build_id"] = "build_ffffffffffffffff"
    _write_json(manifest, value)
    with pytest.raises(ValueError, match="Resource Build ID"):
        validate_frozen_root(source, TEST_IDENTITY)


def test_rejects_wrong_evidence_build_id(tmp_path: Path) -> None:
    source = make_frozen(tmp_path / "source")
    manifest = source / RUNTIME_FILES["evidence_manifest"]
    value = json.loads(manifest.read_text())
    value["evidence_build_id"] = "evidence_build_ffffffffffffffff"
    _write_json(manifest, value)
    with pytest.raises(ValueError, match="Evidence Build ID"):
        validate_frozen_root(source, TEST_IDENTITY)


def test_rejects_wrong_counts(tmp_path: Path) -> None:
    source = make_frozen(tmp_path / "source")
    path = source / RUNTIME_FILES["chunks"]
    path.write_text(path.read_text().splitlines()[0] + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="record counts mismatch"):
        validate_frozen_root(source, TEST_IDENTITY)


def test_rejects_incomplete_source(tmp_path: Path) -> None:
    source = make_frozen(tmp_path / "source")
    (source / RUNTIME_FILES["resource_provenance"]).unlink()
    with pytest.raises(FileNotFoundError, match="incomplete"):
        validate_frozen_root(source, TEST_IDENTITY)


def test_copies_only_required_runtime_artifacts_and_preserves_bytes(tmp_path: Path) -> None:
    source = make_frozen(tmp_path / "source")
    target = tmp_path / "target"
    result = materialize_runtime_data(source, project_root=target, expected=TEST_IDENTITY)
    assert result["status"] == "materialized"
    assert set(result["copied"]) == set(RUNTIME_FILES)
    for name, relative in RUNTIME_FILES.items():
        assert (target / "data" / relative).read_bytes() == (source / relative).read_bytes(), name


def test_second_run_is_idempotent_no_op(tmp_path: Path) -> None:
    source = make_frozen(tmp_path / "source")
    target = tmp_path / "target"
    materialize_runtime_data(source, project_root=target, expected=TEST_IDENTITY)
    result = materialize_runtime_data(source, project_root=target, expected=TEST_IDENTITY)
    assert result["status"] == "already_materialized"
    assert result["copied"] == []
    assert set(result["already_matching"]) == set(RUNTIME_FILES)


def test_matching_existing_canonical_files_are_preserved(tmp_path: Path) -> None:
    source = make_frozen(tmp_path / "source")
    target = tmp_path / "target"
    materialize_runtime_data(source, project_root=target, expected=TEST_IDENTITY)
    before = {
        name: (target / "data" / relative).stat().st_mtime_ns
        for name, relative in RUNTIME_FILES.items()
    }
    materialize_runtime_data(source, project_root=target, expected=TEST_IDENTITY)
    after = {
        name: (target / "data" / relative).stat().st_mtime_ns
        for name, relative in RUNTIME_FILES.items()
    }
    assert before == after


def test_conflicting_canonical_identity_fails_before_copy(tmp_path: Path) -> None:
    source = make_frozen(tmp_path / "source")
    target = tmp_path / "target"
    conflict = target / "data" / RUNTIME_FILES["resource_manifest"]
    _write_json(conflict, {"build_id": "build_conflict"})
    with pytest.raises(FileExistsError, match="Canonical dataset conflict"):
        materialize_runtime_data(source, project_root=target, expected=TEST_IDENTITY)
    assert not (target / "data" / RUNTIME_FILES["resources"]).exists()


def test_rejects_orphan_chunk_identity(tmp_path: Path) -> None:
    source = make_frozen(tmp_path / "source")
    chunks = source / RUNTIME_FILES["chunks"]
    values = [json.loads(line) for line in chunks.read_text().splitlines()]
    values[0]["resource_id"] = "github:999"
    _write_jsonl(chunks, values)
    with pytest.raises(ValueError, match="orphan identities"):
        validate_frozen_root(source, TEST_IDENTITY)


def test_resource_only_materialization_leaves_existing_evidence_untouched(tmp_path: Path) -> None:
    source = make_frozen(tmp_path / "source")
    target = tmp_path / "target"
    evidence_target = target / "data" / RUNTIME_FILES["documents"]
    evidence_target.parent.mkdir(parents=True)
    evidence_target.write_text("existing user-owned Evidence\n", encoding="utf-8")
    result = materialize_runtime_data(
        source, project_root=target, expected=TEST_IDENTITY, components="resources",
    )
    assert result["components"] == "resources"
    assert set(result["copied"]) == {
        "resources", "resource_provenance", "resource_manifest",
    }
    assert evidence_target.read_text(encoding="utf-8") == "existing user-owned Evidence\n"


def test_canonical_runtime_outputs_remain_git_ignored() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = [str(Path("data") / relative) for relative in RUNTIME_FILES.values()]
    result = subprocess.run(
        ["git", "-c", "core.quotePath=false", "check-ignore", "--stdin", "-z"],
        cwd=root, check=True,
        input="\0".join(paths) + "\0",
        capture_output=True, text=True, encoding="utf-8",
    )
    actual = {item.replace("\\", "/") for item in result.stdout.split("\0") if item}
    expected = {item.replace("\\", "/") for item in paths}
    assert actual == expected
