from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from data_pipeline.collectors.github_collector import stable_hash
from data_pipeline.evidence.chunker import ChunkSpan, LIST_ITEM_RE, chunk_sections
from data_pipeline.evidence.config import EvidenceConfig, load_config
from data_pipeline.evidence.core import (
    OUTPUTS,
    _atomic_publish,
    _duplicate_candidates,
    _quality,
    _span_quality_counts,
    build_corpus,
    chunks_for_document,
    digest,
    evidence_build_id_for,
    serialize_metadata,
    text_hash,
    validate_records,
)
from data_pipeline.evidence.loader import load_evidence_documents
from data_pipeline.evidence.markdown_parser import Section, parse_markdown
from data_pipeline.evidence.normalizer import normalize_markdown
from data_pipeline.evidence.schema import EvidenceChunk, EvidenceDocument
from data_pipeline.evidence.validate import validate_canonical
from data_pipeline.resource_dataset.schema import Resource


STAMP = "2026-08-29T08:51:25Z"


def config_for(tmp_path: Path, **chunking: int) -> EvidenceConfig:
    raw = copy.deepcopy(load_config().raw)
    raw["chunking"].update(chunking)
    raw["review"]["oversized_chunk_chars"] = max(
        int(raw["review"]["oversized_chunk_chars"]), int(raw["chunking"]["max_chunk_chars"]),
    )
    raw["paths"]["staging_root"] = str(tmp_path / "staging")
    return EvidenceConfig(raw=raw, config_path=tmp_path / "evidence.yaml")


def resource(repo_id: int, full_name: str) -> Resource:
    name = full_name.split("/")[-1]
    return Resource(
        resource_id=f"github:{repo_id}", resource_type="github_project", name=name,
        canonical_name=name, full_name=full_name, description="machine learning project",
        url=f"https://github.com/{full_name}", canonical_url=f"https://github.com/{full_name}",
        source_platform="github", source_id=f"github:{repo_id}", source_type="community",
        source_reliability=0.7, source_reliability_method="heuristic_v0",
        owner_name=full_name.split("/")[0], owner_type="user", organization_name=None,
        language="Python", topics=["machine-learning"], frameworks=[], skills=[], prerequisites=[],
        difficulty="unknown", learning_style=[], tasks=[], stars=1, forks=0, watchers=1,
        open_issues=0, license="MIT", is_archived=False, is_fork=False,
        created_at=STAMP, updated_at=STAMP, pushed_at=STAMP, activity_score=1.0,
        freshness_score=1.0, metadata_completeness=1.0, domain_relevance_score=1.0,
        data_quality_score=1.0, raw_source_id=full_name, retrieved_at=STAMP,
        record_version="v0.1.0",
    )


def document(text: str, config: EvidenceConfig, *, kind: str = "github_readme", build_id: str = "evidence_build_0123456789abcdef") -> EvidenceDocument:
    suffix = "readme" if kind == "github_readme" else "metadata"
    source_id = f"github:1:{suffix}"
    content = text_hash(text)
    return EvidenceDocument(
        evidence_id=f"ev_{digest(['github:1', kind, content])[:16]}", document_key=source_id,
        evidence_build_id=build_id, resource_id="github:1", evidence_source_type=kind,
        source_type="community", source_id=source_id, raw_source_id="owner/repo",
        source_url=f"https://github.com/owner/repo/{suffix}", raw_source_hash="a" * 64,
        title="title", text=text, text_format="markdown" if suffix == "readme" else "structured_metadata",
        content_hash=content, retrieved_at=STAMP, published_at=None,
        updated_at=STAMP if suffix == "metadata" else None, source_reliability=0.7,
        source_reliability_method="heuristic_v0",
        extraction_method="raw_snapshot" if suffix == "readme" else "api_metadata_serialization",
        run_id="run", collector_version="0.1.0", snapshot_schema_version="1.0.0",
        evidence_schema_version="0.1.0", normalizer_version="v0.1.0",
        markdown_parser_version="v0.1.0", metadata={
            "raw_snapshot_path": "fixture.json", "raw_provenance": {},
            "resource_discovery_provenance": {}, "resource_record_version": "v0.1.0",
            "normalization_warnings": [],
        },
    )


def write_fixture(tmp_path: Path, count: int = 10, *, missing_readme: bool = False) -> tuple[Path, Path, EvidenceConfig]:
    resources = [resource(index, f"owner/repo-{index}") for index in range(1, count + 1)]
    resources_path = tmp_path / "resources.parquet"
    pq.write_table(pa.Table.from_pylist([item.model_dump(mode="json") for item in resources]), resources_path)
    run_dir = tmp_path / "raw" / "run-123"
    (run_dir / "github_repository").mkdir(parents=True)
    (run_dir / "github_readme").mkdir(parents=True)
    for item in resources:
        repo_id = int(item.resource_id.split(":")[1])
        metadata_payload = {
            "id": repo_id, "full_name": item.full_name, "description": item.description,
            "html_url": item.url, "owner": {"login": "owner", "type": "User"},
            "language": "Python", "topics": ["z-topic", "a-topic"],
            "stargazers_count": 1, "forks_count": 0, "watchers_count": 1,
            "open_issues_count": 0, "license": {"spdx_id": "MIT"}, "archived": False,
            "fork": False, "default_branch": "main", "created_at": STAMP,
            "updated_at": STAMP, "pushed_at": STAMP,
        }
        readme_payload = {
            "api_response": {"path": "README.md", "sha": f"sha-{repo_id}", "download_url": f"https://raw.example/{repo_id}"},
            "decoded_text": f"# Repo {repo_id}\r\n\r\nMachine learning evidence for repository {repo_id}.\r\n",
        }
        for source_type, payload in (("github_repository", metadata_payload), ("github_readme", readme_payload)):
            if missing_readme and repo_id == count and source_type == "github_readme":
                continue
            wrapper = {
                "collector_version": "0.1.0", "schema_version": "1.0.0", "run_id": run_dir.name,
                "retrieved_at": STAMP, "source_type": source_type, "source_id": item.full_name,
                "source_url": item.url if source_type == "github_repository" else f"{item.url}/blob/main/README.md",
                "source_hash": stable_hash(payload),
                "provenance": {"query_used": "topic:machine-learning", "search_page": 1, "search_position": repo_id},
                "payload": payload,
            }
            target = run_dir / source_type / f"owner__repo-{repo_id}.json"
            target.write_text(json.dumps(wrapper), encoding="utf-8")
    manifest_path = tmp_path / "resource_manifest.json"
    manifest_path.write_text(json.dumps({
        "build_id": "build_fixture", "schema_version": "0.1.0", "normalizer_version": "v0.1.0",
        "dataset_scale": "validation_scale", "final_resources": count, "input_source_hashes": [],
    }), encoding="utf-8")
    config = config_for(tmp_path)
    config.raw["inputs"]["resource_manifest"] = str(manifest_path)
    return resources_path, run_dir, config


def test_document_schema_enforces_identity_and_forbids_extra_fields(tmp_path: Path) -> None:
    item = document("# Heading\nBody", config_for(tmp_path))
    invalid = item.model_dump()
    invalid["source_id"] = "owner/repo"
    with pytest.raises(ValidationError):
        EvidenceDocument.model_validate(invalid)
    invalid = item.model_dump()
    invalid["unexpected"] = True
    with pytest.raises(ValidationError):
        EvidenceDocument.model_validate(invalid)


def test_normalization_is_nfc_lf_and_preserves_fenced_code(tmp_path: Path) -> None:
    value = "Cafe\u0301\r\n\r\n\r\n\r\n```text\r\n\r\n\r\n\r\ncode\r\n```\r\n"
    normalized = normalize_markdown(value, config_for(tmp_path))
    assert normalized.startswith("Café\n\n\n```text")
    assert "```text\n\n\n\ncode\n```" in normalized
    assert "\r" not in normalized


def test_markdown_parser_distinguishes_setext_headings_and_horizontal_rules() -> None:
    text = "Title\n---\nbody\n\n---\n\n- list\n---\n\n```md\nInside\n---\n```\n\n| A |\n| --- |"
    parsed = parse_markdown(text)
    assert [(item.title, item.level) for item in parsed.sections] == [("Title", 2)]
    assert parsed.sections[0].start == 0
    assert parsed.sections[0].body_start == len("Title\n---\n")


def test_markdown_parser_tracks_hierarchy_offsets_and_unclosed_fence() -> None:
    text = "preamble\n# One\nbody\n## Two\n```\n# not heading"
    parsed = parse_markdown(text)
    assert [item.section_path for item in parsed.sections] == [["__preamble__"], ["One"], ["One", "Two"]]
    assert parsed.sections[1].start == text.index("# One")
    assert parsed.warnings == ["unclosed_fenced_code_block"]


def test_chunks_are_exact_contiguous_slices_with_boundary_overlap(tmp_path: Path) -> None:
    config = config_for(tmp_path, max_chunk_chars=90, overlap_chars=20, min_chunk_chars=25)
    text = "# Section\n" + "Sentence one has words. Sentence two has more words. " * 5
    item = document(text, config)
    chunks = chunks_for_document(item, config)
    assert len(chunks) > 1
    for chunk in chunks:
        assert item.text[chunk.source_start_char:chunk.source_end_char] == chunk.text
        assert chunk.text.strip()
    for left, right in zip(chunks, chunks[1:]):
        assert right.source_start_char <= left.source_end_char
        assert right.text[0].isalnum() or right.text[0] in "#`"


def test_code_and_table_are_not_split_or_overlapped(tmp_path: Path) -> None:
    config = config_for(tmp_path, max_chunk_chars=80, overlap_chars=20, min_chunk_chars=20)
    code = "```python\n" + "print('long line')\n" * 8 + "```\n"
    table = "| A | B |\n| --- | --- |\n" + "| x | y |\n" * 10
    text = "# Blocks\nintro sentence with enough words.\n\n" + code + "\n" + table
    chunks = chunks_for_document(document(text, config), config)
    code_chunks = [item for item in chunks if item.contains_code]
    table_chunks = [item for item in chunks if item.contains_table]
    assert len(code_chunks) == 1 and code in code_chunks[0].text
    assert len(table_chunks) == 1 and table.strip() in table_chunks[0].text
    assert sum(code in item.text for item in chunks) == 1


def test_identical_text_at_different_positions_is_retained_for_review(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    text = "# First\nSame content.\n# Second\nSame content."
    chunks = chunks_for_document(document(text, config), config)
    assert len(chunks) == 2
    # Simulate identical canonical slices at distinct positions without allowing generation-artifact removal.
    records = [chunk.model_copy(update={"content_hash": "sha256:" + "f" * 64}) for chunk in chunks]
    assert len(_duplicate_candidates(records)) == 1


def test_metadata_serialization_uses_allow_list_order_and_sorted_topics() -> None:
    text = serialize_metadata({"id": 1, "full_name": "o/r", "topics": ["z", "a"], "secret": "omit"})
    assert text.splitlines()[0] == "repository_id: 1"
    assert 'topics: ["a","z"]' in text
    assert "secret" not in text
    assert text.splitlines()[-1] == "pushed_at: null"


def test_build_identity_ignores_machine_paths_and_config_serialization(tmp_path: Path) -> None:
    resources = [resource(1, "owner/repo")]
    manifest = {"build_id": "build_x", "schema_version": "0.1", "normalizer_version": "v1", "dataset_scale": "validation", "input_source_hashes": ["a"]}
    left = config_for(tmp_path / "left")
    right = config_for(tmp_path / "right")
    left.raw["paths"]["evidence_root"] = "D:/machine-a/evidence"
    right.raw["paths"]["evidence_root"] = "E:/machine-b/evidence"
    assert evidence_build_id_for(resources, manifest, "run", ["raw"], left)[0] == evidence_build_id_for(resources, manifest, "run", ["raw"], right)[0]


def test_validation_rejects_non_contiguous_or_tampered_chunk(tmp_path: Path) -> None:
    config = config_for(tmp_path)
    doc = document("# Heading\nBody text", config)
    chunk = chunks_for_document(doc, config)[0]
    manifest = {"evidence_build_id": doc.evidence_build_id, "input_resource_count": 1}
    bad = chunk.model_copy(update={"text": "tampered", "char_count": 8})
    with pytest.raises(ValueError, match="source slice"):
        validate_records([doc], [bad], manifest)


def test_atomic_publish_rolls_back_all_replaced_files(tmp_path: Path) -> None:
    root, stage = tmp_path / "root", tmp_path / "stage"
    for relative in OUTPUTS:
        target, source = root / relative, stage / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        source.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("old", encoding="utf-8")
        source.write_text("new", encoding="utf-8")
    calls = 0

    def failing_replace(source: str | os.PathLike[str], target: str | os.PathLike[str]) -> None:
        nonlocal calls
        calls += 1
        if calls == 4:
            raise OSError("publish failure")
        os.replace(source, target)

    with pytest.raises(OSError, match="publish failure"):
        _atomic_publish(stage, root, failing_replace)
    assert all((root / relative).read_text(encoding="utf-8") == "old" for relative in OUTPUTS)


def test_ten_resource_offline_integration_and_deterministic_rebuild(tmp_path: Path) -> None:
    resources_path, run_dir, config = write_fixture(tmp_path, 10)
    output = tmp_path / "published"
    first = build_corpus(resources_path, run_dir, config, output_root=output)
    first_documents = (output / "data/evidence/documents.jsonl").read_bytes()
    first_chunks = (output / "data/evidence/chunks.jsonl").read_bytes()
    second = build_corpus(resources_path, run_dir, config, output_root=output)
    assert first["evidence_build_id"] == second["evidence_build_id"]
    assert first["documents"] == 20
    assert (output / "data/evidence/documents.jsonl").read_bytes() == first_documents
    assert (output / "data/evidence/chunks.jsonl").read_bytes() == first_chunks
    docs = [json.loads(line) for line in first_documents.decode().splitlines()]
    chunks = [json.loads(line) for line in first_chunks.decode().splitlines()]
    assert all(item["source_id"] == item["document_key"] for item in docs)
    assert all(item["chunker_version"] == "v0.1.1" for item in chunks)
    parent = {item["evidence_id"]: item for item in docs}
    assert all(parent[item["evidence_id"]]["text"][item["source_start_char"]:item["source_end_char"]] == item["text"] for item in chunks)


def test_isolated_output_root_and_explicit_resource_manifest(tmp_path: Path) -> None:
    resources_path, run_dir, fixture = write_fixture(tmp_path, 10)
    output = tmp_path / "isolated"
    config = load_config(
        output_root=output,
        resource_manifest=Path(fixture.raw["inputs"]["resource_manifest"]),
    )
    result = build_corpus(resources_path, run_dir, config, output_root=output)
    manifest = json.loads((output / "data/metadata/evidence_build_manifest.json").read_text(encoding="utf-8"))
    assert result["documents"] == 20
    assert manifest["input_resource_build_id"] == "build_fixture"
    assert manifest["code_version"] == "summer_s3_5_p0"
    assert "git_commit" in manifest
    assert validate_canonical(resources_path, output)["valid"] is True


def test_missing_readme_is_reported_without_synthetic_evidence(tmp_path: Path) -> None:
    resources_path, run_dir, config = write_fixture(tmp_path, 10, missing_readme=True)
    output = tmp_path / "published"
    result = build_corpus(resources_path, run_dir, config, output_root=output)
    assert result["documents"] == 19
    missing = (output / "data/review/evidence/missing_expected_sources.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(missing) == 1 and json.loads(missing[0])["missing"] == "github_readme"


def test_preflight_failure_does_not_replace_existing_corpus(tmp_path: Path) -> None:
    resources_path, run_dir, config = write_fixture(tmp_path, 10)
    output = tmp_path / "published"
    marker = output / "data/evidence/documents.jsonl"
    marker.parent.mkdir(parents=True)
    marker.write_text("last-valid", encoding="utf-8")
    manifest_path = Path(config.raw["inputs"]["resource_manifest"])
    bad_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bad_manifest["final_resources"] = 11
    manifest_path.write_text(json.dumps(bad_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest"):
        build_corpus(resources_path, run_dir, config, output_root=output)
    assert marker.read_text(encoding="utf-8") == "last-valid"


def test_loader_rejects_staging_without_canonical_commit_marker(tmp_path: Path) -> None:
    documents = tmp_path / "staging" / "documents.jsonl"
    documents.parent.mkdir(parents=True)
    documents.write_text("", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="manifest"):
        load_evidence_documents(documents, manifest_path=tmp_path / "staging" / "manifest.json")


def test_heading_empty_and_structural_only_sections_emit_no_chunks(tmp_path: Path) -> None:
    config = config_for(tmp_path, max_chunk_chars=120, overlap_chars=20, min_chunk_chars=30)
    text = "<div align=\"center\">\n\n# Heading only\n\n## Structural only\n\n<!-- layout marker -->\n\n---\n\n## Content\n\nUseful evidence body."
    chunks = chunks_for_document(document(text, config), config)
    assert len(chunks) == 1
    assert chunks[0].text == "Useful evidence body."
    assert chunks[0].section_path == ["Heading only", "Content"]
    assert not chunks[0].text.startswith("#")


def test_short_prose_prefix_is_packed_with_adjacent_atomic_code(tmp_path: Path) -> None:
    config = config_for(tmp_path, max_chunk_chars=90, overlap_chars=10, min_chunk_chars=30)
    code = "```python\n" + "print('evidence')\n" * 8 + "```\n"
    text = "# Example\nand:\n\n" + code
    chunks = chunks_for_document(document(text, config), config)
    assert len(chunks) == 1
    assert chunks[0].text.startswith("and:\n\n```python")
    assert chunks[0].contains_code
    assert text[chunks[0].source_start_char:chunks[0].source_end_char] == chunks[0].text


@pytest.mark.parametrize("marker", ["-", "*", "+", "1.", "1)"])
def test_list_overlap_starts_at_complete_generic_item_boundary(tmp_path: Path, marker: str) -> None:
    config = config_for(tmp_path, max_chunk_chars=190, overlap_chars=80, min_chunk_chars=30)
    items = [f"  {marker} [Item {index}](https://example.com/{index}) with complete supporting evidence text.\n" for index in range(8)]
    text = "# Links\n" + "".join(items)
    chunks = chunks_for_document(document(text, config), config)
    overlaps = [item for item in chunks if item.metadata["overlap_applied"]]
    assert overlaps
    for chunk in overlaps:
        first_line = chunk.text.splitlines()[0]
        assert LIST_ITEM_RE.match(first_line)
        assert first_line.startswith(f"  {marker} [")


def test_prose_overlap_never_begins_inside_inline_link(tmp_path: Path) -> None:
    config = config_for(tmp_path, max_chunk_chars=170, overlap_chars=70, min_chunk_chars=30)
    paragraph = (
        "Opening context has enough useful words. "
        "Read [AWS Least Privilege for Distributed Development](https://example.com/least-privilege) "
        "for a complete security example. Additional evidence follows with several ordinary sentences. "
    ) * 4
    text = "# Security\n" + paragraph
    chunks = chunks_for_document(document(text, config), config)
    link_spans = [(match.start(), match.end()) for match in __import__("re").finditer(r"!?\[[^\]\n]*\]\([^\)\n]*\)", text)]
    assert any(item.metadata["overlap_applied"] for item in chunks)
    assert all(not any(start < item.source_start_char < end for start, end in link_spans) for item in chunks)


def test_overlap_has_substantive_new_nonwhitespace_coverage_and_no_containment(tmp_path: Path) -> None:
    config = config_for(tmp_path, max_chunk_chars=180, overlap_chars=50, min_chunk_chars=35)
    text = "# Coverage\n" + "Each sentence contributes substantive new source evidence for retrieval. " * 12
    item = document(text, config)
    chunks = chunks_for_document(item, config)
    counts = _span_quality_counts([item], chunks, 35, 180)
    assert counts["fully_contained_chunk_spans"] == 0
    assert counts["chunks_with_insufficient_new_coverage"] == 0
    assert any(chunk.metadata["overlap_applied"] for chunk in chunks)
    assert all(
        not chunk.metadata["overlap_applied"] or chunk.metadata["new_nonwhitespace_coverage"] >= 35
        for chunk in chunks
    )


def test_validator_rejects_fully_contained_same_section_span(tmp_path: Path) -> None:
    config = config_for(tmp_path, max_chunk_chars=200, overlap_chars=30, min_chunk_chars=20)
    item = document("# Section\n" + "Evidence words. " * 20, config)
    chunks = chunks_for_document(item, config)
    original = chunks[0]
    start, end = original.source_start_char + 1, original.source_end_char
    contained = original.model_copy(update={
        "chunk_index": 1,
        "source_start_char": start,
        "source_start_line": 2,
        "text": item.text[start:end],
        "char_count": end - start,
        "content_hash": text_hash(item.text[start:end]),
        "chunk_id": f"ch_{digest([item.evidence_id, original.section_path, 1, start, end, item.text[start:end]])[:16]}",
    })
    manifest = {
        "evidence_build_id": item.evidence_build_id, "input_resource_count": 1,
        "min_chunk_chars": 20, "max_chunk_chars": 200,
    }
    with pytest.raises(ValueError, match="fully-contained"):
        validate_records([item], [original, contained], manifest)


def test_validator_rejects_overlap_with_insufficient_new_nonwhitespace(tmp_path: Path) -> None:
    config = config_for(tmp_path, max_chunk_chars=220, overlap_chars=40, min_chunk_chars=30)
    item = document("# Section\n" + "substantive evidence words " * 20, config)
    original = chunks_for_document(item, config)[0]
    start = original.source_end_char - 20
    end = original.source_end_char + 10
    overlap_text = item.text[start:end]
    overlapping = original.model_copy(update={
        "chunk_index": 1,
        "source_start_char": start,
        "source_end_char": end,
        "source_start_line": 2,
        "source_end_line": 2,
        "text": overlap_text,
        "char_count": len(overlap_text),
        "content_hash": text_hash(overlap_text),
        "chunk_id": f"ch_{digest([item.evidence_id, original.section_path, 1, start, end, overlap_text])[:16]}",
    })
    manifest = {
        "evidence_build_id": item.evidence_build_id, "input_resource_count": 1,
        "min_chunk_chars": 30, "max_chunk_chars": 220,
    }
    with pytest.raises(ValueError, match="insufficient new non-whitespace"):
        validate_records([item], [original, overlapping], manifest)


def test_oversized_ordinary_prose_is_split_but_atomic_table_is_preserved(tmp_path: Path) -> None:
    config = config_for(tmp_path, max_chunk_chars=140, overlap_chars=20, min_chunk_chars=30)
    prose = "Ordinary prose includes safe words and sentence boundaries for deterministic splitting. " * 12
    table = "| A | B |\n| --- | --- |\n" + "| long value | evidence |\n" * 12
    text = "# Prose\n" + prose + "\n# Table\n" + table
    chunks = chunks_for_document(document(text, config), config)
    assert all(item.char_count <= 140 for item in chunks if not item.contains_code and not item.contains_table)
    table_chunks = [item for item in chunks if item.contains_table]
    assert len(table_chunks) == 1
    assert table.strip() in table_chunks[0].text


def test_generation_artifact_is_removed_but_distinct_identical_text_remains(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = config_for(tmp_path)
    text = "same text\n\nsame text"
    item = document(text, config)
    section = Section(["__preamble__"], "__preamble__", 0, 0, len(text), 0)
    duplicate = ChunkSpan(section, 0, 9, False, False)
    distinct = ChunkSpan(section, 11, 20, False, False)
    monkeypatch.setattr("data_pipeline.evidence.core.chunk_sections", lambda *_: [duplicate, duplicate, distinct])
    chunks = chunks_for_document(item, config)
    assert len(chunks) == 2
    assert [chunk.text for chunk in chunks] == ["same text", "same text"]
    assert chunks[0].source_start_char != chunks[1].source_start_char


def test_quality_reports_distribution_and_new_chunk_invariants(tmp_path: Path) -> None:
    config = config_for(tmp_path, max_chunk_chars=180, overlap_chars=30, min_chunk_chars=25)
    item = document("# Report\n" + "Quality evidence sentence. " * 30, config)
    chunks = chunks_for_document(item, config)
    manifest = {
        "evidence_build_id": item.evidence_build_id, "dataset_scale": "validation_scale",
        "input_resource_count": 1, "min_chunk_chars": 25, "max_chunk_chars": 180,
    }
    validation = validate_records([item], chunks, manifest)
    quality = _quality([item], chunks, manifest, [], [], [], [], [], [], [], validation)
    assert quality["chunk_characters"]["median"] is not None
    assert quality["chunk_characters"]["p95"] is not None
    assert quality["chunks_below_min_chunk_chars"] == 0
    assert quality["fully_contained_chunk_spans"] == 0
    assert quality["chunks_with_insufficient_new_coverage"] == 0
    assert quality["known_limitations"][0] == "Corpus from 1 public GitHub repositories."
