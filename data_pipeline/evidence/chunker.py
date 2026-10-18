"""Section-first Markdown chunking with exact contiguous source slices."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .config import EvidenceConfig
from .markdown_parser import FENCE_RE, Section, source_lines


TABLE_SEPARATOR_RE = re.compile(
    r"^[ \t]*\|?[ \t]*:?-{3,}:?[ \t]*(?:\|[ \t]*:?-{3,}:?[ \t]*)+\|?[ \t]*$"
)
LIST_ITEM_RE = re.compile(r"^[ \t]*(?:[-+*]|\d+[.)])[ \t]+")
STRUCTURAL_RE = re.compile(
    r"^[ \t]{0,3}(?:(?:\*[ \t]*){3,}|(?:-[ \t]*){3,}|(?:_[ \t]*){3,})[ \t]*$"
)
STRUCTURAL_HTML_RE = re.compile(
    r"^[ \t]*(?:<!--.*-->|</?(?:div|details|center|p|br|hr)(?:[ \t]+[^>]*)?>)[ \t]*$",
    re.IGNORECASE,
)
INLINE_LINK_RE = re.compile(r"!?\[[^\]\n]*\]\([^\)\n]*\)")


@dataclass(frozen=True)
class SourceBlock:
    kind: str
    start: int
    end: int

    @property
    def atomic(self) -> bool:
        return self.kind in {"code", "table"}


@dataclass(frozen=True)
class ChunkSpan:
    section: Section
    start: int
    end: int
    contains_code: bool
    contains_table: bool
    overlap_applied: bool = False
    new_nonwhitespace_coverage: int | None = None


def _nonwhitespace_count(text: str, start: int, end: int) -> int:
    return sum(not character.isspace() for character in text[start:end])


def _atomic_spans(text: str, start: int, end: int) -> list[SourceBlock]:
    lines = [line for line in source_lines(text) if line.end > start and line.start < end]
    spans: list[SourceBlock] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        fence = FENCE_RE.match(line.text)
        if fence:
            marker = fence.group(1)
            stop = index + 1
            while stop < len(lines):
                closing = FENCE_RE.match(lines[stop].text)
                if closing and closing.group(1)[0] == marker[0] and len(closing.group(1)) >= len(marker):
                    stop += 1
                    break
                stop += 1
            spans.append(SourceBlock("code", max(start, line.start), min(end, lines[min(stop, len(lines)) - 1].end)))
            index = stop
            continue
        if index + 1 < len(lines) and "|" in line.text and TABLE_SEPARATOR_RE.match(lines[index + 1].text):
            stop = index + 2
            while stop < len(lines) and "|" in lines[stop].text and lines[stop].text.strip():
                stop += 1
            spans.append(SourceBlock("table", max(start, line.start), min(end, lines[stop - 1].end)))
            index = stop
            continue
        index += 1
    return spans


def _plain_blocks(text: str, start: int, end: int) -> list[SourceBlock]:
    """Split a non-atomic source region into prose, list-item, and structural blocks."""
    lines = [line for line in source_lines(text) if line.end > start and line.start < end]
    result: list[SourceBlock] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        line_start, line_end = max(start, line.start), min(end, line.end)
        if not text[line_start:line_end].strip():
            index += 1
            continue
        if STRUCTURAL_RE.fullmatch(line.text) or STRUCTURAL_HTML_RE.fullmatch(line.text):
            result.append(SourceBlock("structural", line_start, line_end))
            index += 1
            continue
        if LIST_ITEM_RE.match(line.text):
            stop = index + 1
            while stop < len(lines):
                candidate = lines[stop]
                if (
                    not candidate.text.strip()
                    or LIST_ITEM_RE.match(candidate.text)
                    or STRUCTURAL_RE.fullmatch(candidate.text)
                    or STRUCTURAL_HTML_RE.fullmatch(candidate.text)
                ):
                    break
                stop += 1
            result.append(SourceBlock("list", line_start, min(end, lines[stop - 1].end)))
            index = stop
            continue
        stop = index + 1
        while stop < len(lines):
            candidate = lines[stop]
            if (
                not candidate.text.strip()
                or LIST_ITEM_RE.match(candidate.text)
                or STRUCTURAL_RE.fullmatch(candidate.text)
                or STRUCTURAL_HTML_RE.fullmatch(candidate.text)
            ):
                break
            stop += 1
        result.append(SourceBlock("prose", line_start, min(end, lines[stop - 1].end)))
        index = stop
    return result


def _source_blocks(text: str, start: int, end: int) -> list[SourceBlock]:
    atomic = _atomic_spans(text, start, end)
    result: list[SourceBlock] = []
    cursor = start
    for span in atomic:
        if cursor < span.start:
            result.extend(_plain_blocks(text, cursor, span.start))
        result.append(span)
        cursor = span.end
    if cursor < end:
        result.extend(_plain_blocks(text, cursor, end))
    return sorted(result, key=lambda item: (item.start, item.end, item.kind))


def _inside_inline_link(text: str, position: int, start: int, end: int) -> bool:
    # A window-local regex cannot see the opening bracket when its window starts
    # inside a link, so inspect the complete normalized document.
    return any(match.start() < position < match.end() for match in INLINE_LINK_RE.finditer(text))


def _safe_boundaries(text: str, start: int, end: int) -> list[int]:
    segment = text[start:end]
    positions: set[int] = set()
    for match in re.finditer(r"\n", segment):
        positions.add(start + match.end())
    for match in re.finditer(r"[.!?。！？](?:[\"')\]]*)[ \t]+", segment):
        positions.add(start + match.end())
    for match in re.finditer(r"[ \t]+", segment):
        positions.add(start + match.end())
    return sorted(position for position in positions if start < position < end and not _inside_inline_link(text, position, start, end))


def _split_non_atomic(text: str, block: SourceBlock, maximum: int, minimum: int) -> list[SourceBlock]:
    if block.end - block.start <= maximum:
        return [block]
    result: list[SourceBlock] = []
    cursor = block.start
    while block.end - cursor > maximum:
        limit = cursor + maximum
        candidates = [position for position in _safe_boundaries(text, cursor, limit + 1) if position <= limit]
        viable = [
            position for position in candidates
            if _nonwhitespace_count(text, cursor, position) >= minimum
            and _nonwhitespace_count(text, position, block.end) >= minimum
        ]
        target = viable[-1] if viable else (candidates[-1] if candidates else limit)
        if target <= cursor:
            target = limit
        result.append(SourceBlock(block.kind, cursor, target))
        cursor = target
        while cursor < block.end and text[cursor].isspace() and cursor + 1 < block.end:
            cursor += 1
    if text[cursor:block.end].strip():
        result.append(SourceBlock(block.kind, cursor, block.end))
    return result


def _flags(blocks: list[SourceBlock], start: int, end: int) -> tuple[bool, bool]:
    hits = [block for block in blocks if block.start < end and block.end > start]
    return any(block.kind == "code" for block in hits), any(block.kind == "table" for block in hits)


def _base_chunks(
    text: str, section: Section, blocks: list[SourceBlock], maximum: int, overlap: int, minimum: int,
) -> list[ChunkSpan]:
    content = [block for block in blocks if block.kind != "structural"]
    if not content:
        return []
    payload_limit = maximum - overlap if overlap and maximum - overlap >= minimum else maximum
    units: list[SourceBlock] = []
    for block in content:
        units.extend([block] if block.atomic else _split_non_atomic(text, block, payload_limit, minimum))

    packed: list[ChunkSpan] = []
    current_start = units[0].start
    current_end = units[0].end
    current_code, current_table = _flags(blocks, current_start, current_end)
    for unit in units[1:]:
        proposed_length = unit.end - current_start
        current_is_short = _nonwhitespace_count(text, current_start, current_end) < minimum
        combine_with_atomic = unit.atomic and current_is_short
        if proposed_length <= payload_limit or combine_with_atomic:
            current_end = unit.end
            current_code, current_table = _flags(blocks, current_start, current_end)
            continue
        packed.append(ChunkSpan(section, current_start, current_end, current_code, current_table))
        current_start, current_end = unit.start, unit.end
        current_code, current_table = _flags(blocks, current_start, current_end)
    packed.append(ChunkSpan(section, current_start, current_end, current_code, current_table))

    index = 0
    while len(packed) > 1 and index < len(packed):
        item = packed[index]
        if _nonwhitespace_count(text, item.start, item.end) >= minimum:
            index += 1
            continue
        previous = packed[index - 1] if index else None
        following = packed[index + 1] if index + 1 < len(packed) else None
        if previous and (
            item.end - previous.start <= payload_limit
            or previous.contains_code or previous.contains_table or item.contains_code or item.contains_table
        ):
            code, table = _flags(blocks, previous.start, item.end)
            packed[index - 1:index + 1] = [ChunkSpan(section, previous.start, item.end, code, table)]
            index = max(0, index - 1)
            continue
        if following and (
            following.end - item.start <= payload_limit
            or following.contains_code or following.contains_table or item.contains_code or item.contains_table
        ):
            code, table = _flags(blocks, item.start, following.end)
            packed[index:index + 2] = [ChunkSpan(section, item.start, following.end, code, table)]
            continue
        index += 1
    return packed


def _prose_overlap_start(text: str, previous_start: int, boundary: int, overlap: int, list_blocks: list[SourceBlock]) -> int | None:
    desired = max(previous_start, boundary - overlap)
    candidates = _safe_boundaries(text, desired, boundary)
    for position in candidates:
        if position < desired or position >= boundary:
            continue
        if any(block.start < position < block.end for block in list_blocks):
            continue
        return position
    return None


def _list_overlap_start(boundary: int, overlap: int, list_blocks: list[SourceBlock]) -> int | None:
    candidates = [
        block.start for block in list_blocks
        if block.end <= boundary and boundary - block.start <= overlap
    ]
    return min(candidates) if candidates else None


def _new_nonwhitespace_coverage(text: str, start: int, end: int, covered: list[tuple[int, int]]) -> int:
    count = 0
    for position in range(start, end):
        if text[position].isspace():
            continue
        if not any(left <= position < right for left, right in covered):
            count += 1
    return count


def chunk_sections(text: str, sections: list[Section], config: EvidenceConfig) -> list[ChunkSpan]:
    maximum = int(config.chunking["max_chunk_chars"])
    overlap = int(config.chunking["overlap_chars"])
    minimum = int(config.chunking["min_chunk_chars"])
    result: list[ChunkSpan] = []

    for section in sections:
        body_start = section.body_start
        if not text[body_start:section.end].strip():
            continue
        blocks = _source_blocks(text, body_start, section.end)
        base = _base_chunks(text, section, blocks, maximum, overlap, minimum)
        if not base:
            continue
        list_blocks = [block for block in blocks if block.kind == "list"]
        emitted: list[tuple[int, int]] = []
        for index, item in enumerate(base):
            start = item.start
            overlap_applied = False
            if index and overlap and not (item.contains_code or item.contains_table):
                previous = base[index - 1]
                if not (previous.contains_code or previous.contains_table):
                    starts_in_list = any(block.start == item.start for block in list_blocks)
                    candidate = (
                        _list_overlap_start(item.start, overlap, list_blocks)
                        if starts_in_list
                        else _prose_overlap_start(text, previous.start, item.start, overlap, list_blocks)
                    )
                    if candidate is not None and candidate < item.start and item.end - candidate <= maximum:
                        proposed_new = _new_nonwhitespace_coverage(text, candidate, item.end, emitted)
                        if proposed_new >= minimum:
                            start = candidate
                            overlap_applied = True
            new_coverage = _new_nonwhitespace_coverage(text, start, item.end, emitted)
            code, table = _flags(blocks, start, item.end)
            result.append(ChunkSpan(section, start, item.end, code, table, overlap_applied, new_coverage))
            emitted.append((start, item.end))
    return result


def line_number(text: str, offset: int, *, end: bool = False) -> int:
    point = max(0, offset - 1) if end else offset
    return text.count("\n", 0, point) + 1
