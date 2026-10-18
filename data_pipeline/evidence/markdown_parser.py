"""Deterministic Markdown section parser with normalized-source offsets."""

from __future__ import annotations

import re
from dataclasses import dataclass


ATX_RE = re.compile(r"^[ \t]{0,3}(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
SETEXT_RE = re.compile(r"^[ \t]{0,3}(=+|-+)[ \t]*$")
FENCE_RE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")
BLOCK_PREFIX_RE = re.compile(r"^[ \t]*(?:[-+*][ \t]+|\d+[.)][ \t]+|>|#{1,6}[ \t]+|```|~~~)")


@dataclass(frozen=True)
class Line:
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class Section:
    section_path: list[str]
    title: str
    level: int
    start: int
    end: int
    body_start: int


@dataclass(frozen=True)
class ParseResult:
    sections: list[Section]
    warnings: list[str]


def source_lines(text: str) -> list[Line]:
    result: list[Line] = []
    offset = 0
    for item in text.splitlines(keepends=True):
        result.append(Line(item.rstrip("\r\n"), offset, offset + len(item)))
        offset += len(item)
    if offset < len(text) or not result:
        result.append(Line(text[offset:], offset, len(text)))
    return result


def _valid_setext_title(value: str) -> bool:
    stripped = value.strip()
    return bool(
        stripped
        and "|" not in stripped
        and not BLOCK_PREFIX_RE.match(value)
        and not re.fullmatch(r"[-=_* ]+", stripped)
        and not value.startswith("    ")
    )


def parse_markdown(text: str) -> ParseResult:
    lines = source_lines(text)
    headings: list[tuple[int, int, str, int, int]] = []
    warnings: list[str] = []
    fence_char: str | None = None
    fence_length = 0
    fenced_lines: set[int] = set()

    for index, line in enumerate(lines):
        match = FENCE_RE.match(line.text)
        if fence_char is not None:
            fenced_lines.add(index)
            if match and match.group(1)[0] == fence_char and len(match.group(1)) >= fence_length:
                fence_char = None
                fence_length = 0
            continue
        if match:
            marker = match.group(1)
            fence_char, fence_length = marker[0], len(marker)
            fenced_lines.add(index)
            continue
        atx = ATX_RE.match(line.text)
        if atx:
            headings.append((line.start, line.end, atx.group(2).strip(), len(atx.group(1)), index))
            continue
        setext = SETEXT_RE.match(line.text)
        if setext and index > 0 and index - 1 not in fenced_lines and _valid_setext_title(lines[index - 1].text):
            previous = lines[index - 1]
            headings.append((previous.start, line.end, previous.text.strip(), 1 if setext.group(1)[0] == "=" else 2, index - 1))
    if fence_char is not None:
        warnings.append("unclosed_fenced_code_block")

    headings.sort(key=lambda item: item[0])
    sections: list[Section] = []
    if not headings:
        if text.strip():
            sections.append(Section(["__preamble__"], "__preamble__", 0, 0, len(text), 0))
        return ParseResult(sections, warnings)
    first_start = headings[0][0]
    if text[:first_start].strip():
        sections.append(Section(["__preamble__"], "__preamble__", 0, 0, first_start, 0))
    stack: list[tuple[int, str]] = []
    for position, (start, heading_end, title, level, _) in enumerate(headings):
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        end = headings[position + 1][0] if position + 1 < len(headings) else len(text)
        sections.append(Section([item[1] for item in stack], title, level, start, end, heading_end))
    return ParseResult(sections, warnings)
