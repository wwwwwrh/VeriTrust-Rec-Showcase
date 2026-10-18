"""Conservative, code-aware Markdown normalization."""

from __future__ import annotations

import re
import unicodedata

from .config import EvidenceConfig


FENCE_RE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")


def normalize_markdown(text: str, config: EvidenceConfig) -> str:
    text = unicodedata.normalize(str(config.normalization["unicode_form"]), text)
    if config.normalization["normalize_line_endings"]:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
    if not config.normalization["collapse_blank_lines"]:
        return text.strip()

    maximum = int(config.normalization["max_consecutive_blank_lines"])
    output: list[str] = []
    fence_char: str | None = None
    fence_length = 0
    blank_count = 0
    for line in text.split("\n"):
        match = FENCE_RE.match(line)
        if fence_char is not None:
            output.append(line)
            if match and match.group(1)[0] == fence_char and len(match.group(1)) >= fence_length:
                fence_char = None
                fence_length = 0
            continue
        if match:
            marker = match.group(1)
            fence_char, fence_length = marker[0], len(marker)
            blank_count = 0
            output.append(line)
            continue
        if not line.strip():
            blank_count += 1
            if blank_count <= maximum:
                output.append("")
            continue
        blank_count = 0
        output.append(line)
    return "\n".join(output).strip()
