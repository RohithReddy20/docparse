"""Monotone character alignment that retains source and target offsets."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher


@dataclass
class TextView:
    text: str
    positions: list[int]


def normalized_view(text: str, hidden: set[int] | None = None) -> TextView:
    chars = []
    positions = []
    for index, char in enumerate(text):
        if hidden and index in hidden:
            continue
        for normalized in unicodedata.normalize("NFKC", char).casefold():
            if normalized.isspace():
                normalized = " "
                if not chars or chars[-1] == " ":
                    continue
            chars.append(normalized)
            positions.append(index)
    return TextView("".join(chars), positions)


@dataclass
class CharacterAlignment:
    source: str
    target: str
    source_view: TextView
    target_view: TextView
    matches: dict[int, tuple[int, int]]
    similarity: float

    def target_interval(self, start: int, stop: int) -> tuple[int, int] | None:
        significant = [i for i in range(start, stop) if not self.source[i].isspace()]
        if not significant or significant[0] not in self.matches or significant[-1] not in self.matches:
            return None
        matched = [self.matches[i] for i in significant if i in self.matches]
        if len(matched) / len(significant) < 0.85:
            return None
        return min(a for a, _ in matched), max(b for _, b in matched)


def align_characters(source: str, target: str, hidden: set[int] | None = None) -> CharacterAlignment:
    left = normalized_view(source)
    right = normalized_view(target, hidden)
    matcher = SequenceMatcher(None, left.text, right.text, autojunk=False)
    matches = {}
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            src = left.positions[block.a + offset]
            dst = right.positions[block.b + offset]
            old = matches.get(src)
            matches[src] = (min(old[0], dst), max(old[1], dst + 1)) if old else (dst, dst + 1)
    return CharacterAlignment(source, target, left, right, matches, matcher.ratio())


_TAGS = {"bold": "strong|b", "italic": "em|i", "strike": "del|s|strike",
         "sup": "sup", "underline": "u", "mark": "mark"}


def markup_intervals(text: str) -> tuple[set[int], list[tuple[int, int, str]], list[tuple[int, int]]]:
    """Existing style containers, hidden syntax, and immutable code/math regions."""
    hidden = set()
    existing = []
    protected = []
    for match in re.finditer(r"<[^>]+>", text):
        hidden.update(range(*match.span()))
    for attr, tags in _TAGS.items():
        pattern = rf"<({tags})\b[^>]*>(.*?)</\1\s*>"
        existing.extend((m.start(2), m.end(2), attr) for m in re.finditer(pattern, text, re.S | re.I))
    patterns = (
        (("bold", "italic"), r"(?<![\\*])\*\*\*(?!\*)(.+?)(?<!\*)\*\*\*(?!\*)"),
        (("bold",), r"(?<![\\*])\*\*(?!\*)(.+?)(?<!\*)\*\*(?!\*)"),
        (("italic",), r"(?<![\\*])\*(?!\*)([^\n]+?)(?<![\\*])\*(?!\*)"),
        (("bold", "italic"), r"(?<![\w\\_])___(?!_)(.+?)(?<!_)___(?![\w_])"),
        (("bold",), r"(?<![\w\\_])__(?!_)(.+?)(?<!_)__(?![\w_])"),
        (("italic",), r"(?<![\w\\_])_(?!_)([^\n]+?)(?<![\\_])_(?![\w_])"),
        (("strike",), r"(?<!\\)~~(.+?)~~"),
    )
    for attrs, pattern in patterns:
        for match in re.finditer(pattern, text):
            start, stop = match.span(1)
            hidden.update(range(match.start(), start))
            hidden.update(range(stop, match.end()))
            existing.extend((start, stop, attr) for attr in attrs)
    for pattern in (
        r"(`+)(?!`)[\s\S]*?(?<!`)\1(?!`)",
        r"\$\$[\s\S]*?\$\$", r"\$(?!\s)[^$\n]+?\$",
        r"\\\([\s\S]*?\\\)", r"\\\[[\s\S]*?\\\]",
        r"<(?:code|pre)\b[^>]*>[\s\S]*?</(?:code|pre)\s*>",
    ):
        protected.extend(m.span() for m in re.finditer(pattern, text))
    return hidden, existing, protected
