"""Transfer native styles through one shared, occurrence-aware alignment."""

from __future__ import annotations

from docparse.textalign import align_characters, markup_intervals, normalized_view
from docparse.textlayer import Line

_ATTRS = ("bold", "italic", "underline", "strike", "sup", "mark")
_OPEN = {"bold": "**", "italic": "*", "underline": "<u>", "strike": "~~", "sup": "<sup>", "mark": "<mark>"}
_CLOSE = {**_OPEN, "underline": "</u>", "sup": "</sup>", "mark": "</mark>"}


def _native_styles(lines: list[Line]) -> tuple[str, list[tuple[int, int, str]]]:
    chars = []
    styles = []
    for line in lines:
        if chars and line.spans:
            chars.append(" ")
            next_styles = {a for a in _ATTRS if getattr(line.spans[0], a)}
            styles.append(styles[-1] & next_styles)
        for span in line.spans:
            attrs = {a for a in _ATTRS if getattr(span, a)}
            chars.extend(span.text)
            styles.extend([attrs] * len(span.text))
    ranges = []
    for attr in _ATTRS:
        start = None
        for index in range(len(chars) + 1):
            enabled = index < len(chars) and attr in styles[index]
            if enabled and start is None:
                start = index
            elif not enabled and start is not None:
                ranges.append((start, index, attr))
                start = None
    return "".join(chars), ranges


def transfer_aligned_styles(target: str, lines: list[Line]) -> str:
    source, native_ranges = _native_styles(lines)
    if not source or not native_ranges or max(len(source), len(target)) > 12000:
        return target
    hidden, existing, protected = markup_intervals(target)
    alignment = align_characters(source, target, hidden)
    if alignment.similarity < 0.6:
        return target
    unavailable = hidden | {i for a, b in protected for i in range(a, b)}
    unavailable.update(i for i, char in enumerate(target) if char == "\n")
    ranges = []
    for start, stop, attr in native_ranges:
        interval = alignment.target_interval(start, stop)
        if interval is None:
            continue
        a, b = interval
        # A whole native word may differ by a suffix in the VLM text. Preserve
        # its whole-word styling, while superscripts keep exact subword bounds.
        if attr != "sup":
            starts_word = start == 0 or not source[start - 1].isalnum()
            ends_word = stop == len(source) or not source[stop].isalnum()
            if starts_word:
                while a > 0 and target[a - 1].isalnum() and target[a].isalnum():
                    a -= 1
            if ends_word:
                while b < len(target) and target[b - 1].isalnum() and target[b].isalnum():
                    b += 1
        query = normalized_view(source[start:stop]).text.strip()
        if len(query) >= 2 and alignment.source_view.text.count(query) > 1:
            if alignment.source_view.text.count(query) != alignment.target_view.text.count(query):
                continue  # An omitted repeated occurrence has no reliable identity.
        # Split at existing syntax, so additions nest inside each container and
        # never duplicate a partially overlapping bold/italic run.
        first = None
        for index in range(a, b + 1):
            enabled = index < b and index not in unavailable
            if enabled and first is None:
                first = index
            elif not enabled and first is not None:
                start, stop = first, index
                while start < stop and target[start].isspace():
                    start += 1
                while stop > start and target[stop - 1].isspace():
                    stop -= 1
                if start < stop and not any(x <= start and stop <= y and style == attr for x, y, style in existing):
                    ranges.append((start, stop, attr))
                first = None
    if not ranges:
        return target
    boundaries = sorted({0, len(target), *(p for a, b, _ in ranges for p in (a, b))})
    out = []
    active = []
    for start, stop in zip(boundaries, boundaries[1:]):
        desired = [attr for attr in _ATTRS if any(a <= start < b and style == attr for a, b, style in ranges)]
        common = 0
        while common < min(len(active), len(desired)) and active[common] == desired[common]:
            common += 1
        out.extend(_CLOSE[attr] for attr in reversed(active[common:]))
        out.extend(_OPEN[attr] for attr in desired[common:])
        out.append(target[start:stop])
        active = desired
    out.extend(_CLOSE[attr] for attr in reversed(active))
    return "".join(out)
