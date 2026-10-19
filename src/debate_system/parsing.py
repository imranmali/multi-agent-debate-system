"""Tolerant parsing of model replies.

Models are asked to reply with JSON but often wrap it in prose or Markdown
fences. :func:`extract_json` recovers the first JSON object it can find; the
numeric helpers turn loosely typed values into safe floats.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterator
from typing import Any

_FENCE_RE = re.compile(r"```[ \t]*(?:json|JSON)?[ \t]*\r?\n?(.*?)```", re.DOTALL)


def extract_json(text: str) -> dict[str, Any] | None:
    """Return the first JSON object in ``text``, or None if none parses.

    Tries, in order: the whole text, fenced ```` ```json ```` / ```` ``` ````
    blocks, then every balanced ``{...}`` span (string-aware, so braces inside
    string literals do not confuse the scan).
    """
    for candidate in _candidates(text):
        try:
            value = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _candidates(text: str) -> Iterator[str]:
    yield text.strip()
    for match in _FENCE_RE.finditer(text):
        yield match.group(1).strip()
    yield from _balanced_objects(text)


def _balanced_objects(text: str) -> Iterator[str]:
    """Yield each balanced ``{...}`` span, starting from every ``{`` in order."""
    start = text.find("{")
    while start != -1:
        end = _matching_brace(text, start)
        if end is not None:
            yield text[start : end + 1]
        start = text.find("{", start + 1)


def _matching_brace(text: str, start: int) -> int | None:
    """Index of the ``}`` closing the ``{`` at ``start``, ignoring braces in strings."""
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    return None


def clamp(value: float, low: float, high: float) -> float:
    """Limit ``value`` to the closed interval ``[low, high]``."""
    return max(low, min(high, value))


def parse_float(value: Any) -> float | None:
    """Return ``value`` as a finite float, or None if it is not numeric.

    Accepts ints, floats and numeric strings; booleans, NaN and infinities are
    rejected.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        result = float(value.strip() if isinstance(value, str) else value)
    except (ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def coerce_float(value: Any, default: float) -> float:
    """Return ``value`` as a finite float, or ``default`` if it is not numeric."""
    result = parse_float(value)
    return default if result is None else result
