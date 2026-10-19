"""Tests for tolerant JSON extraction and numeric coercion."""

from __future__ import annotations

import math
from typing import Any

import pytest

from debate_system.parsing import clamp, coerce_float, extract_json, parse_float


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"answer": "42"}', {"answer": "42"}),
        ('  \n{"a": 1}\n ', {"a": 1}),
        ('```json\n{"a": 1}\n```', {"a": 1}),
        ('Here you go:\n```JSON\n{"a": 1}\n```\nThanks', {"a": 1}),
        ('```\n{"a": 2}\n```', {"a": 2}),
        ('```json {"a": 3}```', {"a": 3}),
        (
            'Sure! My reply is {"a": 1, "b": {"c": [1, 2]}} - hope that helps.',
            {"a": 1, "b": {"c": [1, 2]}},
        ),
        ('{"s": "a } brace and a \\" quote {"} trailing', {"s": 'a } brace and a " quote {'}),
        ('{broken {"ok": true} done', {"ok": True}),
        ('first {"a": 1} then {"b": 2}', {"a": 1}),
        ('```json\n{not json}\n```\n{"fallback": 1}', {"fallback": 1}),
    ],
)
def test_extract_json_finds_first_object(text: str, expected: dict[str, Any]) -> None:
    assert extract_json(text) == expected


@pytest.mark.parametrize(
    "text",
    ["", "no json here", "[1, 2, 3]", '"just a string"', "{unclosed", "{'single': 1}", "42"],
)
def test_extract_json_returns_none_when_nothing_parses(text: str) -> None:
    assert extract_json(text) is None


def test_extract_json_skips_non_dict_fenced_block() -> None:
    assert extract_json('```json\n[1]\n```\nand {"x": 1}') == {"x": 1}


@pytest.mark.parametrize(
    ("value", "low", "high", "expected"),
    [(14, 0, 10, 10), (-2, 0, 10, 0), (5.5, 0, 10, 5.5), (0, 0, 10, 0), (10, 0, 10, 10)],
)
def test_clamp(value: float, low: float, high: float, expected: float) -> None:
    assert clamp(value, low, high) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (7, 7.0),
        (7.5, 7.5),
        ("8", 8.0),
        (" 6.25 ", 6.25),
        ("-1e1", -10.0),
        (True, None),
        (False, None),
        (None, None),
        ("eight", None),
        ("", None),
        ([1], None),
        ({"v": 1}, None),
        (math.nan, None),
        (math.inf, None),
        ("-inf", None),
        ("nan", None),
        (10**400, None),
    ],
)
def test_parse_float(value: Any, expected: float | None) -> None:
    assert parse_float(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [(3, 3.0), ("4.5", 4.5), (True, -1.0), ("x", -1.0), (math.nan, -1.0), (math.inf, -1.0)],
)
def test_coerce_float(value: Any, expected: float) -> None:
    result = coerce_float(value, -1.0)
    assert result == expected
    assert type(result) is float
