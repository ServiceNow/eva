"""Tests for eva.utils.json_utils module."""

from eva.utils.json_utils import (
    extract_and_load_json,
    extract_and_load_json_iter,
)


def test_plain_json():
    result = extract_and_load_json('{"rating": 3}')
    assert result == {"rating": 3}


def test_json_embedded_in_text():
    text = 'Here is the result: {"rating": 3, "explanation": "good"} as requested.'
    result = extract_and_load_json(text)
    assert result["rating"] == 3


def test_json_array():
    result = extract_and_load_json("[1, 2, 3]")
    assert result == [1, 2, 3]


def test_no_json_returns_none():
    result = extract_and_load_json("no json here")
    assert result is None


def test_multiple_objects():
    text = '{"a": 1} {"b": 2} [3, 4]'
    results = list(extract_and_load_json_iter(text))
    assert len(results) == 3
    assert results[0][0] == {"a": 1}
    assert results[1][0] == {"b": 2}
    assert results[2][0] == [3, 4]


def test_no_json():
    results = list(extract_and_load_json_iter("hello world"))
    assert results == []


def test_start_offset():
    text = '{"a": 1} {"b": 2}'
    results = list(extract_and_load_json_iter(text, start=9))
    assert len(results) == 1
    assert results[0][0] == {"b": 2}
