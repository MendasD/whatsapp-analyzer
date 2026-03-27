"""Tests for utils.py."""

import pytest
from pathlib import Path
from whatsapp_analyzer.utils import (
    resolve_input,
    detect_input_type,
    find_chat_txt,
    anonymize_phone,
    anonymize_author,
    normalize_encoding,
    is_too_short,
    safe_mkdir,
    format_duration,
)


def test_resolve_input_valid(tmp_path):
    f = tmp_path / "chat.txt"
    f.write_text("hello")
    assert resolve_input(f) == f.resolve()


def test_resolve_input_missing():
    with pytest.raises(FileNotFoundError):
        resolve_input("/nonexistent/path/chat.txt")


def test_detect_input_type_zip(tmp_path):
    f = tmp_path / "export.zip"
    f.touch()
    assert detect_input_type(f) == "zip"


def test_detect_input_type_txt(tmp_path):
    f = tmp_path / "_chat.txt"
    f.touch()
    assert detect_input_type(f) == "txt"


def test_detect_input_type_dir(tmp_path):
    assert detect_input_type(tmp_path) == "dir"


def test_detect_input_type_unsupported(tmp_path):
    f = tmp_path / "file.csv"
    f.touch()
    with pytest.raises(ValueError):
        detect_input_type(f)


def test_find_chat_txt_finds_file(tmp_path):
    chat = tmp_path / "_chat.txt"
    chat.write_text("data")
    assert find_chat_txt(tmp_path) == chat


def test_find_chat_txt_prefers_named(tmp_path):
    other = tmp_path / "other.txt"
    other.write_text("x")
    chat = tmp_path / "_chat.txt"
    chat.write_text("y")
    assert find_chat_txt(tmp_path) == chat


def test_find_chat_txt_returns_none_when_empty(tmp_path):
    assert find_chat_txt(tmp_path) is None


def test_anonymize_phone_replaces():
    result = anonymize_phone("Call me at +221 77 123 45 67 please")
    assert "+221" not in result
    assert "USER_" in result


def test_anonymize_phone_no_phone():
    text = "No phone number here."
    assert anonymize_phone(text) == text


def test_anonymize_author_deterministic():
    a1 = anonymize_author("Aminata")
    a2 = anonymize_author("Aminata")
    assert a1 == a2
    assert a1.startswith("USER_")


def test_normalize_encoding_removes_control_chars():
    text = "Hello\x00World\x08!"
    result = normalize_encoding(text)
    assert "\x00" not in result
    assert "\x08" not in result
    assert "Hello" in result


def test_normalize_encoding_keeps_newlines():
    text = "line1\nline2"
    assert "\n" in normalize_encoding(text)


def test_is_too_short_true():
    assert is_too_short("ok") is True


def test_is_too_short_false():
    assert is_too_short("this is long enough") is False


def test_safe_mkdir_creates(tmp_path):
    target = tmp_path / "a" / "b" / "c"
    result = safe_mkdir(target)
    assert result.exists()


def test_safe_mkdir_idempotent(tmp_path):
    safe_mkdir(tmp_path)
    safe_mkdir(tmp_path)  # should not raise


def test_format_duration_seconds():
    assert format_duration(45) == "45s"


def test_format_duration_minutes():
    assert format_duration(154) == "2m 34s"


def test_format_duration_hours():
    assert format_duration(3661) == "1h 1m 1s"