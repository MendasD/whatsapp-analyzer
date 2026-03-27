"""Tests for parser.py."""

import pytest
import pandas as pd
from whatsapp_analyzer.parser import Parser


ANDROID_CHAT = """\
12/01/2024, 08:15 - Aminata: Bonjour tout le monde !
12/01/2024, 08:17 - Moussa: Salut, ça va ?
12/01/2024, 08:20 - Aminata: <Media omitted>
12/01/2024, 08:21 - Messages and calls are end-to-end encrypted
12/01/2024, 09:00 - Ibrahima: Le cours est annulé demain
12/01/2024, 09:01 - Ibrahima: suite du message
"""

IOS_CHAT = """\
[12/01/2024 à 08:15:00] Aminata : Bonjour tout le monde !
[12/01/2024 à 08:17:00] Moussa : Salut, ça va ?
[12/01/2024, 09:00:00] Ibrahima : Le cours est annulé demain
"""


@pytest.fixture
def android_file(tmp_path):
    f = tmp_path / "_chat.txt"
    f.write_text(ANDROID_CHAT, encoding="utf-8")
    return f


@pytest.fixture
def ios_file(tmp_path):
    f = tmp_path / "_chat.txt"
    f.write_text(IOS_CHAT, encoding="utf-8")
    return f


def test_parse_android_returns_dataframe(android_file):
    df = Parser().parse(android_file)
    assert isinstance(df, pd.DataFrame)


def test_parse_android_columns(android_file):
    df = Parser().parse(android_file)
    assert set(["timestamp", "author", "message", "msg_type", "group_name"]).issubset(df.columns)


def test_parse_android_message_count(android_file):
    df = Parser().parse(android_file)
    # 4 real messages (media + system lines are parsed but typed accordingly)
    assert len(df) >= 4


def test_parse_android_authors(android_file):
    df = Parser().parse(android_file)
    authors = df["author"].unique()
    assert "Aminata" in authors
    assert "Moussa" in authors


def test_parse_android_timestamp_dtype(android_file):
    df = Parser().parse(android_file)
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])


def test_parse_android_media_classified(android_file):
    df = Parser().parse(android_file)
    assert (df["msg_type"] == "media").any()


def test_parse_android_system_classified(tmp_path):
    # System lines without an author prefix are dropped by the regex.
    # System lines WITH an author prefix are classified as 'system'.
    chat = tmp_path / "_chat.txt"
    chat.write_text(
        "12/01/2024, 08:00 - Moussa: message deleted\n",
        encoding="utf-8",
    )
    df = Parser().parse(chat)
    assert (df["msg_type"] == "system").any()


def test_parse_ios_returns_dataframe(ios_file):
    df = Parser().parse(ios_file)
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 3


def test_parse_ios_authors(ios_file):
    df = Parser().parse(ios_file)
    assert "Aminata" in df["author"].values


def test_parse_multiline_merged(tmp_path):
    chat = tmp_path / "_chat.txt"
    chat.write_text(
        "12/01/2024, 08:00 - Moussa: First line\n"
        "continuation here\n"
        "12/01/2024, 08:05 - Aminata: Next message\n",
        encoding="utf-8",
    )
    df = Parser().parse(chat)
    assert "continuation here" in df.loc[df["author"] == "Moussa", "message"].values[0]


def test_parse_anonymize(android_file):
    df = Parser(anonymize=True).parse(android_file)
    assert "Aminata" not in df["author"].values
    assert all(a.startswith("USER_") for a in df["author"].unique())


def test_parse_group_name_override(android_file):
    df = Parser(group_name="TestGroup").parse(android_file)
    assert (df["group_name"] == "TestGroup").all()


def test_parse_empty_file_raises(tmp_path):
    f = tmp_path / "_chat.txt"
    f.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        Parser().parse(f)


def test_parse_garbage_file_raises(tmp_path):
    f = tmp_path / "_chat.txt"
    f.write_text("this is not a whatsapp export\nrandom text\n", encoding="utf-8")
    with pytest.raises(ValueError):
        Parser().parse(f)