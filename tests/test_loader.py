"""Tests for loader.py."""

import zipfile
import pytest
from whatsapp_analyzer.loader import Loader, LoadedGroup


ANDROID_CHAT = (
    "12/01/2024, 08:15 - Aminata: Bonjour tout le monde !\n"
    "12/01/2024, 08:17 - Moussa: Salut, ça va ?\n"
)


@pytest.fixture
def chat_txt(tmp_path):
    f = tmp_path / "_chat.txt"
    f.write_text(ANDROID_CHAT, encoding="utf-8")
    return f


@pytest.fixture
def chat_zip(tmp_path, chat_txt):
    zip_path = tmp_path / "export.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(chat_txt, arcname="_chat.txt")
    return zip_path


@pytest.fixture
def chat_dir(tmp_path):
    chat = tmp_path / "_chat.txt"
    chat.write_text(ANDROID_CHAT, encoding="utf-8")
    media = tmp_path / "media"
    media.mkdir()
    (media / "photo.jpg").write_bytes(b"fake")
    return tmp_path


def test_load_txt(chat_txt):
    group = Loader().load(chat_txt)
    assert isinstance(group, LoadedGroup)
    assert group.chat_path == chat_txt
    assert group.media_dir is None


def test_load_zip(chat_zip):
    group = Loader().load(chat_zip)
    assert isinstance(group, LoadedGroup)
    assert group.chat_path.name == "_chat.txt"
    group.cleanup()


def test_load_dir(chat_dir):
    group = Loader().load(chat_dir)
    assert group.chat_path.exists()
    assert group.media_dir is not None


def test_load_many(chat_txt, chat_zip):
    groups = Loader().load_many([chat_txt, chat_zip])
    assert len(groups) == 2
    groups[-1].cleanup()


def test_load_many_skips_invalid(chat_txt, tmp_path):
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"not a zip")
    groups = Loader().load_many([chat_txt, bad])
    assert len(groups) == 1


def test_load_missing_raises():
    with pytest.raises(FileNotFoundError):
        Loader().load("/nonexistent/chat.txt")


def test_load_no_chat_txt_raises(tmp_path):
    (tmp_path / "readme.md").write_text("nothing")
    with pytest.raises(FileNotFoundError):
        Loader()._from_dir(tmp_path)


def test_loaded_group_repr(chat_txt):
    group = Loader().load(chat_txt)
    assert "LoadedGroup" in repr(group)


def test_cleanup_does_not_raise_when_no_tmp(chat_txt):
    group = Loader().load(chat_txt)
    group.cleanup()  # _tmp_dir is None — should be a no-op