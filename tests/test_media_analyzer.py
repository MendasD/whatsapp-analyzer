"""
Tests for media_analyzer.py.

Stats tests use real files created in tmp_path — no mocking needed.
Whisper is mocked via patch.dict("sys.modules", {"whisper": ...}).
ffmpeg is mocked via patch on shutil.which and subprocess.run.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from whatsapp_analyzer.media_analyzer import MediaAnalyzer

_SHUTIL_WHICH = "whatsapp_analyzer.media_analyzer.shutil.which"
_SUBPROCESS_RUN = "whatsapp_analyzer.media_analyzer.subprocess.run"


def _whisper_mock(text: str = "hello world"):
    """Return (mock_whisper_module, mock_model) ready for sys.modules injection."""
    mock_model = MagicMock()
    mock_model.transcribe.return_value = {"text": text}
    mock_whisper = MagicMock()
    mock_whisper.load_model.return_value = mock_model
    return mock_whisper, mock_model


def _proc(returncode: int = 0) -> MagicMock:
    """Return a fake subprocess.CompletedProcess-like mock."""
    p = MagicMock()
    p.returncode = returncode
    return p


# --- analyze() return shape -------------------------------------------------

class TestReturnShape:

    def test_returns_dict(self, tmp_path):
        assert isinstance(MediaAnalyzer().analyze(tmp_path), dict)

    def test_has_stats_key(self, tmp_path):
        assert "stats" in MediaAnalyzer().analyze(tmp_path)

    def test_has_transcriptions_key(self, tmp_path):
        assert "transcriptions" in MediaAnalyzer().analyze(tmp_path)

    def test_stats_is_dataframe(self, tmp_path):
        assert isinstance(MediaAnalyzer().analyze(tmp_path)["stats"], pd.DataFrame)

    def test_transcriptions_is_dataframe(self, tmp_path):
        assert isinstance(MediaAnalyzer().analyze(tmp_path)["transcriptions"], pd.DataFrame)


# --- stats ------------------------------------------------------------------

class TestStats:

    def test_empty_dir_stats_is_empty(self, tmp_path):
        assert MediaAnalyzer().analyze(tmp_path)["stats"].empty

    def test_empty_dir_stats_has_correct_columns(self, tmp_path):
        cols = list(MediaAnalyzer().analyze(tmp_path)["stats"].columns)
        assert cols == ["file_type", "count", "total_size_mb"]

    def test_nonexistent_dir_returns_empty_stats(self, tmp_path):
        assert MediaAnalyzer().analyze(tmp_path / "nope")["stats"].empty

    def test_image_jpg_counted(self, tmp_path):
        (tmp_path / "photo.jpg").write_bytes(b"x" * 512)
        stats = MediaAnalyzer().analyze(tmp_path)["stats"]
        assert ".jpg" in stats["file_type"].values

    def test_image_png_counted(self, tmp_path):
        (tmp_path / "photo.png").write_bytes(b"x" * 512)
        stats = MediaAnalyzer().analyze(tmp_path)["stats"]
        assert ".png" in stats["file_type"].values

    def test_audio_opus_counted(self, tmp_path):
        (tmp_path / "voice.opus").write_bytes(b"a" * 512)
        with patch.dict("sys.modules", {"whisper": None}):
            stats = MediaAnalyzer().analyze(tmp_path)["stats"]
        assert ".opus" in stats["file_type"].values

    def test_video_mp4_counted(self, tmp_path):
        (tmp_path / "clip.mp4").write_bytes(b"v" * 1024)
        with patch.dict("sys.modules", {"whisper": None}):
            stats = MediaAnalyzer().analyze(tmp_path)["stats"]
        assert ".mp4" in stats["file_type"].values

    def test_same_extension_aggregated_into_one_row(self, tmp_path):
        (tmp_path / "a.jpg").write_bytes(b"x" * 100)
        (tmp_path / "b.jpg").write_bytes(b"x" * 200)
        stats = MediaAnalyzer().analyze(tmp_path)["stats"]
        assert len(stats[stats["file_type"] == ".jpg"]) == 1

    def test_count_correct_for_multiple_files(self, tmp_path):
        (tmp_path / "a.jpg").write_bytes(b"x" * 100)
        (tmp_path / "b.jpg").write_bytes(b"x" * 200)
        stats = MediaAnalyzer().analyze(tmp_path)["stats"]
        assert stats[stats["file_type"] == ".jpg"].iloc[0]["count"] == 2

    def test_different_extensions_produce_separate_rows(self, tmp_path):
        (tmp_path / "photo.jpg").write_bytes(b"x" * 100)
        (tmp_path / "voice.opus").write_bytes(b"a" * 100)
        with patch.dict("sys.modules", {"whisper": None}):
            stats = MediaAnalyzer().analyze(tmp_path)["stats"]
        assert len(stats) == 2

    def test_total_size_mb_is_positive(self, tmp_path):
        (tmp_path / "photo.png").write_bytes(b"x" * 2048)
        stats = MediaAnalyzer().analyze(tmp_path)["stats"]
        assert stats.iloc[0]["total_size_mb"] > 0

    def test_unknown_extension_ignored(self, tmp_path):
        (tmp_path / "file.xyz").write_bytes(b"x" * 100)
        assert MediaAnalyzer().analyze(tmp_path)["stats"].empty

    def test_stats_computed_when_whisper_missing(self, tmp_path):
        (tmp_path / "voice.opus").write_bytes(b"a" * 512)
        with patch.dict("sys.modules", {"whisper": None}):
            stats = MediaAnalyzer().analyze(tmp_path)["stats"]
        assert not stats.empty

    def test_stats_computed_when_ffmpeg_missing(self, tmp_path):
        (tmp_path / "clip.mp4").write_bytes(b"v" * 1024)
        mock_whisper, _ = _whisper_mock()
        with patch.dict("sys.modules", {"whisper": mock_whisper}), \
             patch(_SHUTIL_WHICH, return_value=None):
            stats = MediaAnalyzer().analyze(tmp_path)["stats"]
        assert ".mp4" in stats["file_type"].values


# --- transcriptions: Whisper absent ----------------------------------------

class TestTranscriptionsNoWhisper:

    def test_empty_when_whisper_missing(self, tmp_path):
        (tmp_path / "voice.opus").write_bytes(b"a" * 100)
        with patch.dict("sys.modules", {"whisper": None}):
            t = MediaAnalyzer().analyze(tmp_path)["transcriptions"]
        assert t.empty

    def test_correct_columns_when_whisper_missing(self, tmp_path):
        (tmp_path / "voice.mp3").write_bytes(b"a" * 100)
        with patch.dict("sys.modules", {"whisper": None}):
            t = MediaAnalyzer().analyze(tmp_path)["transcriptions"]
        assert list(t.columns) == ["file_path", "text"]

    def test_empty_when_only_images(self, tmp_path):
        (tmp_path / "photo.jpg").write_bytes(b"x" * 100)
        t = MediaAnalyzer().analyze(tmp_path)["transcriptions"]
        assert t.empty

    def test_correct_columns_when_only_images(self, tmp_path):
        (tmp_path / "photo.png").write_bytes(b"x" * 100)
        t = MediaAnalyzer().analyze(tmp_path)["transcriptions"]
        assert list(t.columns) == ["file_path", "text"]

    def test_empty_dir_transcriptions_empty(self, tmp_path):
        t = MediaAnalyzer().analyze(tmp_path)["transcriptions"]
        assert t.empty


# --- transcriptions: audio with Whisper ------------------------------------

class TestAudioTranscriptions:

    def test_opus_produces_one_row(self, tmp_path):
        (tmp_path / "voice.opus").write_bytes(b"a" * 100)
        mock_whisper, _ = _whisper_mock("bonjour")
        with patch.dict("sys.modules", {"whisper": mock_whisper}):
            t = MediaAnalyzer().analyze(tmp_path)["transcriptions"]
        assert len(t) == 1

    def test_transcription_text_matches_whisper_output(self, tmp_path):
        (tmp_path / "voice.mp3").write_bytes(b"a" * 100)
        mock_whisper, _ = _whisper_mock("hello whisper")
        with patch.dict("sys.modules", {"whisper": mock_whisper}):
            t = MediaAnalyzer().analyze(tmp_path)["transcriptions"]
        assert t.iloc[0]["text"] == "hello whisper"

    def test_file_path_column_is_string(self, tmp_path):
        (tmp_path / "voice.ogg").write_bytes(b"a" * 100)
        mock_whisper, _ = _whisper_mock()
        with patch.dict("sys.modules", {"whisper": mock_whisper}):
            t = MediaAnalyzer().analyze(tmp_path)["transcriptions"]
        assert isinstance(t.iloc[0]["file_path"], str)

    def test_file_path_contains_filename(self, tmp_path):
        (tmp_path / "voice.opus").write_bytes(b"a" * 100)
        mock_whisper, _ = _whisper_mock()
        with patch.dict("sys.modules", {"whisper": mock_whisper}):
            t = MediaAnalyzer().analyze(tmp_path)["transcriptions"]
        assert "voice.opus" in t.iloc[0]["file_path"]

    def test_multiple_audio_files_produce_multiple_rows(self, tmp_path):
        (tmp_path / "a.opus").write_bytes(b"a" * 100)
        (tmp_path / "b.mp3").write_bytes(b"b" * 100)
        mock_whisper, _ = _whisper_mock()
        with patch.dict("sys.modules", {"whisper": mock_whisper}):
            t = MediaAnalyzer().analyze(tmp_path)["transcriptions"]
        assert len(t) == 2

    def test_whisper_load_model_called_with_base(self, tmp_path):
        (tmp_path / "voice.opus").write_bytes(b"a" * 100)
        mock_whisper, _ = _whisper_mock()
        with patch.dict("sys.modules", {"whisper": mock_whisper}):
            MediaAnalyzer().analyze(tmp_path)
        mock_whisper.load_model.assert_called_once_with("base")

    def test_images_do_not_appear_in_transcriptions(self, tmp_path):
        (tmp_path / "photo.jpg").write_bytes(b"x" * 100)
        (tmp_path / "voice.opus").write_bytes(b"a" * 100)
        mock_whisper, _ = _whisper_mock("text")
        with patch.dict("sys.modules", {"whisper": mock_whisper}):
            t = MediaAnalyzer().analyze(tmp_path)["transcriptions"]
        assert len(t) == 1

    def test_transcription_exception_skipped_gracefully(self, tmp_path):
        (tmp_path / "voice.opus").write_bytes(b"a" * 100)
        mock_whisper, mock_model = _whisper_mock()
        mock_model.transcribe.side_effect = RuntimeError("whisper error")
        with patch.dict("sys.modules", {"whisper": mock_whisper}):
            t = MediaAnalyzer().analyze(tmp_path)["transcriptions"]
        assert t.empty


# --- transcriptions: video with ffmpeg + Whisper ---------------------------

class TestVideoTranscriptions:

    def test_video_row_produced_when_ffmpeg_and_whisper_available(self, tmp_path):
        (tmp_path / "clip.mp4").write_bytes(b"v" * 1024)
        mock_whisper, _ = _whisper_mock("video text")
        with patch.dict("sys.modules", {"whisper": mock_whisper}), \
             patch(_SHUTIL_WHICH, return_value="/usr/bin/ffmpeg"), \
             patch(_SUBPROCESS_RUN, return_value=_proc(0)):
            t = MediaAnalyzer().analyze(tmp_path)["transcriptions"]
        assert len(t) == 1

    def test_video_text_comes_from_whisper(self, tmp_path):
        (tmp_path / "clip.mp4").write_bytes(b"v" * 1024)
        mock_whisper, _ = _whisper_mock("video text")
        with patch.dict("sys.modules", {"whisper": mock_whisper}), \
             patch(_SHUTIL_WHICH, return_value="/usr/bin/ffmpeg"), \
             patch(_SUBPROCESS_RUN, return_value=_proc(0)):
            t = MediaAnalyzer().analyze(tmp_path)["transcriptions"]
        assert t.iloc[0]["text"] == "video text"

    def test_video_file_path_is_original_mp4(self, tmp_path):
        (tmp_path / "clip.mp4").write_bytes(b"v" * 1024)
        mock_whisper, _ = _whisper_mock()
        with patch.dict("sys.modules", {"whisper": mock_whisper}), \
             patch(_SHUTIL_WHICH, return_value="/usr/bin/ffmpeg"), \
             patch(_SUBPROCESS_RUN, return_value=_proc(0)):
            t = MediaAnalyzer().analyze(tmp_path)["transcriptions"]
        assert "clip.mp4" in t.iloc[0]["file_path"]

    def test_video_skipped_when_ffmpeg_missing(self, tmp_path):
        (tmp_path / "clip.mp4").write_bytes(b"v" * 1024)
        mock_whisper, _ = _whisper_mock()
        with patch.dict("sys.modules", {"whisper": mock_whisper}), \
             patch(_SHUTIL_WHICH, return_value=None):
            t = MediaAnalyzer().analyze(tmp_path)["transcriptions"]
        assert t.empty

    def test_video_skipped_when_ffmpeg_fails(self, tmp_path):
        (tmp_path / "clip.mp4").write_bytes(b"v" * 1024)
        mock_whisper, _ = _whisper_mock()
        with patch.dict("sys.modules", {"whisper": mock_whisper}), \
             patch(_SHUTIL_WHICH, return_value="/usr/bin/ffmpeg"), \
             patch(_SUBPROCESS_RUN, return_value=_proc(1)):
            t = MediaAnalyzer().analyze(tmp_path)["transcriptions"]
        assert t.empty

    def test_subprocess_called_with_ffmpeg_command(self, tmp_path):
        (tmp_path / "clip.mp4").write_bytes(b"v" * 1024)
        mock_whisper, _ = _whisper_mock()
        with patch.dict("sys.modules", {"whisper": mock_whisper}), \
             patch(_SHUTIL_WHICH, return_value="/usr/bin/ffmpeg"), \
             patch(_SUBPROCESS_RUN, return_value=_proc(0)) as mock_run:
            MediaAnalyzer().analyze(tmp_path)
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "ffmpeg"

    def test_subprocess_receives_video_path(self, tmp_path):
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"v" * 1024)
        mock_whisper, _ = _whisper_mock()
        with patch.dict("sys.modules", {"whisper": mock_whisper}), \
             patch(_SHUTIL_WHICH, return_value="/usr/bin/ffmpeg"), \
             patch(_SUBPROCESS_RUN, return_value=_proc(0)) as mock_run:
            MediaAnalyzer().analyze(tmp_path)
        cmd = mock_run.call_args[0][0]
        assert str(video) in cmd

    def test_audio_and_video_both_transcribed(self, tmp_path):
        (tmp_path / "voice.opus").write_bytes(b"a" * 100)
        (tmp_path / "clip.mp4").write_bytes(b"v" * 1024)
        mock_whisper, _ = _whisper_mock("text")
        with patch.dict("sys.modules", {"whisper": mock_whisper}), \
             patch(_SHUTIL_WHICH, return_value="/usr/bin/ffmpeg"), \
             patch(_SUBPROCESS_RUN, return_value=_proc(0)):
            t = MediaAnalyzer().analyze(tmp_path)["transcriptions"]
        assert len(t) == 2
