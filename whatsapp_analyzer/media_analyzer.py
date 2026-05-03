"""
Media analysis module for WhatsApp conversation analysis.

Computes per-extension file statistics for all known media files in a
directory, and optionally transcribes audio / video via Whisper.

Output keys
-----------
"stats"          — DataFrame: file_type, count, total_size_mb
                   Always populated, even when Whisper is absent.
"transcriptions" — DataFrame: file_path, text
                   Empty when Whisper is not installed or no audio/video found.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_AUDIO_EXTENSIONS = frozenset({".opus", ".ogg", ".mp3"})
_VIDEO_EXTENSIONS = frozenset({".mp4"})
_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp"})
_ALL_EXTENSIONS = _AUDIO_EXTENSIONS | _VIDEO_EXTENSIONS | _IMAGE_EXTENSIONS

_EMPTY_STATS = pd.DataFrame(columns=["file_type", "count", "total_size_mb"])
_EMPTY_TRANSCRIPTIONS = pd.DataFrame(columns=["file_path", "text"])


class MediaAnalyzer:
    """Analyse media files produced by a WhatsApp export."""

    def analyze(self, media_dir: Path) -> dict:
        """
        Scan a media directory, compute stats, and transcribe audio/video.

        Args:
            media_dir: Directory containing WhatsApp media files.

        Returns:
            dict with keys ``"stats"`` and ``"transcriptions"``.
        """
        files = self._scan(Path(media_dir))
        return {
            "stats": self._compute_stats(files),
            "transcriptions": self._transcribe_all(files),
        }

    # scanning

    def _scan(self, media_dir: Path) -> list[Path]:
        if not media_dir.exists() or not media_dir.is_dir():
            return []
        return [
            f for f in media_dir.iterdir()
            if f.is_file() and f.suffix.lower() in _ALL_EXTENSIONS
        ]

    # stats

    def _compute_stats(self, files: list[Path]) -> pd.DataFrame:
        if not files:
            return _EMPTY_STATS.copy()

        by_ext: dict[str, dict] = {}
        for f in files:
            ext = f.suffix.lower()
            entry = by_ext.setdefault(ext, {"count": 0, "size_bytes": 0})
            entry["count"] += 1
            entry["size_bytes"] += f.stat().st_size

        rows = [
            {
                "file_type": ext,
                "count": data["count"],
                "total_size_mb": round(data["size_bytes"] / (1024 * 1024), 6),
            }
            for ext, data in sorted(by_ext.items())
        ]
        return pd.DataFrame(rows)

    # transcription

    def _load_whisper(self):
        """Return a Whisper model, or None if Whisper is not installed."""
        try:
            import whisper
            return whisper.load_model("base")
        except ImportError:
            logger.warning("whisper not installed; transcription disabled.")
            return None

    def _transcribe_all(self, files: list[Path]) -> pd.DataFrame:
        audio_files = [f for f in files if f.suffix.lower() in _AUDIO_EXTENSIONS]
        video_files = [f for f in files if f.suffix.lower() in _VIDEO_EXTENSIONS]

        if not audio_files and not video_files:
            return _EMPTY_TRANSCRIPTIONS.copy()

        model = self._load_whisper()
        if model is None:
            return _EMPTY_TRANSCRIPTIONS.copy()

        rows: list[dict] = []

        for f in audio_files:
            text = self._transcribe_file(model, f)
            if text is not None:
                rows.append({"file_path": str(f), "text": text})

        for f in video_files:
            extracted = self._extract_audio(f)
            if extracted is not None:
                text = self._transcribe_file(model, extracted)
                if text is not None:
                    rows.append({"file_path": str(f), "text": text})
                try:
                    extracted.unlink()
                except Exception:
                    pass

        if not rows:
            return _EMPTY_TRANSCRIPTIONS.copy()
        return pd.DataFrame(rows)

    def _transcribe_file(self, model, path: Path) -> str | None:
        """Return stripped transcription text, or None on failure."""
        try:
            result = model.transcribe(str(path))
            return result.get("text", "").strip()
        except Exception as exc:
            logger.warning("Transcription failed for %s: %s", Path(path).name, exc)
            return None

    def _extract_audio(self, video_path: Path) -> Path | None:
        """
        Extract audio track from a video file using ffmpeg.

        Returns the path to the temporary .mp3 file, or None if ffmpeg is
        unavailable or the extraction fails.
        """
        if shutil.which("ffmpeg") is None:
            logger.warning("ffmpeg not found; skipping video %s", video_path.name)
            return None

        tmp_audio = Path(tempfile.mktemp(suffix=".mp3"))
        proc = subprocess.run(
            [
                "ffmpeg", "-i", str(video_path),
                "-vn", "-acodec", "libmp3lame",
                "-y", "-loglevel", "error",
                str(tmp_audio),
            ],
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            logger.warning(
                "ffmpeg exited with code %d for %s", proc.returncode, video_path.name
            )
            return None
        return tmp_audio
