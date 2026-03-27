"""Shared utility functions used across all modules."""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    """Configure global logger, using Rich handler when available."""
    try:
        from rich.logging import RichHandler
        logging.basicConfig(
            level=level,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(rich_tracebacks=True)],
        )
    except ImportError:
        logging.basicConfig(
            level=level,
            format="%(asctime)s — %(levelname)s — %(message)s",
        )


def resolve_input(path: str | Path) -> Path:
    """Resolve and validate an input path (file or directory)."""
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Path not found: {p}")
    return p


def detect_input_type(path: Path) -> str:
    """
    Detect the type of input path.

    Returns:
        'zip' | 'txt' | 'dir'
    """
    if path.is_dir():
        return "dir"
    suffix = path.suffix.lower()
    if suffix == ".zip":
        return "zip"
    if suffix == ".txt":
        return "txt"
    raise ValueError(
        f"Unsupported format: {suffix}. Expected .zip, .txt or a directory."
    )


def find_chat_txt(directory: Path) -> Optional[Path]:
    """Search for _chat.txt inside a decompressed WhatsApp export directory."""
    candidates = list(directory.glob("*chat*.txt")) + list(directory.glob("*.txt"))
    if not candidates:
        return None
    # Prioritise files explicitly named _chat.txt
    for candidate in candidates:
        if "_chat" in candidate.name.lower():
            return candidate
    return candidates[0]


# Compiled once at module level for performance
_PHONE_RE = re.compile(r"\+?\d[\d\s\-().]{7,}\d")


def anonymize_phone(text: str) -> str:
    """Replace phone numbers with a short deterministic hash."""
    def _replace(match: re.Match) -> str:
        raw = match.group(0)
        digest = hashlib.md5(raw.encode()).hexdigest()[:6]
        return f"[USER_{digest}]"
    return _PHONE_RE.sub(_replace, text)


def anonymize_author(author: str) -> str:
    """Hash an author name or phone number for anonymisation."""
    digest = hashlib.md5(author.strip().encode()).hexdigest()[:6]
    return f"USER_{digest}"


def normalize_encoding(text: str) -> str:
    """Enforce UTF-8 and strip stray control characters."""
    text = text.encode("utf-8", errors="ignore").decode("utf-8")
    # Keep newlines and tabs, remove everything else in C0/C1 range
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text


def is_too_short(text: str, min_words: int = 3) -> bool:
    """Return True if the message contains fewer than min_words tokens."""
    return len(text.split()) < min_words


def detect_language(text: str) -> str:
    """
    Detect the language of a text string.

    Returns an ISO 639-1 code ('fr', 'en', 'wo', ...).
    Falls back to 'unknown' on detection failure.
    """
    try:
        from langdetect import detect
        return detect(text)
    except Exception:
        return "unknown"


def safe_mkdir(path: str | Path) -> Path:
    """Create a directory and all parents without raising if it already exists."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def format_duration(seconds: float) -> str:
    """Format a duration in seconds into a human-readable string."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"