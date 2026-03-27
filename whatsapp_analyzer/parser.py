"""
Parse a WhatsApp _chat.txt export into a structured pandas DataFrame.

WhatsApp produces slightly different timestamp formats depending on
the platform (Android vs iOS) and the device locale. This module
handles both families with two regex patterns and falls back
gracefully when neither matches a line.

Output DataFrame columns:
    timestamp   : datetime64[ns]
    author      : str
    message     : str   (raw, untouched)
    msg_type    : str   ('text' | 'media' | 'system')
    group_name  : str
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd

from whatsapp_analyzer.utils import anonymize_author, normalize_encoding

logger = logging.getLogger(__name__)


# Android: "12/01/2024, 08:15 - Author: Message"
_ANDROID_RE = re.compile(
    r"^(?P<date>\d{1,2}/\d{1,2}/\d{2,4}),\s"
    r"(?P<time>\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AP]M)?)\s-\s"
    r"(?P<author>[^:]+):\s"
    r"(?P<message>.+)$",
    re.IGNORECASE,
)

# iOS: "[12/01/2024 à 08:15:00] Author : Message"
# also handles "[12/01/2024, 08:15:00]" variants
_IOS_RE = re.compile(
    r"^\[(?P<date>\d{1,2}/\d{1,2}/\d{2,4})[,\s à]+(?P<time>\d{1,2}:\d{2}(?::\d{2})?)\]\s"
    r"(?P<author>[^:]+)\s?:\s"
    r"(?P<message>.+)$",
    re.IGNORECASE,
)

# Lines that are WhatsApp system messages, not user messages
_SYSTEM_PATTERNS = re.compile(
    r"(Messages and calls are end-to-end encrypted"
    r"|<Media omitted>"
    r"|Média omis"
    r"|message deleted"
    r"|Message supprimé"
    r"|vous avez rejoint"
    r"|added|removed|left|joined|changed the subject"
    r"|created group)",
    re.IGNORECASE,
)

# Lines that reference an omitted media attachment
_MEDIA_PATTERN = re.compile(
    r"(<Media omitted>|Média omis|image omitted|video omitted|audio omitted)",
    re.IGNORECASE,
)


class Parser:
    """
    Parse a WhatsApp _chat.txt file into a DataFrame.

    Args:
        anonymize: Replace author names/phones with hashed identifiers.
        group_name: Override the group name stored in the output DataFrame.
    """

    def __init__(
        self,
        anonymize: bool = False,
        group_name: Optional[str] = None,
    ) -> None:
        self.anonymize = anonymize
        self.group_name = group_name

    def parse(self, chat_path: Path) -> pd.DataFrame:
        """
        Read and parse a _chat.txt file.

        Args:
            chat_path: Path to the WhatsApp export text file.

        Returns:
            DataFrame with columns [timestamp, author, message, msg_type, group_name].
        """
        raw_text = self._read(chat_path)
        lines = self._merge_multiline(raw_text.splitlines())
        records = [self._parse_line(line) for line in lines]
        records = [r for r in records if r is not None]

        if not records:
            raise ValueError(
                f"No messages could be parsed from {chat_path}. "
                "Check that the file is a valid WhatsApp export."
            )

        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True, errors="coerce")
        df["group_name"] = self.group_name or chat_path.stem
        df = df.dropna(subset=["timestamp"]).reset_index(drop=True)

        logger.info(
            "Parsed %d messages from %d authors in '%s'.",
            len(df),
            df["author"].nunique(),
            df["group_name"].iloc[0],
        )
        return df

    def _read(self, path: Path) -> str:
        """Read the file, enforcing UTF-8 and stripping control characters."""
        raw = path.read_text(encoding="utf-8", errors="ignore")
        return normalize_encoding(raw)

    def _merge_multiline(self, lines: list[str]) -> list[str]:
        """
        Merge continuation lines (no timestamp prefix) back into
        the message they belong to.
        """
        merged: list[str] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if self._is_message_start(line):
                merged.append(line)
            elif merged:
                # Continuation of the previous message
                merged[-1] += " " + line
        return merged

    def _is_message_start(self, line: str) -> bool:
        """Return True if the line starts with a recognised timestamp pattern."""
        return bool(_ANDROID_RE.match(line) or _IOS_RE.match(line))

    def _parse_line(self, line: str) -> Optional[dict]:
        """
        Extract fields from a single (possibly merged) message line.

        Returns None for lines that cannot be matched.
        """
        match = _ANDROID_RE.match(line) or _IOS_RE.match(line)
        if not match:
            return None

        author = match.group("author").strip()
        message = match.group("message").strip()
        timestamp_str = f"{match.group('date')} {match.group('time')}"

        if self.anonymize:
            author = anonymize_author(author)

        msg_type = self._classify_message(message)

        return {
            "timestamp": timestamp_str,
            "author": author,
            "message": message,
            "msg_type": msg_type,
        }

    @staticmethod
    def _classify_message(message: str) -> str:
        """Classify a message as 'text', 'media', or 'system'."""
        if _MEDIA_PATTERN.search(message):
            return "media"
        if _SYSTEM_PATTERNS.search(message):
            return "system"
        return "text"