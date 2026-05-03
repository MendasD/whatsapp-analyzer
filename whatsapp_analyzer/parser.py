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

# Matches the timestamp prefix only — used to detect line boundaries regardless
# of whether the line has an Author: Message suffix.  A line that starts with a
# timestamp but does NOT match the full _ANDROID_RE / _IOS_RE is a WhatsApp
# system notification (e.g. "Alice a ajouté Bob") and must be treated as a new
# entry, not merged as a continuation of the previous message.
_TIMESTAMP_PREFIX_RE = re.compile(
    r"^(?:\d{1,2}/\d{1,2}/\d{2,4}[,\s]|\[\d{1,2}/\d{1,2}/\d{2,4})",
)

# Message-level patterns that flag a parsed Author:Message line as a system event.
# Patterns are anchored or use word boundaries to reduce false positives on real
# user text (e.g. "I left the meeting" would NOT match "\bleft$" alone, but it
# would match "left" — so we scope each pattern as tightly as possible).
_SYSTEM_PATTERNS = re.compile(
    r"(?:"
    # Encryption notices
    r"messages and calls are end-to-end encrypted"
    r"|les messages et les appels sont chiffrés de bout en bout"
    r"|tap to learn more"
    r"|appuyez pour en savoir plus"
    # Deleted / missing messages
    r"|this message was deleted"
    r"|ce message a été supprimé"
    r"|message (?:deleted|supprimé)"
    # Media placeholders
    r"|<media omitted>"
    r"|média omis"
    r"|(?:image|video|audio|sticker|document|gif) omitted"
    # Group membership events — French
    r"|\ba ajouté\b"
    r"|\ba quitté\b"
    r"|\ba été supprimé[e]?\b"
    r"|\ba supprimé\b"
    r"|\ba rejoint\b"
    r"|\ba créé (?:le )?groupe"
    r"|\bont été ajouté[e]?s?\b"
    r"|\bavez (?:été ajouté|rejoint)\b"
    r"|\ba changé (?:le nom|l'intitulé|l'icône) du groupe"
    r"|\ba modifié la description du groupe"
    r"|\ba épinglé un message"
    r"|\ba rejoint en utilisant le lien"
    # Group membership events — English
    r"|\bwas added\b"
    r"|\bwere added\b"
    r"|\byou were added\b"
    r"|\bjoined using (?:this|the) group"
    r"|\bjoined the group\b"
    r"|\bcreated (?:the )?group\b"
    r"|\bchanged the (?:subject|group name|group description|group(?:'s)? icon)\b"
    r"|\bchanged this group"
    r"|\bpinned a message\b"
    r"|\bwas removed from the group\b"
    r"|\bremoved .{1,50} from the group\b"
    # Missed calls
    r"|missed (?:voice|video) call"
    r"|appel (?:vocal|vidéo) manqué"
    # Security-code change
    r"|your security code with .{1,60} changed"
    r"|votre code de sécurité avec .{1,60} a changé"
    r")",
    re.IGNORECASE,
)

# Lines that reference an omitted media attachment
_MEDIA_PATTERN = re.compile(
    r"(<Media omitted>|Média omis|image omitted|video omitted|audio omitted"
    r"|sticker omitted|document omitted|gif omitted)",
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
        """
        Return True for any line that begins with a WhatsApp timestamp.

        Using only the timestamp prefix (not the full Author: pattern) ensures
        that system-event lines such as "12/01/2024, 08:15 - Alice a ajouté Bob"
        are treated as new entries rather than being appended to the previous
        user message as a continuation.
        """
        return bool(_TIMESTAMP_PREFIX_RE.match(line))

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