"""
Detect the input format, decompress archives when needed,
and route to the parser with a normalised LoadedGroup object.

Supported inputs:
    - .zip  (native WhatsApp export)
    - .txt  (_chat.txt file alone)
    - dir   (already-decompressed export folder)
    - list  (multiple groups passed at once)
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

from whatsapp_analyzer.utils import detect_input_type, find_chat_txt, resolve_input

logger = logging.getLogger(__name__)

# Media extensions produced by WhatsApp exports
_MEDIA_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp",
    ".mp4", ".opus", ".ogg", ".mp3",
    ".pdf", ".gif",
}


class LoadedGroup:
    """
    Holds everything needed to analyse one WhatsApp group.

    Attributes:
        chat_path:  Path to the _chat.txt file.
        media_dir:  Path to the media folder, or None if absent.
        group_name: Human-readable group identifier.
    """

    def __init__(
        self,
        chat_path: Path,
        media_dir: Optional[Path] = None,
        group_name: Optional[str] = None,
        _tmp_dir: Optional[Path] = None,
    ) -> None:
        self.chat_path = chat_path
        self.media_dir = media_dir
        self.group_name = group_name or chat_path.parent.name
        self._tmp_dir = _tmp_dir

    def cleanup(self) -> None:
        """Remove the temporary decompression directory if one was created."""
        if self._tmp_dir and self._tmp_dir.exists():
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
            logger.debug("Removed temp dir: %s", self._tmp_dir)

    def __repr__(self) -> str:
        return (
            f"LoadedGroup(group={self.group_name!r}, "
            f"has_media={self.media_dir is not None})"
        )


class Loader:
    """
    Resolve any supported input format into one or more LoadedGroup objects.

    Usage:
        group  = Loader().load("chat.zip")
        groups = Loader().load_many(["g1.zip", "g2.txt", "g3/"])
    """

    def load(self, path: str | Path) -> LoadedGroup:
        """Load a single WhatsApp group from any supported format."""
        resolved = resolve_input(path)
        kind = detect_input_type(resolved)
        logger.info("Loading [%s]: %s", kind.upper(), resolved.name)

        if kind == "zip":
            return self._from_zip(resolved)
        if kind == "txt":
            return self._from_txt(resolved)
        return self._from_dir(resolved)

    def load_many(self, paths: list[str | Path]) -> list[LoadedGroup]:
        """Load multiple groups, skipping any that fail with a warning."""
        groups: list[LoadedGroup] = []
        for path in paths:
            try:
                groups.append(self.load(path))
            except Exception as exc:
                logger.warning("Skipping %s — %s", path, exc)
        if not groups:
            raise RuntimeError("No groups could be loaded.")
        return groups

    def _from_zip(self, zip_path: Path) -> LoadedGroup:
        """Decompress a .zip archive into a temp directory, then delegate to _from_dir."""
        tmp_dir = Path(tempfile.mkdtemp(prefix="wac_"))
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(tmp_dir)
            logger.debug("ZIP extracted to: %s", tmp_dir)
            group = self._from_dir(tmp_dir, group_name=zip_path.stem)
            group._tmp_dir = tmp_dir
            return group
        except zipfile.BadZipFile:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise ValueError(f"Invalid or corrupted ZIP file: {zip_path}")

    def _from_txt(self, txt_path: Path) -> LoadedGroup:
        """Wrap a bare _chat.txt file with no media directory."""
        return LoadedGroup(
            chat_path=txt_path,
            media_dir=None,
            group_name=txt_path.stem,
        )

    def _from_dir(
        self, dir_path: Path, group_name: Optional[str] = None
    ) -> LoadedGroup:
        """Locate _chat.txt and optional media folder inside a directory."""
        chat_txt = find_chat_txt(dir_path)
        if chat_txt is None:
            raise FileNotFoundError(f"No _chat.txt found in: {dir_path}")

        media_dir = self._find_media_dir(dir_path)
        if media_dir:
            logger.info("Media folder detected: %s", media_dir.name)

        return LoadedGroup(
            chat_path=chat_txt,
            media_dir=media_dir,
            group_name=group_name or dir_path.name,
        )

    @staticmethod
    def _find_media_dir(base: Path) -> Optional[Path]:
        """
        Return the directory that contains WhatsApp media files, or None.

        WhatsApp places media alongside _chat.txt or in a named sub-folder
        depending on the platform and export version.
        """
        for item in base.iterdir():
            if item.is_dir():
                if any(
                    f.suffix.lower() in _MEDIA_EXTENSIONS
                    for f in item.iterdir()
                    if f.is_file()
                ):
                    return item

        # Media files may sit directly in the root folder
        if any(
            f.suffix.lower() in _MEDIA_EXTENSIONS
            for f in base.iterdir()
            if f.is_file()
        ):
            return base

        return None