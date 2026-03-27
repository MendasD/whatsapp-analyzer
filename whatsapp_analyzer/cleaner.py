"""
Clean and preprocess a parsed WhatsApp DataFrame for NLP analysis.

Steps applied in order:
    1. Drop system and media messages.
    2. Normalise unicode and strip residual control characters.
    3. Remove emoji characters (or transliterate, depending on mode).
    4. Detect language per message (or use the forced lang parameter).
    5. Lowercase and strip punctuation.
    6. Remove stopwords for the detected language.
    7. Lemmatise tokens using spaCy when a matching model is available.
    8. Drop messages that are too short after cleaning.

Input:  DataFrame produced by Parser  (columns: timestamp, author, message, msg_type, group_name)
Output: same DataFrame + columns: cleaned_message, language, tokens
"""

from __future__ import annotations

import logging
import re
import string
from typing import Optional

import pandas as pd

from whatsapp_analyzer.utils import detect_language, is_too_short, normalize_encoding

logger = logging.getLogger(__name__)

# spaCy model names mapped to ISO language codes
_SPACY_MODELS: dict[str, str] = {
    "fr": "fr_core_news_sm",
    "en": "en_core_web_sm",
}

# Fallback NLTK stopword lists by language name (used when spaCy unavailable)
_NLTK_LANG_MAP: dict[str, str] = {
    "fr": "french",
    "en": "english",
}

# Punctuation pattern compiled once
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)

# WhatsApp inline artefacts that survive the parser
_ARTEFACT_RE = re.compile(
    r"(<media omitted>|média omis|message deleted|message supprimé)",
    re.IGNORECASE,
)


class Cleaner:
    """
    Preprocess a parsed WhatsApp DataFrame for downstream NLP modules.

    Args:
        lang:           Force a language code ('fr', 'en', ...).
                        When None, language is detected per message.
        remove_emoji:   Strip emoji characters from messages.
        min_words:      Drop cleaned messages shorter than this word count.
        use_lemma:      Apply spaCy lemmatisation when a model is available.
    """

    def __init__(
        self,
        lang: Optional[str] = None,
        remove_emoji: bool = True,
        min_words: int = 3,
        use_lemma: bool = True,
    ) -> None:
        self.lang = lang
        self.remove_emoji = remove_emoji
        self.min_words = min_words
        self.use_lemma = use_lemma
        self.detected_lang: Optional[str] = lang

        self._stopwords: set[str] = set()
        self._nlp = None  # spaCy model, loaded lazily

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the full cleaning pipeline to a parsed DataFrame.

        Args:
            df: Output of Parser.parse().

        Returns:
            Filtered DataFrame with additional columns:
            cleaned_message (str), language (str), tokens (list[str]).
        """
        # Work on a copy to avoid mutating the caller's DataFrame
        result = df.copy()

        # Keep only user text messages
        result = result[result["msg_type"] == "text"].reset_index(drop=True)

        if result.empty:
            logger.warning("No text messages remaining after type filter.")
            return result

        # Detect dominant language from the full corpus when not forced
        if self.lang is None:
            self.detected_lang = self._detect_corpus_language(result["message"])
            logger.info("Detected language: %s", self.detected_lang)
        else:
            self.detected_lang = self.lang

        # Load stopwords for the detected language
        self._stopwords = self._load_stopwords(self.detected_lang)

        # Load spaCy model once if lemmatisation is enabled
        if self.use_lemma:
            self._nlp = self._load_spacy(self.detected_lang)

        result["language"] = self.detected_lang
        result["cleaned_message"] = result["message"].apply(self._clean_text)
        result["tokens"] = result["cleaned_message"].apply(str.split)

        # Drop messages that became too short after cleaning
        before = len(result)
        result = result[
            result["cleaned_message"].apply(
                lambda t: not is_too_short(t, self.min_words)
            )
        ].reset_index(drop=True)

        dropped = before - len(result)
        if dropped:
            logger.info("Dropped %d messages shorter than %d words.", dropped, self.min_words)

        logger.info("Cleaning done — %d messages retained.", len(result))
        return result

    def _clean_text(self, text: str) -> str:
        """Apply all text-level cleaning steps to a single message."""
        text = normalize_encoding(text)
        text = _ARTEFACT_RE.sub("", text)

        if self.remove_emoji:
            text = self._strip_emoji(text)

        text = text.lower()
        text = _PUNCT_RE.sub(" ", text)
        text = re.sub(r"\s+", " ", text).strip()

        tokens = text.split()
        tokens = [t for t in tokens if t not in self._stopwords and len(t) > 1]

        if self._nlp and tokens:
            tokens = self._lemmatise(tokens)

        return " ".join(tokens)

    def _lemmatise(self, tokens: list[str]) -> list[str]:
        """Return the lemma of each token using the loaded spaCy model."""
        doc = self._nlp(" ".join(tokens))
        return [token.lemma_ for token in doc if not token.is_space]

    @staticmethod
    def _strip_emoji(text: str) -> str:
        """Remove emoji characters using the emoji library when available."""
        try:
            import emoji
            return emoji.replace_emoji(text, replace="")
        except ImportError:
            # Fallback: remove characters outside the Basic Multilingual Plane
            return text.encode("ascii", errors="ignore").decode("ascii")

    @staticmethod
    def _detect_corpus_language(messages: pd.Series) -> str:
        """
        Detect the dominant language from a sample of messages.

        Samples up to 50 messages to limit detection overhead.
        Falls back to 'fr' when detection fails.
        """
        sample = messages.dropna().head(50).tolist()
        combined = " ".join(sample[:10])  # use first 10 for a quick estimate
        lang = detect_language(combined)
        return lang if lang != "unknown" else "fr"

    @staticmethod
    def _load_stopwords(lang: str) -> set[str]:
        """
        Load stopwords for the given language code.

        Tries spaCy first, then NLTK, then returns an empty set.
        """
        # Try spaCy stopwords (no model download required, part of the package)
        try:
            import spacy
            model_name = _SPACY_MODELS.get(lang)
            if model_name:
                nlp = spacy.load(model_name, disable=["parser", "ner", "tagger"])
                return nlp.Defaults.stop_words
        except Exception:
            pass

        # Try NLTK stopwords
        try:
            from nltk.corpus import stopwords as nltk_sw
            nltk_lang = _NLTK_LANG_MAP.get(lang, "english")
            return set(nltk_sw.words(nltk_lang))
        except Exception:
            pass

        logger.warning("No stopwords found for lang='%s'. Proceeding without.", lang)
        return set()

    @staticmethod
    def _load_spacy(lang: str):
        """Load a spaCy model for the given language, or return None on failure."""
        model_name = _SPACY_MODELS.get(lang)
        if not model_name:
            return None
        try:
            import spacy
            return spacy.load(model_name)
        except Exception:
            logger.warning(
                "spaCy model '%s' not found. Run: python -m spacy download %s",
                model_name,
                model_name,
            )
            return None