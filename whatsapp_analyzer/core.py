"""
Orchestrator for the WhatsApp analysis pipeline.

Exposes a fluent API:

    WhatsAppAnalyzer(path)
        .parse()
        .clean()
        .analyze()
        .report()

All sub-module imports are lazy (inside method bodies) to prevent circular
imports and keep startup instant.  No NLP logic lives here — this file
only coordinates calls between specialist modules.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class WhatsAppAnalyzer:
    """
    Orchestrate the full WhatsApp analysis pipeline on a single export.

    Args:
        input_path: Path to a .zip archive, _chat.txt, or folder.
        n_topics:   Number of LDA topics to extract (default 5).
        lang:       Force a language code ('fr', 'en'). None = auto-detect.
        min_words:  Minimum token count to keep a cleaned message (default 3).
        output_dir: Default directory for report.html and CSV exports.
    """

    def __init__(
        self,
        input_path: str | Path,
        n_topics: int = 5,
        lang: str | None = None,
        min_words: int = 3,
        output_dir: str | Path = "reports",
    ) -> None:
        self.input_path = Path(input_path)
        self.n_topics = n_topics
        self.lang = lang
        self.min_words = min_words
        self.output_dir = Path(output_dir)
        self._results: dict = {}
        self._media_dir: Path | None = None

    # fluent steps 

    def parse(self) -> "WhatsAppAnalyzer":
        """
        Load and parse the WhatsApp export.

        Populates ``_results["group_name"]`` and ``_results["df_raw"]``.

        Returns:
            self (fluent)
        """
        from whatsapp_analyzer.loader import Loader
        from whatsapp_analyzer.parser import Parser

        loaded = Loader().load(self.input_path)
        df = Parser().parse(loaded.chat_path)
        self._media_dir = getattr(loaded, "media_dir", None)
        self._results["group_name"] = loaded.group_name
        self._results["df_raw"] = df
        logger.info(
            "Parsed %d messages for group '%s'.", len(df), loaded.group_name
        )
        return self

    def clean(
        self,
        lang: str | None = None,
        min_words: int | None = None,
    ) -> "WhatsAppAnalyzer":
        """
        Apply NLP preprocessing to the parsed DataFrame.

        Populates ``_results["df_clean"]``.

        Args:
            lang:      Override the instance-level language setting.
            min_words: Override the instance-level min_words setting.

        Returns:
            self (fluent)

        Raises:
            RuntimeError: if parse() has not been called yet.
        """
        from whatsapp_analyzer.cleaner import Cleaner

        df = self._results.get("df_raw")
        if df is None:
            raise RuntimeError("Call parse() before clean().")

        self._results["df_clean"] = Cleaner(
            lang=lang if lang is not None else self.lang,
            min_words=min_words if min_words is not None else self.min_words,
        ).clean(df)
        logger.info(
            "Cleaned DataFrame: %d rows retained.", len(self._results["df_clean"])
        )
        return self

    def analyze(
        self,
        topics: bool = True,
        sentiment: bool = True,
        temporal: bool = True,
        media: bool = False,
    ) -> "WhatsAppAnalyzer":
        """
        Run analysis modules in order.

        Each module failure is caught and logged — it sets the corresponding
        key to None rather than crashing the whole pipeline.

        Populates: topics, sentiment, temporal, media (optional), users.

        Args:
            topics:    Run topic modelling.
            sentiment: Run sentiment analysis.
            temporal:  Run temporal analysis.
            media:     Run media analysis (requires a media_dir from parse()).

        Returns:
            self (fluent)

        Raises:
            RuntimeError: if clean() has not been called yet.
        """
        if self._results.get("df_clean") is None:
            raise RuntimeError("Call clean() before analyze().")

        df = self._results["df_clean"]

        if topics:
            self._run_step("topics", self._step_topics, df)
        if sentiment:
            self._run_step("sentiment", self._step_sentiment, df)
        if temporal:
            self._run_step("temporal", self._step_temporal, df)
        if media and self._media_dir is not None:
            self._run_step("media", self._step_media, self._media_dir)

        self._run_step("users", self._step_users)
        return self

    def report(self, output: str | Path | None = None) -> Path:
        """
        Generate a self-contained HTML report.

        Args:
            output: Output directory. Falls back to self.output_dir.

        Returns:
            Path to the written report.html file.
        """
        from whatsapp_analyzer.visualizer import Visualizer

        return Visualizer().generate_report(
            self._results,
            Path(output) if output else self.output_dir,
        )

    def to_csv(self, output: str | Path | None = None) -> Path:
        """
        Export the enriched DataFrame to a CSV file.

        Args:
            output: Output directory. Falls back to self.output_dir.

        Returns:
            Path to the written CSV file.

        Raises:
            RuntimeError: if clean() has not been called yet.
        """
        df = self._results.get("df_clean")
        if df is None:
            raise RuntimeError("No cleaned data — call clean() first.")

        out_dir = Path(output) if output else self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        group = self._results.get("group_name", "export")
        path = out_dir / f"{group}.csv"
        df.to_csv(path, index=False, encoding="utf-8")
        logger.info("Exported CSV to %s", path)
        return path

    def user(self, author: str) -> "UserView":
        """
        Return a UserView scoped to a single author.

        Args:
            author: Author name as it appears in the DataFrame.

        Returns:
            UserView for the given author.
        """
        return UserView(author, self._results)

    # convenience 

    def run(self) -> dict:
        """
        Convenience: chain parse → clean → analyze → report in one call.

        Returns:
            Populated _results dict (also stored on self._results).
        """
        self.parse().clean().analyze()
        self._results["report_path"] = self.report()
        return self._results

    # private helpers 
    def _run_step(self, key: str, fn, *args) -> None:
        """Execute fn(*args), store result in _results[key], log on failure."""
        try:
            self._results[key] = fn(*args)
        except Exception as exc:
            logger.warning("Step '%s' failed: %s", key, exc)
            self._results[key] = None

    def _step_topics(self, df):
        from whatsapp_analyzer.topic_classifier import TopicClassifier
        return TopicClassifier(n_topics=self.n_topics).fit_transform(df)

    def _step_sentiment(self, df):
        from whatsapp_analyzer.sentiment_analyzer import SentimentAnalyzer
        return SentimentAnalyzer().analyze(df)

    def _step_temporal(self, df):
        from whatsapp_analyzer.temporal_analyzer import TemporalAnalyzer
        return TemporalAnalyzer().analyze(df)

    def _step_media(self, media_dir):
        from whatsapp_analyzer.media_analyzer import MediaAnalyzer
        return MediaAnalyzer().analyze(media_dir)

    def _step_users(self):
        from whatsapp_analyzer.user_analyzer import UserAnalyzer
        return UserAnalyzer().build_profiles(self._results)


class UserView:
    """
    Scoped view of analysis results for a single author.

    Delegates all computation to UserAnalyzer static methods.
    """

    def __init__(self, author: str, results: dict) -> None:
        self._author = author
        self._results = results

    def summary(self) -> dict:
        """Return the full profile dict for this author."""
        from whatsapp_analyzer.user_analyzer import UserAnalyzer
        return UserAnalyzer.summary_for(self._author, self._results)

    def topics(self):
        """Return topic assignment DataFrame for this author."""
        from whatsapp_analyzer.user_analyzer import UserAnalyzer
        return UserAnalyzer.topics_for(self._author, self._results)

    def sentiment_over_time(self):
        """Return sentiment-over-time DataFrame for this author."""
        from whatsapp_analyzer.user_analyzer import UserAnalyzer
        return UserAnalyzer.sentiment_over_time_for(self._author, self._results)

    def activity_heatmap(self):
        """Return 7×24 activity heatmap DataFrame for this author."""
        from whatsapp_analyzer.user_analyzer import UserAnalyzer
        return UserAnalyzer.activity_heatmap_for(self._author, self._results)
