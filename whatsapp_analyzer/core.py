"""
Orchestrator for the WhatsApp analysis pipeline.

Chains loader → parser → cleaner → analyzers → visualizer.
All heavy module imports are lazy (inside run()) so this file loads instantly.

Output: populates ._results dict and returns it from run().
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class WhatsAppAnalyzer:
    """
    Run the full analysis pipeline on a single WhatsApp export.

    Args:
        input_path: Path to a .zip archive, _chat.txt, or folder.
        n_topics:   Number of LDA topics to extract (default 5).
        output_dir: Directory where report.html will be written.
    """

    def __init__(
        self,
        input_path: str | Path,
        n_topics: int = 5,
        output_dir: str | Path = "reports",
    ) -> None:
        self.input_path = Path(input_path)
        self.n_topics = n_topics
        self.output_dir = Path(output_dir)
        self._results: dict = {}

    def run(self) -> dict:
        """
        Execute every pipeline step and return the results dict.

        Keys populated:
            group_name, df_clean, topics, sentiment, temporal,
            users, report_path.

        Returns:
            dict with all analysis results.

        Raises:
            ValueError: if the input file cannot be loaded or parsed.
        """
        from whatsapp_analyzer.loader import Loader
        from whatsapp_analyzer.parser import Parser
        from whatsapp_analyzer.cleaner import Cleaner
        from whatsapp_analyzer.topic_classifier import TopicClassifier
        from whatsapp_analyzer.sentiment_analyzer import SentimentAnalyzer
        from whatsapp_analyzer.temporal_analyzer import TemporalAnalyzer
        from whatsapp_analyzer.user_analyzer import UserAnalyzer
        from whatsapp_analyzer.visualizer import Visualizer

        loaded = Loader().load(self.input_path)
        df = Parser().parse(loaded.chat_path)
        df = Cleaner().clean(df)

        self._results = {
            "group_name": loaded.group_name,
            "df_clean": df,
            "topics": None,
            "sentiment": None,
            "temporal": None,
            "users": None,
        }

        for step_name, step_fn in [
            ("topics", lambda: TopicClassifier(n_topics=self.n_topics).fit_transform(df)),
            ("sentiment", lambda: SentimentAnalyzer().analyze(df)),
            ("temporal", lambda: TemporalAnalyzer().analyze(df)),
        ]:
            try:
                self._results[step_name] = step_fn()
            except Exception as exc:
                logger.warning("Step '%s' failed: %s", step_name, exc)

        try:
            self._results["users"] = UserAnalyzer().build_profiles(self._results)
        except Exception as exc:
            logger.warning("Step 'users' failed: %s", exc)

        report_path = Visualizer().generate_report(self._results, self.output_dir)
        self._results["report_path"] = report_path

        return self._results
