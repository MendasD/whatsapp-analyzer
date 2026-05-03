"""
Multi-group comparison module for WhatsApp conversation analysis.

Reads exclusively from each analyzer's ._results dict — never re-runs
NLP steps. The Visualizer is imported lazily in report() to avoid a
circular import (visualizer → utils, comparator → visualizer).

Input:  list of WhatsAppAnalyzer instances, each with a populated ._results
Output: comparison DataFrames and an HTML report
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class GroupComparator:
    """
    Compare activity, topics, sentiment, and users across several groups.

    Args:
        analyzers: List of WhatsAppAnalyzer instances to compare.
                   Each must have a ._results dict populated by the pipeline.
    """

    def __init__(self, analyzers: list) -> None:
        self.analyzers = analyzers

    def compare_topics(self) -> pd.DataFrame:
        """
        Return a pivot table of topic weights across groups.

        Rows are group names, columns are topic labels, values are weights.
        Groups that do not have a given topic label receive 0.

        Returns:
            DataFrame with group names as index, topic labels as columns.
            Empty DataFrame if no analyzer has topic data.
        """
        rows = []
        for az in self.analyzers:
            results = az._results
            group_name = results.get("group_name", "Unknown")
            topics_result = results.get("topics")
            if topics_result is None:
                logger.debug("No topic data for group '%s'; skipping.", group_name)
                continue
            row: dict = {"group": group_name}
            for _, r in topics_result["group_topics"].iterrows():
                row[r["topic_label"]] = float(r["weight"])
            rows.append(row)

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(rows).set_index("group").fillna(0.0)

    def compare_activity(self) -> pd.DataFrame:
        """
        Return one activity-summary row per group.

        Columns: nb_messages, nb_participants, msgs_per_day,
                 period_start, period_end.

        Returns:
            DataFrame with group names as index.
            Empty DataFrame if no analyzer has clean message data.
        """
        rows = []
        for az in self.analyzers:
            results = az._results
            group_name = results.get("group_name", "Unknown")
            df = results.get("df_clean")
            if df is None or df.empty:
                logger.debug("No df_clean for group '%s'; skipping.", group_name)
                continue

            period_start = df["timestamp"].min()
            period_end = df["timestamp"].max()
            days = max((period_end - period_start).days, 1)

            rows.append({
                "group": group_name,
                "nb_messages": len(df),
                "nb_participants": int(df["author"].nunique()),
                "msgs_per_day": round(len(df) / days, 2),
                "period_start": period_start,
                "period_end": period_end,
            })

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(rows).set_index("group")

    def compare_sentiment(self) -> pd.DataFrame:
        """
        Return one sentiment-summary row per group.

        Columns: sentiment_mean, pos_pct, neg_pct.

        Returns:
            DataFrame with group names as index.
            Empty DataFrame if no analyzer has sentiment data.
        """
        rows = []
        for az in self.analyzers:
            results = az._results
            group_name = results.get("group_name", "Unknown")
            sentiment_result = results.get("sentiment")
            if sentiment_result is None:
                logger.debug("No sentiment data for group '%s'; skipping.", group_name)
                continue

            g = sentiment_result.get("global", {})
            rows.append({
                "group": group_name,
                "sentiment_mean": float(g.get("mean", 0.0)),
                "pos_pct": float(g.get("pos_pct", 0.0)),
                "neg_pct": float(g.get("neg_pct", 0.0)),
            })

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(rows).set_index("group")

    def common_users(self) -> pd.DataFrame:
        """
        Return authors present in more than one group.

        Returns:
            DataFrame with columns 'author' and 'groups' (list of group names).
            Empty DataFrame (with those columns) if no author appears in 2+ groups.
        """
        author_groups: dict[str, list[str]] = {}

        for az in self.analyzers:
            results = az._results
            group_name = results.get("group_name", "Unknown")
            df = results.get("df_clean")
            if df is None:
                continue
            for author in df["author"].unique():
                author_groups.setdefault(author, []).append(group_name)

        rows = [
            {"author": author, "groups": groups}
            for author, groups in author_groups.items()
            if len(groups) > 1
        ]

        if not rows:
            return pd.DataFrame(columns=["author", "groups"])

        return pd.DataFrame(rows)

    def report(self, output: Path) -> Path:
        """
        Generate a multi-group comparison HTML report.

        Delegates rendering to Visualizer.generate_comparison_report().

        Args:
            output: Directory where comparison_report.html will be written.

        Returns:
            Path to the written file.
        """
        from whatsapp_analyzer.visualizer import Visualizer

        return Visualizer().generate_comparison_report(self.analyzers, Path(output))
