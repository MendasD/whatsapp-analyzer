"""
User profiling module for WhatsApp conversation analysis.

Builds per-user profiles by aggregating from the results dict produced by
the analysis pipeline. No NLP logic — pure aggregation over the results dict.

Input:  results dict from core.py (keys: df_clean, topics, sentiment)
Output: dict mapping author names to profile dicts
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

_WEEKDAYS = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday",
]


class UserAnalyzer:
    """
    Build per-user profiles from pipeline results.

    Reads exclusively from the results dict — never re-runs NLP steps.
    """

    def build_profiles(self, results: dict) -> dict[str, dict]:
        """
        Build a profile dict for every author found in results["df_clean"].

        Args:
            results: Pipeline results dict with keys 'df_clean', 'topics',
                     'sentiment'.

        Returns:
            Dict mapping each author name to a profile dict with keys:
            message_count, avg_message_length, activity_hours,
            most_active_day, top_topics, sentiment_mean.
        """
        df = results["df_clean"]
        topics_result = results.get("topics")
        sentiment_result = results.get("sentiment")

        profiles: dict[str, dict] = {}
        for author in df["author"].unique():
            profiles[author] = _build_author_profile(
                author, df, topics_result, sentiment_result
            )
            logger.debug("Built profile for author: %s", author)

        logger.info("Built profiles for %d authors.", len(profiles))
        return profiles

    @staticmethod
    def summary_for(author: str, results: dict) -> dict:
        """
        Return the profile dict for a single author.

        Args:
            author:  Author name to look up.
            results: Pipeline results dict.

        Returns:
            Profile dict, or empty dict if the author is not found.
        """
        profiles = UserAnalyzer().build_profiles(results)
        return profiles.get(author, {})

    @staticmethod
    def topics_for(author: str, results: dict) -> pd.DataFrame:
        """
        Return a DataFrame of topic assignments for a single author.

        Args:
            author:  Author name.
            results: Pipeline results dict (must contain 'topics').

        Returns:
            DataFrame with columns topic_id, topic_label, topic_score.
            Empty DataFrame if topics were not computed.
        """
        topics_result = results.get("topics")
        if topics_result is None:
            return pd.DataFrame(columns=["topic_id", "topic_label", "topic_score"])

        topics_df = topics_result["df"]
        group_topics = topics_result["group_topics"]

        author_df = topics_df[topics_df["author"] == author][
            ["topic_id", "topic_score"]
        ].copy()

        if author_df.empty:
            return pd.DataFrame(columns=["topic_id", "topic_label", "topic_score"])

        merged = author_df.merge(
            group_topics[["topic_id", "topic_label"]], on="topic_id", how="left"
        )
        return merged[["topic_id", "topic_label", "topic_score"]].reset_index(drop=True)

    @staticmethod
    def sentiment_over_time_for(author: str, results: dict) -> pd.DataFrame:
        """
        Return sentiment scores over time for a single author.

        Args:
            author:  Author name.
            results: Pipeline results dict (must contain 'sentiment').

        Returns:
            DataFrame with columns timestamp, sentiment_score, sentiment_label.
            Empty DataFrame if sentiment was not computed.
        """
        sentiment_result = results.get("sentiment")
        if sentiment_result is None:
            return pd.DataFrame(
                columns=["timestamp", "sentiment_score", "sentiment_label"]
            )

        sentiment_df = sentiment_result["df"]
        author_df = sentiment_df[sentiment_df["author"] == author][
            ["timestamp", "sentiment_score", "sentiment_label"]
        ]
        return author_df.reset_index(drop=True)

    @staticmethod
    def activity_heatmap_for(author: str, results: dict) -> pd.DataFrame:
        """
        Return a 7×24 activity heatmap (weekday × hour) for a single author.

        Args:
            author:  Author name.
            results: Pipeline results dict (must contain 'df_clean').

        Returns:
            DataFrame with weekday names as index and hours 0–23 as columns.
            Cell values are message counts. Missing slots are filled with 0.
        """
        df = results["df_clean"]
        author_df = df[df["author"] == author]

        empty = pd.DataFrame(0, index=_WEEKDAYS, columns=range(24))
        if author_df.empty:
            return empty

        tmp = author_df.copy()
        tmp["weekday"] = tmp["timestamp"].dt.day_name()
        tmp["hour"] = tmp["timestamp"].dt.hour

        heatmap = tmp.groupby(["weekday", "hour"]).size().unstack(fill_value=0)
        return heatmap.reindex(index=_WEEKDAYS, columns=range(24), fill_value=0)


def _build_author_profile(
    author: str,
    df: pd.DataFrame,
    topics_result: Optional[dict],
    sentiment_result: Optional[dict],
) -> dict:
    """Compute the full profile dict for a single author."""
    author_df = df[df["author"] == author]

    message_count = len(author_df)

    if "tokens" in author_df.columns:
        avg_message_length = float(
            author_df["tokens"]
            .apply(lambda t: len(t) if isinstance(t, list) else 0)
            .mean()
        )
    else:
        avg_message_length = float(
            author_df["cleaned_message"].apply(lambda t: len(str(t).split())).mean()
        )

    activity_hours: dict[int, int] = (
        author_df["timestamp"].dt.hour.value_counts().sort_index().to_dict()
    )

    day_counts = author_df["timestamp"].dt.day_name().value_counts()
    most_active_day: Optional[str] = (
        day_counts.idxmax() if not day_counts.empty else None
    )

    top_topics = _extract_top_topics(author, topics_result)

    sentiment_mean: Optional[float] = None
    if sentiment_result is not None:
        scores = sentiment_result["df"]
        author_scores = scores[scores["author"] == author]["sentiment_score"]
        if not author_scores.empty:
            sentiment_mean = float(author_scores.mean())

    return {
        "message_count": message_count,
        "avg_message_length": avg_message_length,
        "activity_hours": activity_hours,
        "most_active_day": most_active_day,
        "top_topics": top_topics,
        "sentiment_mean": sentiment_mean,
    }


def _extract_top_topics(
    author: str, topics_result: Optional[dict]
) -> list[tuple[str, float]]:
    """Return (label, mean_score) tuples sorted by message count descending."""
    if topics_result is None:
        return []

    topics_df = topics_result["df"]
    group_topics = topics_result["group_topics"]
    author_df = topics_df[topics_df["author"] == author]

    if author_df.empty:
        return []

    agg = (
        author_df.groupby("topic_id")
        .agg(score=("topic_score", "mean"), count=("topic_score", "count"))
        .reset_index()
    )
    agg = agg.merge(
        group_topics[["topic_id", "topic_label"]], on="topic_id", how="left"
    )
    agg = agg.sort_values("count", ascending=False)

    return [(str(label), float(score)) for label, score in zip(agg["topic_label"], agg["score"])]
