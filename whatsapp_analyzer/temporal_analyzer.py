"""
Temporal analysis module for WhatsApp conversation analysis.

Analyses when participants are active: by hour, day of week, and month.
Pure pandas — no NLP dependencies.

Input:  DataFrame with a 'timestamp' column typed as datetime64[ns].
Output: dict with keys: timeline, hourly_heatmap, weekly_activity,
        monthly_activity, peak_hour, peak_day.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

_WEEKDAYS = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday",
]


class TemporalAnalyzer:
    """Analyse the temporal activity patterns in a cleaned WhatsApp DataFrame."""

    def analyze(self, df: pd.DataFrame) -> dict:
        """
        Compute temporal activity metrics from the cleaned DataFrame.

        Args:
            df: DataFrame with at least a 'timestamp' column (datetime64[ns]).

        Returns:
            Dict with keys:
              - 'timeline'         — DataFrame: message count per day (datetime index).
              - 'hourly_heatmap'   — DataFrame 7×24: weekday names × hours 0–23.
              - 'weekly_activity'  — Series: message count per day of week (Mon–Sun).
              - 'monthly_activity' — Series: message count per calendar month.
              - 'peak_hour'        — int (0–23), overall busiest hour.
              - 'peak_day'         — str, overall busiest day of week.

        Raises:
            ValueError: If df is empty or 'timestamp' column is missing.
        """
        if df.empty:
            raise ValueError("Input DataFrame is empty.")
        if "timestamp" not in df.columns:
            raise ValueError("DataFrame must contain a 'timestamp' column.")

        logger.info("Running temporal analysis on %d messages.", len(df))

        return {
            "timeline": self._timeline(df),
            "hourly_heatmap": self._hourly_heatmap(df),
            "weekly_activity": self._weekly_activity(df),
            "monthly_activity": self._monthly_activity(df),
            "peak_hour": self._peak_hour(df),
            "peak_day": self._peak_day(df),
        }

    @staticmethod
    def _timeline(df: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame of message counts per day (index = midnight datetime)."""
        return (
            df.groupby(df["timestamp"].dt.normalize())
            .size()
            .rename("count")
            .to_frame()
        )

    @staticmethod
    def _hourly_heatmap(df: pd.DataFrame) -> pd.DataFrame:
        """Return a 7×24 DataFrame of message counts (weekday × hour)."""
        tmp = df.assign(
            weekday=df["timestamp"].dt.day_name(),
            hour=df["timestamp"].dt.hour,
        )
        heatmap = tmp.groupby(["weekday", "hour"]).size().unstack(fill_value=0)
        return heatmap.reindex(index=_WEEKDAYS, columns=range(24), fill_value=0)

    @staticmethod
    def _weekly_activity(df: pd.DataFrame) -> pd.Series:
        """Return message counts per day of week, ordered Monday to Sunday."""
        counts = df["timestamp"].dt.day_name().value_counts()
        return counts.reindex(_WEEKDAYS, fill_value=0)

    @staticmethod
    def _monthly_activity(df: pd.DataFrame) -> pd.Series:
        """Return message counts per calendar month, ordered chronologically."""
        return df.groupby(df["timestamp"].dt.to_period("M")).size()

    @staticmethod
    def _peak_hour(df: pd.DataFrame) -> int:
        """Return the hour of day (0–23) with the most messages."""
        return int(df["timestamp"].dt.hour.value_counts().idxmax())

    @staticmethod
    def _peak_day(df: pd.DataFrame) -> str:
        """Return the day of week with the most messages."""
        return str(df["timestamp"].dt.day_name().value_counts().idxmax())
