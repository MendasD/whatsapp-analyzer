"""
Tests for comparator.py.

All WhatsAppAnalyzer instances are replaced by MagicMock objects whose
._results attribute is a plain dict built in this file.  No pipeline code
runs during these tests.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from whatsapp_analyzer.comparator import GroupComparator


# Helpers

def _make_df(timestamps, authors, group_name):
    return pd.DataFrame({
        "timestamp": pd.to_datetime(timestamps),
        "author": authors,
        "message": ["msg"] * len(timestamps),
        "msg_type": ["text"] * len(timestamps),
        "group_name": [group_name] * len(timestamps),
    })


def _make_topics_result(labels, weights):
    group_topics = pd.DataFrame({
        "topic_id": list(range(len(labels))),
        "topic_label": labels,
        "weight": weights,
    })
    return {"group_topics": group_topics}


def _make_sentiment_result(mean, pos_pct, neg_pct):
    return {
        "global": {"mean": mean, "pos_pct": pos_pct, "neg_pct": neg_pct},
    }


def _make_mock_analyzer(group_name, timestamps, authors,
                        topics_labels=None, topics_weights=None,
                        sentiment_mean=0.0, sentiment_pos=0.0, sentiment_neg=0.0):
    df = _make_df(timestamps, authors, group_name)
    results = {
        "group_name": group_name,
        "df_clean": df,
        "topics": (
            _make_topics_result(topics_labels, topics_weights)
            if topics_labels is not None else None
        ),
        "sentiment": _make_sentiment_result(sentiment_mean, sentiment_pos, sentiment_neg),
    }
    az = MagicMock()
    az._results = results
    return az


# Two groups that share one author ("Aminata")
_AZ_ETUDES = _make_mock_analyzer(
    group_name="Groupe Etudes",
    timestamps=["2024-01-10 09:00", "2024-01-10 10:00",
                "2024-01-11 08:00", "2024-01-12 11:00"],
    authors=["Aminata", "Moussa", "Aminata", "Fatou"],
    topics_labels=["sport / match / jouer", "cours / td / examen"],
    topics_weights=[0.6, 0.4],
    sentiment_mean=0.3, sentiment_pos=0.75, sentiment_neg=0.05,
)

_AZ_FAMILLE = _make_mock_analyzer(
    group_name="Famille",
    timestamps=["2024-01-15 07:00", "2024-01-16 08:00", "2024-01-16 09:00"],
    authors=["Aminata", "Mamadou", "Mamadou"],
    topics_labels=["sport / match / jouer", "famille / maison / repas"],
    topics_weights=[0.3, 0.7],
    sentiment_mean=0.5, sentiment_pos=0.9, sentiment_neg=0.0,
)

_AZ_NO_TOPICS = _make_mock_analyzer(
    group_name="Travail",
    timestamps=["2024-02-01 10:00"],
    authors=["Kader"],
    topics_labels=None,
)

_AZ_NO_SENTIMENT = _make_mock_analyzer(
    group_name="Amis",
    timestamps=["2024-02-02 10:00"],
    authors=["Sonia"],
)
_AZ_NO_SENTIMENT._results["sentiment"] = None


# compare_topics

class TestCompareTopics:

    def test_returns_dataframe(self):
        result = GroupComparator([_AZ_ETUDES, _AZ_FAMILLE]).compare_topics()
        assert isinstance(result, pd.DataFrame)

    def test_row_per_group(self):
        result = GroupComparator([_AZ_ETUDES, _AZ_FAMILLE]).compare_topics()
        assert len(result) == 2

    def test_index_contains_group_names(self):
        result = GroupComparator([_AZ_ETUDES, _AZ_FAMILLE]).compare_topics()
        assert "Groupe Etudes" in result.index
        assert "Famille" in result.index

    def test_columns_are_topic_labels(self):
        result = GroupComparator([_AZ_ETUDES, _AZ_FAMILLE]).compare_topics()
        assert "sport / match / jouer" in result.columns
        assert "cours / td / examen" in result.columns
        assert "famille / maison / repas" in result.columns

    def test_known_weight_value(self):
        result = GroupComparator([_AZ_ETUDES, _AZ_FAMILLE]).compare_topics()
        assert result.loc["Groupe Etudes", "sport / match / jouer"] == pytest.approx(0.6)

    def test_missing_topic_filled_with_zero(self):
        result = GroupComparator([_AZ_ETUDES, _AZ_FAMILLE]).compare_topics()
        assert result.loc["Groupe Etudes", "famille / maison / repas"] == pytest.approx(0.0)

    def test_skips_analyzer_without_topics(self):
        result = GroupComparator([_AZ_ETUDES, _AZ_NO_TOPICS]).compare_topics()
        assert "Travail" not in result.index
        assert "Groupe Etudes" in result.index

    def test_empty_when_no_topics_at_all(self):
        result = GroupComparator([_AZ_NO_TOPICS]).compare_topics()
        assert result.empty

    def test_empty_list_returns_empty_dataframe(self):
        result = GroupComparator([]).compare_topics()
        assert result.empty


# compare_activity

class TestCompareActivity:

    def test_returns_dataframe(self):
        result = GroupComparator([_AZ_ETUDES, _AZ_FAMILLE]).compare_activity()
        assert isinstance(result, pd.DataFrame)

    def test_row_per_group(self):
        result = GroupComparator([_AZ_ETUDES, _AZ_FAMILLE]).compare_activity()
        assert len(result) == 2

    def test_index_contains_group_names(self):
        result = GroupComparator([_AZ_ETUDES, _AZ_FAMILLE]).compare_activity()
        assert "Groupe Etudes" in result.index
        assert "Famille" in result.index

    def test_has_required_columns(self):
        result = GroupComparator([_AZ_ETUDES, _AZ_FAMILLE]).compare_activity()
        for col in ("nb_messages", "nb_participants", "msgs_per_day",
                    "period_start", "period_end"):
            assert col in result.columns

    def test_nb_messages_correct(self):
        result = GroupComparator([_AZ_ETUDES]).compare_activity()
        assert result.loc["Groupe Etudes", "nb_messages"] == 4

    def test_nb_participants_correct(self):
        result = GroupComparator([_AZ_ETUDES]).compare_activity()
        # Aminata, Moussa, Fatou → 3 participants
        assert result.loc["Groupe Etudes", "nb_participants"] == 3

    def test_msgs_per_day_positive(self):
        result = GroupComparator([_AZ_ETUDES]).compare_activity()
        assert result.loc["Groupe Etudes", "msgs_per_day"] > 0

    def test_period_start_before_period_end(self):
        result = GroupComparator([_AZ_ETUDES]).compare_activity()
        assert result.loc["Groupe Etudes", "period_start"] < result.loc["Groupe Etudes", "period_end"]

    def test_single_day_msgs_per_day_equals_nb_messages(self):
        az = _make_mock_analyzer(
            group_name="OneDay",
            timestamps=["2024-03-01 08:00", "2024-03-01 09:00"],
            authors=["Alice", "Bob"],
        )
        result = GroupComparator([az]).compare_activity()
        # start == end → days = max(0, 1) = 1 → msgs_per_day = 2/1 = 2.0
        assert result.loc["OneDay", "msgs_per_day"] == pytest.approx(2.0)

    def test_empty_list_returns_empty_dataframe(self):
        result = GroupComparator([]).compare_activity()
        assert result.empty


# compare_sentiment

class TestCompareSentiment:

    def test_returns_dataframe(self):
        result = GroupComparator([_AZ_ETUDES, _AZ_FAMILLE]).compare_sentiment()
        assert isinstance(result, pd.DataFrame)

    def test_row_per_group(self):
        result = GroupComparator([_AZ_ETUDES, _AZ_FAMILLE]).compare_sentiment()
        assert len(result) == 2

    def test_index_contains_group_names(self):
        result = GroupComparator([_AZ_ETUDES, _AZ_FAMILLE]).compare_sentiment()
        assert "Groupe Etudes" in result.index
        assert "Famille" in result.index

    def test_has_required_columns(self):
        result = GroupComparator([_AZ_ETUDES, _AZ_FAMILLE]).compare_sentiment()
        for col in ("sentiment_mean", "pos_pct", "neg_pct"):
            assert col in result.columns

    def test_sentiment_mean_correct(self):
        result = GroupComparator([_AZ_ETUDES]).compare_sentiment()
        assert result.loc["Groupe Etudes", "sentiment_mean"] == pytest.approx(0.3)

    def test_pos_pct_correct(self):
        result = GroupComparator([_AZ_ETUDES]).compare_sentiment()
        assert result.loc["Groupe Etudes", "pos_pct"] == pytest.approx(0.75)

    def test_neg_pct_correct(self):
        result = GroupComparator([_AZ_ETUDES]).compare_sentiment()
        assert result.loc["Groupe Etudes", "neg_pct"] == pytest.approx(0.05)

    def test_skips_analyzer_without_sentiment(self):
        result = GroupComparator([_AZ_ETUDES, _AZ_NO_SENTIMENT]).compare_sentiment()
        assert "Amis" not in result.index
        assert "Groupe Etudes" in result.index

    def test_empty_when_no_sentiment_at_all(self):
        result = GroupComparator([_AZ_NO_SENTIMENT]).compare_sentiment()
        assert result.empty

    def test_empty_list_returns_empty_dataframe(self):
        result = GroupComparator([]).compare_sentiment()
        assert result.empty


# common_users

class TestCommonUsers:

    def test_returns_dataframe(self):
        result = GroupComparator([_AZ_ETUDES, _AZ_FAMILLE]).common_users()
        assert isinstance(result, pd.DataFrame)

    def test_has_author_and_groups_columns(self):
        result = GroupComparator([_AZ_ETUDES, _AZ_FAMILLE]).common_users()
        assert "author" in result.columns
        assert "groups" in result.columns

    def test_identifies_shared_author(self):
        result = GroupComparator([_AZ_ETUDES, _AZ_FAMILLE]).common_users()
        assert "Aminata" in result["author"].values

    def test_excludes_unique_authors(self):
        result = GroupComparator([_AZ_ETUDES, _AZ_FAMILLE]).common_users()
        for name in ("Moussa", "Fatou", "Mamadou"):
            assert name not in result["author"].values

    def test_groups_column_contains_both_group_names(self):
        result = GroupComparator([_AZ_ETUDES, _AZ_FAMILLE]).common_users()
        row = result[result["author"] == "Aminata"].iloc[0]
        assert "Groupe Etudes" in row["groups"]
        assert "Famille" in row["groups"]

    def test_no_overlap_returns_empty_dataframe_with_correct_columns(self):
        az_a = _make_mock_analyzer("A", ["2024-01-01 08:00"], ["Alice"])
        az_b = _make_mock_analyzer("B", ["2024-01-01 09:00"], ["Bob"])
        result = GroupComparator([az_a, az_b]).common_users()
        assert result.empty
        assert "author" in result.columns
        assert "groups" in result.columns

    def test_single_analyzer_returns_empty_dataframe(self):
        result = GroupComparator([_AZ_ETUDES]).common_users()
        assert result.empty

    def test_empty_list_returns_empty_dataframe(self):
        result = GroupComparator([]).common_users()
        assert result.empty


# report

class TestReport:

    def test_report_returns_path(self, tmp_path):
        mock_viz_cls = MagicMock()
        mock_viz_instance = MagicMock()
        mock_viz_cls.return_value = mock_viz_instance
        mock_viz_instance.generate_comparison_report.return_value = (
            tmp_path / "comparison_report.html"
        )
        with patch("whatsapp_analyzer.visualizer.Visualizer", mock_viz_cls):
            result = GroupComparator([_AZ_ETUDES]).report(tmp_path)
        assert isinstance(result, Path)

    def test_report_delegates_to_visualizer(self, tmp_path):
        mock_viz_cls = MagicMock()
        mock_viz_instance = MagicMock()
        mock_viz_cls.return_value = mock_viz_instance
        mock_viz_instance.generate_comparison_report.return_value = (
            tmp_path / "comparison_report.html"
        )
        with patch("whatsapp_analyzer.visualizer.Visualizer", mock_viz_cls):
            GroupComparator([_AZ_ETUDES]).report(tmp_path)
        mock_viz_instance.generate_comparison_report.assert_called_once()
