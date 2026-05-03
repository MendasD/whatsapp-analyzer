"""
Tests for visualizer.py.

matplotlib and seaborn are mocked throughout so this file runs in complete
isolation — no rendering, no display, no file writes except via tmp_path.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch, patch as mock_patch

import pandas as pd
import pytest

from whatsapp_analyzer.visualizer import Visualizer

# Patch targets
_SUBPLOTS = "matplotlib.pyplot.subplots"
_MPL_CLOSE = "matplotlib.pyplot.close"
_SNS_HEATMAP = "seaborn.heatmap"
_FIG_TO_BASE64 = "whatsapp_analyzer.visualizer._fig_to_base64"

_WEEKDAYS = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday",
]


def _make_df_clean() -> pd.DataFrame:
    timestamps = pd.to_datetime([
        "2024-01-12 08:00", "2024-01-12 10:00",
        "2024-01-13 09:00", "2024-01-13 14:00",
    ])
    return pd.DataFrame({
        "timestamp": timestamps,
        "author": ["Aminata", "Moussa", "Aminata", "Moussa"],
        "message": ["msg1", "msg2", "msg3", "msg4"],
        "msg_type": ["text"] * 4,
        "group_name": ["TestGroup"] * 4,
        "cleaned_message": ["word one two", "word alpha beta",
                            "word three four", "word gamma delta"],
        "language": ["fr"] * 4,
        "tokens": [["word", "one", "two"], ["word", "alpha", "beta"],
                   ["word", "three", "four"], ["word", "gamma", "delta"]],
    })


def _make_results(with_topics: bool = True,
                  with_sentiment: bool = True,
                  with_temporal: bool = True) -> dict:
    df_clean = _make_df_clean()

    topics = None
    if with_topics:
        topics_df = df_clean.copy()
        topics_df["topic_id"] = [0, 1, 0, 1]
        topics_df["topic_score"] = [0.8, 0.7, 0.75, 0.9]
        group_topics = pd.DataFrame({
            "topic_id": [0, 1],
            "topic_label": [
                "sport / match / jouer / gagner / equipe",
                "cours / td / examen / prof / notes",
            ],
            "weight": [0.5, 0.5],
        })
        topics = {"df": topics_df, "group_topics": group_topics}

    sentiment = None
    if with_sentiment:
        sentiment_df = df_clean.copy()
        sentiment_df["sentiment_score"] = [0.3, -0.1, 0.5, 0.2]
        sentiment_df["sentiment_label"] = ["positive", "neutral", "positive", "positive"]
        sentiment = {
            "df": sentiment_df,
            "by_user": sentiment_df.groupby("author")["sentiment_score"].mean().reset_index(),
            "global": {"mean": 0.2, "pos_pct": 0.75, "neg_pct": 0.0},
        }

    temporal = None
    if with_temporal:
        heatmap = pd.DataFrame(0, index=_WEEKDAYS, columns=range(24))
        heatmap.loc["Friday", 8] = 2
        heatmap.loc["Saturday", 9] = 2
        temporal = {
            "timeline": pd.DataFrame(
                {"count": [2, 2]},
                index=pd.to_datetime(["2024-01-12", "2024-01-13"]),
            ),
            "hourly_heatmap": heatmap,
            "weekly_activity": pd.Series(
                [0, 0, 0, 0, 2, 2, 0], index=_WEEKDAYS
            ),
            "monthly_activity": pd.Series(
                [4],
                index=pd.period_range("2024-01", periods=1, freq="M"),
            ),
            "peak_hour": 8,
            "peak_day": "Friday",
        }

    return {
        "df_clean": df_clean,
        "group_name": "TestGroup",
        "topics": topics,
        "sentiment": sentiment,
        "temporal": temporal,
    }


def _mock_subplots():
    """Return (mock_fig, mock_ax) for patching plt.subplots."""
    return MagicMock(), MagicMock()


# --- plot_topic_distribution ---

def test_plot_topic_distribution_returns_figure():
    mock_fig, mock_ax = _mock_subplots()
    with patch(_SUBPLOTS, return_value=(mock_fig, mock_ax)):
        result = Visualizer().plot_topic_distribution(_make_results())
    assert result is mock_fig


def test_plot_topic_distribution_handles_missing_topics():
    mock_fig, mock_ax = _mock_subplots()
    results = _make_results(with_topics=False)
    with patch(_SUBPLOTS, return_value=(mock_fig, mock_ax)):
        result = Visualizer().plot_topic_distribution(results)
    assert result is mock_fig


# --- plot_wordcloud ---

def test_plot_wordcloud_returns_figure():
    mock_fig, mock_ax = _mock_subplots()
    with patch(_SUBPLOTS, return_value=(mock_fig, mock_ax)), \
         patch.dict("sys.modules", {"wordcloud": MagicMock()}):
        result = Visualizer().plot_wordcloud(_make_results(), topic_id=0)
    assert result is mock_fig


def test_plot_wordcloud_handles_missing_topics():
    mock_fig, mock_ax = _mock_subplots()
    results = _make_results(with_topics=False)
    with patch(_SUBPLOTS, return_value=(mock_fig, mock_ax)):
        result = Visualizer().plot_wordcloud(results, topic_id=0)
    assert result is mock_fig


def test_plot_wordcloud_handles_unknown_topic_id():
    mock_fig, mock_ax = _mock_subplots()
    with patch(_SUBPLOTS, return_value=(mock_fig, mock_ax)), \
         patch.dict("sys.modules", {"wordcloud": MagicMock()}):
        result = Visualizer().plot_wordcloud(_make_results(), topic_id=99)
    assert result is mock_fig


def test_plot_wordcloud_falls_back_when_wordcloud_not_installed():
    mock_fig, mock_ax = _mock_subplots()
    with patch(_SUBPLOTS, return_value=(mock_fig, mock_ax)), \
         patch.dict("sys.modules", {"wordcloud": None}):
        result = Visualizer().plot_wordcloud(_make_results(), topic_id=0)
    assert result is mock_fig


# --- plot_sentiment_timeline ---

def test_plot_sentiment_timeline_returns_figure():
    mock_fig, mock_ax = _mock_subplots()
    with patch(_SUBPLOTS, return_value=(mock_fig, mock_ax)):
        result = Visualizer().plot_sentiment_timeline(_make_results())
    assert result is mock_fig


def test_plot_sentiment_timeline_handles_missing_sentiment():
    mock_fig, mock_ax = _mock_subplots()
    results = _make_results(with_sentiment=False)
    with patch(_SUBPLOTS, return_value=(mock_fig, mock_ax)):
        result = Visualizer().plot_sentiment_timeline(results)
    assert result is mock_fig


# --- plot_user_activity ---

def test_plot_user_activity_returns_figure():
    mock_fig, mock_ax = _mock_subplots()
    with patch(_SUBPLOTS, return_value=(mock_fig, mock_ax)):
        result = Visualizer().plot_user_activity(_make_results())
    assert result is mock_fig


def test_plot_user_activity_handles_missing_df_clean():
    mock_fig, mock_ax = _mock_subplots()
    results = _make_results()
    results["df_clean"] = None
    with patch(_SUBPLOTS, return_value=(mock_fig, mock_ax)):
        result = Visualizer().plot_user_activity(results)
    assert result is mock_fig


# --- plot_hourly_heatmap ---

def test_plot_hourly_heatmap_returns_figure():
    mock_fig, mock_ax = _mock_subplots()
    with patch(_SUBPLOTS, return_value=(mock_fig, mock_ax)), \
         patch(_SNS_HEATMAP):
        result = Visualizer().plot_hourly_heatmap(_make_results())
    assert result is mock_fig


def test_plot_hourly_heatmap_calls_seaborn_heatmap():
    mock_fig, mock_ax = _mock_subplots()
    with patch(_SUBPLOTS, return_value=(mock_fig, mock_ax)), \
         patch(_SNS_HEATMAP) as mock_sns:
        Visualizer().plot_hourly_heatmap(_make_results())
    mock_sns.assert_called_once()


def test_plot_hourly_heatmap_handles_missing_temporal():
    mock_fig, mock_ax = _mock_subplots()
    results = _make_results(with_temporal=False)
    with patch(_SUBPLOTS, return_value=(mock_fig, mock_ax)), \
         patch(_SNS_HEATMAP):
        result = Visualizer().plot_hourly_heatmap(results)
    assert result is mock_fig


# --- generate_report ---

def test_generate_report_returns_path(tmp_path):
    with patch(_SUBPLOTS, return_value=_mock_subplots()), \
         patch(_SNS_HEATMAP), \
         patch.dict("sys.modules", {"wordcloud": MagicMock()}), \
         patch(_FIG_TO_BASE64, return_value="testbase64"):
        result = Visualizer().generate_report(_make_results(), tmp_path)
    assert isinstance(result, Path)


def test_generate_report_creates_html_file(tmp_path):
    with patch(_SUBPLOTS, return_value=_mock_subplots()), \
         patch(_SNS_HEATMAP), \
         patch.dict("sys.modules", {"wordcloud": MagicMock()}), \
         patch(_FIG_TO_BASE64, return_value="testbase64"):
        path = Visualizer().generate_report(_make_results(), tmp_path)
    assert path.exists()


def test_generate_report_filename_is_report_html(tmp_path):
    with patch(_SUBPLOTS, return_value=_mock_subplots()), \
         patch(_SNS_HEATMAP), \
         patch.dict("sys.modules", {"wordcloud": MagicMock()}), \
         patch(_FIG_TO_BASE64, return_value="testbase64"):
        path = Visualizer().generate_report(_make_results(), tmp_path)
    assert path.name == "report.html"


def test_generate_report_html_contains_group_name(tmp_path):
    with patch(_SUBPLOTS, return_value=_mock_subplots()), \
         patch(_SNS_HEATMAP), \
         patch.dict("sys.modules", {"wordcloud": MagicMock()}), \
         patch(_FIG_TO_BASE64, return_value="testbase64"):
        path = Visualizer().generate_report(_make_results(), tmp_path)
    assert "TestGroup" in path.read_text(encoding="utf-8")


def test_generate_report_html_embeds_base64_images(tmp_path):
    with patch(_SUBPLOTS, return_value=_mock_subplots()), \
         patch(_SNS_HEATMAP), \
         patch.dict("sys.modules", {"wordcloud": MagicMock()}), \
         patch(_FIG_TO_BASE64, return_value="testbase64"):
        path = Visualizer().generate_report(_make_results(), tmp_path)
    assert "data:image/png;base64,testbase64" in path.read_text(encoding="utf-8")


def test_generate_report_creates_output_dir_if_missing(tmp_path):
    nested = tmp_path / "sub" / "dir"
    with patch(_SUBPLOTS, return_value=_mock_subplots()), \
         patch(_SNS_HEATMAP), \
         patch.dict("sys.modules", {"wordcloud": MagicMock()}), \
         patch(_FIG_TO_BASE64, return_value="testbase64"):
        Visualizer().generate_report(_make_results(), nested)
    assert nested.exists()


# --- generate_comparison_report ---

def test_generate_comparison_report_returns_path(tmp_path):
    mock_az = MagicMock()
    mock_az._results = _make_results()
    with patch(_SUBPLOTS, return_value=_mock_subplots()), \
         patch(_FIG_TO_BASE64, return_value="testbase64"):
        result = Visualizer().generate_comparison_report([mock_az], tmp_path)
    assert isinstance(result, Path)


def test_generate_comparison_report_filename(tmp_path):
    mock_az = MagicMock()
    mock_az._results = _make_results()
    with patch(_SUBPLOTS, return_value=_mock_subplots()), \
         patch(_FIG_TO_BASE64, return_value="testbase64"):
        path = Visualizer().generate_comparison_report([mock_az], tmp_path)
    assert path.name == "comparison_report.html"


def test_generate_comparison_report_handles_empty_analyzer_list(tmp_path):
    with patch(_SUBPLOTS, return_value=_mock_subplots()), \
         patch(_FIG_TO_BASE64, return_value="testbase64"):
        path = Visualizer().generate_comparison_report([], tmp_path)
    assert path.exists()
