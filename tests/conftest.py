"""
Shared pytest fixtures for whatsapp-analyzer tests.
All fixtures use fake data only — no real chat exports, no real names.
"""

import pandas as pd
import pytest
from unittest.mock import MagicMock


# Fake author names used consistently across all fixtures
_AUTHORS = ["Aminata", "Moussa", "Fatou", "Ibrahima", "Mariama"]


# Helper — build a minimal clean DataFrame

def _make_df(n: int = 10) -> pd.DataFrame:
    """Return a fake cleaned DataFrame with n rows."""
    timestamps = pd.date_range("2024-01-01 08:00", periods=n, freq="2h")
    authors = [_AUTHORS[i % len(_AUTHORS)] for i in range(n)]
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "author": authors,
            "message": [f"message number {i}" for i in range(n)],
            "msg_type": ["text"] * n,
            "group_name": ["TestGroup"] * n,
            "cleaned_message": [f"clean message {i}" for i in range(n)],
            "language": ["fr"] * n,
            "tokens": [[f"token{i}a", f"token{i}b"] for i in range(n)],
            "topic_id": [i % 3 for i in range(n)],
            "topic_label": [f"Topic_{i % 3}" for i in range(n)],
            "topic_score": [0.7 + (i % 3) * 0.05 for i in range(n)],
            "sentiment_score": [0.1 * (i % 5 - 2) for i in range(n)],
            "sentiment_label": ["positive" if i % 2 == 0 else "negative" for i in range(n)],
        }
    )


# sample_results fixture

@pytest.fixture
def sample_results() -> dict:
    """
    Fake dict mimicking the output of resultscore.py / WhatsAppAnalyzer.

    Keys: df_raw, df_clean, topics, sentiment, temporal, users, group_name.
    """
    df_raw = _make_df(10)
    df_clean = df_raw.copy()

    topics = {
        "group_topics": pd.DataFrame(
            {
                "topic_id": [0, 1, 2],
                "topic_label": ["Topic_0", "Topic_1", "Topic_2"],
                "top_words": [
                    ["word1", "word2", "word3"],
                    ["word4", "word5", "word6"],
                    ["word7", "word8", "word9"],
                ],
                "message_count": [4, 3, 3],
            }
        )
    }

    sentiment = {
        "df": df_clean[["author", "sentiment_score", "sentiment_label"]].copy(),
        "by_user": {
            author: {
                "mean_score": 0.1 * i,
                "positive_ratio": 0.5 + 0.1 * i,
                "negative_ratio": 0.5 - 0.1 * i,
            }
            for i, author in enumerate(_AUTHORS)
        },
        "global": {
            "mean_score": 0.05,
            "positive_ratio": 0.6,
            "negative_ratio": 0.4,
        },
    }

    temporal = {
        "timeline": df_clean.set_index("timestamp").resample("D").size().reset_index(
            name="message_count"
        ),
        "hourly_heatmap": pd.DataFrame(
            {
                "hour": list(range(24)),
                "message_count": [i % 5 for i in range(24)],
            }
        ),
        "peak_hour": 8,
        "peak_day": "Monday",
    }

    users = {
        author: {
            "message_count": 2,
            "media_count": 0,
            "avg_message_length": 15.0,
            "active_days": 2,
            "top_topics": ["Topic_0"],
            "sentiment_mean": 0.1,
        }
        for author in _AUTHORS
    }

    return {
        "df_raw": df_raw,
        "df_clean": df_clean,
        "topics": topics,
        "sentiment": sentiment,
        "temporal": temporal,
        "users": users,
        "group_name": "TestGroup",
    }


# sample_topics fixture

@pytest.fixture
def sample_topics() -> pd.DataFrame:
    """
    Fake group_topics DataFrame with 3 topics.
    Mirrors the output of TopicClassifier.
    """
    return pd.DataFrame(
        {
            "topic_id": [0, 1, 2],
            "topic_label": ["Topic_0", "Topic_1", "Topic_2"],
            "top_words": [
                ["word1", "word2", "word3"],
                ["word4", "word5", "word6"],
                ["word7", "word8", "word9"],
            ],
            "message_count": [4, 3, 3],
        }
    )


# sample_sentiment fixture

@pytest.fixture
def sample_sentiment() -> dict:
    """
    Fake sentiment dict with keys: df, by_user, global.
    Mirrors the output of SentimentAnalyzer.
    """
    df = _make_df(10)[["author", "sentiment_score", "sentiment_label"]].copy()
    return {
        "df": df,
        "by_user": {
            author: {
                "mean_score": 0.1 * i,
                "positive_ratio": 0.5 + 0.05 * i,
                "negative_ratio": 0.5 - 0.05 * i,
            }
            for i, author in enumerate(_AUTHORS)
        },
        "global": {
            "mean_score": 0.05,
            "positive_ratio": 0.6,
            "negative_ratio": 0.4,
        },
    }


# sample_temporal fixture

@pytest.fixture
def sample_temporal() -> dict:
    """
    Fake temporal dict with keys: timeline, hourly_heatmap, peak_hour, peak_day.
    Mirrors the output of TemporalAnalyzer.
    """
    timeline = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="D"),
            "message_count": [3, 5, 2, 8, 4],
        }
    )
    hourly_heatmap = pd.DataFrame(
        {
            "hour": list(range(24)),
            "message_count": [i % 5 for i in range(24)],
        }
    )
    return {
        "timeline": timeline,
        "hourly_heatmap": hourly_heatmap,
        "peak_hour": 8,
        "peak_day": "Monday",
    }


# mock_analyzer fixture

@pytest.fixture
def mock_analyzer() -> MagicMock:
    """
    MagicMock that mimics a WhatsAppAnalyzer instance.

    Exposed attributes and methods:
        _results      — fake results dict (same shape as sample_results)
        _group_name   — str
        raw_data()    — returns a fake raw DataFrame
        users()       — returns a fake users dict
        topics()      — returns a fake group_topics DataFrame
    """
    df_raw = _make_df(10)
    df_clean = df_raw.copy()

    fake_results = {
        "df_raw": df_raw,
        "df_clean": df_clean,
        "topics": {
            "group_topics": pd.DataFrame(
                {
                    "topic_id": [0, 1, 2],
                    "topic_label": ["Topic_0", "Topic_1", "Topic_2"],
                    "top_words": [["w1", "w2"], ["w3", "w4"], ["w5", "w6"]],
                    "message_count": [4, 3, 3],
                }
            )
        },
        "sentiment": {
            "df": df_clean[["author", "sentiment_score", "sentiment_label"]].copy(),
            "by_user": {},
            "global": {"mean_score": 0.05, "positive_ratio": 0.6, "negative_ratio": 0.4},
        },
        "temporal": {
            "timeline": pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=3, freq="D"), "message_count": [3, 5, 2]}),
            "hourly_heatmap": pd.DataFrame({"hour": list(range(24)), "message_count": [i % 5 for i in range(24)]}),
            "peak_hour": 8,
            "peak_day": "Monday",
        },
        "users": {a: {"message_count": 2} for a in _AUTHORS},
        "group_name": "TestGroup",
    }

    analyzer = MagicMock()
    analyzer._results = fake_results
    analyzer._group_name = "TestGroup"
    analyzer.raw_data.return_value = df_raw
    analyzer.users.return_value = fake_results["users"]
    analyzer.topics.return_value = fake_results["topics"]["group_topics"]

    return analyzer