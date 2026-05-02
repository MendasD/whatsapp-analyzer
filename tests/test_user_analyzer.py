"""
Tests for user_analyzer.py.

All input data is built locally from scratch — no pipeline modules are
imported or executed. Tests run in complete isolation.
"""

import pandas as pd
import pytest

from whatsapp_analyzer.user_analyzer import UserAnalyzer


# 2024-01-12 = Friday, 2024-01-13 = Saturday, 2024-01-14 = Sunday
_TIMESTAMPS = [
    "2024-01-12 08:00",  # Aminata  — Friday 08h
    "2024-01-12 08:30",  # Moussa   — Friday 08h
    "2024-01-12 10:00",  # Aminata  — Friday 10h
    "2024-01-13 14:00",  # Moussa   — Saturday 14h
    "2024-01-13 09:00",  # Aminata  — Saturday 09h
    "2024-01-13 16:00",  # Moussa   — Saturday 16h
    "2024-01-14 20:00",  # Moussa   — Sunday 20h
]
_AUTHORS = ["Aminata", "Moussa", "Aminata", "Moussa", "Aminata", "Moussa", "Moussa"]


def _make_df_clean() -> pd.DataFrame:
    messages = [
        "sport match jouer gagner",
        "cours td examen prof",
        "sport match jouer equipe",
        "cours td examen notes",
        "cours td prof notes",
        "sport match equipe gagner",
        "famille maison ami sortir",
    ]
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(_TIMESTAMPS),
            "author": _AUTHORS,
            "message": messages,
            "msg_type": ["text"] * 7,
            "group_name": ["TestGroup"] * 7,
            "cleaned_message": messages,
            "language": ["fr"] * 7,
            "tokens": [m.split() for m in messages],
        }
    )


def _make_topics_result(df_clean: pd.DataFrame) -> dict:
    topics_df = df_clean.copy()
    # Aminata rows 0,2,4 → topics 0,0,1  |  Moussa rows 1,3,5,6 → topics 1,1,0,2
    topics_df["topic_id"] = [0, 1, 0, 1, 1, 0, 2]
    topics_df["topic_score"] = [0.8, 0.7, 0.85, 0.9, 0.6, 0.75, 0.65]

    group_topics = pd.DataFrame(
        {
            "topic_id": [0, 1, 2],
            "topic_label": [
                "sport / match / jouer / gagner / equipe",
                "cours / td / examen / prof / notes",
                "famille / maison / ami / sortir / manger",
            ],
            "weight": [0.4, 0.4, 0.2],
        }
    )
    return {"df": topics_df, "group_topics": group_topics}


def _make_sentiment_result(df_clean: pd.DataFrame) -> dict:
    sentiment_df = df_clean.copy()
    sentiment_df["sentiment_score"] = [0.3, -0.1, 0.5, 0.2, -0.2, 0.1, 0.4]
    sentiment_df["sentiment_label"] = [
        "positive", "neutral", "positive", "positive",
        "negative", "positive", "positive",
    ]
    by_user = (
        sentiment_df.groupby("author")["sentiment_score"].mean().reset_index()
    )
    return {
        "df": sentiment_df,
        "by_user": by_user,
        "global": {"mean": 0.17, "pos_pct": 0.71, "neg_pct": 0.14},
    }


def _make_results(with_sentiment: bool = True) -> dict:
    df_clean = _make_df_clean()
    results: dict = {
        "df_clean": df_clean,
        "topics": _make_topics_result(df_clean),
    }
    results["sentiment"] = _make_sentiment_result(df_clean) if with_sentiment else None
    return results


# --- build_profiles: return type and structure ---

def test_build_profiles_returns_dict():
    result = UserAnalyzer().build_profiles(_make_results())
    assert isinstance(result, dict)


def test_build_profiles_has_entry_per_author():
    result = UserAnalyzer().build_profiles(_make_results())
    assert set(result.keys()) == {"Aminata", "Moussa"}


def test_profile_has_required_keys():
    profile = UserAnalyzer().build_profiles(_make_results())["Aminata"]
    expected = {
        "message_count", "avg_message_length", "activity_hours",
        "most_active_day", "top_topics", "sentiment_mean",
    }
    assert set(profile.keys()) == expected


# --- message_count ---

def test_message_count_aminata_is_three():
    profile = UserAnalyzer().build_profiles(_make_results())["Aminata"]
    assert profile["message_count"] == 3


def test_message_count_moussa_is_four():
    profile = UserAnalyzer().build_profiles(_make_results())["Moussa"]
    assert profile["message_count"] == 4


# --- avg_message_length ---

def test_avg_message_length_is_float():
    profile = UserAnalyzer().build_profiles(_make_results())["Aminata"]
    assert isinstance(profile["avg_message_length"], float)


def test_avg_message_length_is_positive():
    profile = UserAnalyzer().build_profiles(_make_results())["Aminata"]
    assert profile["avg_message_length"] > 0


# --- activity_hours ---

def test_activity_hours_is_dict():
    profile = UserAnalyzer().build_profiles(_make_results())["Aminata"]
    assert isinstance(profile["activity_hours"], dict)


def test_activity_hours_keys_are_integers():
    profile = UserAnalyzer().build_profiles(_make_results())["Aminata"]
    assert all(isinstance(h, (int,)) for h in profile["activity_hours"])


def test_activity_hours_values_are_integers():
    profile = UserAnalyzer().build_profiles(_make_results())["Aminata"]
    assert all(isinstance(v, (int,)) for v in profile["activity_hours"].values())


# --- most_active_day ---

def test_most_active_day_is_string():
    profile = UserAnalyzer().build_profiles(_make_results())["Aminata"]
    assert isinstance(profile["most_active_day"], str)


def test_most_active_day_aminata_is_friday():
    # Aminata has 2 messages on Friday and 1 on Saturday
    profile = UserAnalyzer().build_profiles(_make_results())["Aminata"]
    assert profile["most_active_day"] == "Friday"


def test_most_active_day_moussa_is_saturday():
    # Moussa has 2 messages on Saturday and 1 each on Friday and Sunday
    profile = UserAnalyzer().build_profiles(_make_results())["Moussa"]
    assert profile["most_active_day"] == "Saturday"


# --- top_topics ---

def test_top_topics_is_list():
    profile = UserAnalyzer().build_profiles(_make_results())["Aminata"]
    assert isinstance(profile["top_topics"], list)


def test_top_topics_contains_tuples_of_two():
    profile = UserAnalyzer().build_profiles(_make_results())["Aminata"]
    for item in profile["top_topics"]:
        assert isinstance(item, tuple)
        assert len(item) == 2


def test_top_topics_label_is_string():
    profile = UserAnalyzer().build_profiles(_make_results())["Aminata"]
    for label, _ in profile["top_topics"]:
        assert isinstance(label, str)


def test_top_topics_score_is_float():
    profile = UserAnalyzer().build_profiles(_make_results())["Aminata"]
    for _, score in profile["top_topics"]:
        assert isinstance(score, float)


def test_top_topics_empty_when_topics_not_run():
    results = _make_results()
    results["topics"] = None
    profile = UserAnalyzer().build_profiles(results)["Aminata"]
    assert profile["top_topics"] == []


# --- sentiment_mean ---

def test_sentiment_mean_is_float_when_available():
    profile = UserAnalyzer().build_profiles(_make_results())["Aminata"]
    assert isinstance(profile["sentiment_mean"], float)


def test_sentiment_mean_is_none_when_not_run():
    profile = UserAnalyzer().build_profiles(_make_results(with_sentiment=False))["Aminata"]
    assert profile["sentiment_mean"] is None


# --- summary_for ---

def test_summary_for_returns_dict_for_known_author():
    result = UserAnalyzer.summary_for("Aminata", _make_results())
    assert isinstance(result, dict)
    assert result["message_count"] == 3


def test_summary_for_returns_empty_for_unknown_author():
    result = UserAnalyzer.summary_for("Nobody", _make_results())
    assert result == {}


# --- topics_for ---

def test_topics_for_returns_dataframe():
    result = UserAnalyzer.topics_for("Aminata", _make_results())
    assert isinstance(result, pd.DataFrame)


def test_topics_for_has_expected_columns():
    result = UserAnalyzer.topics_for("Aminata", _make_results())
    assert "topic_id" in result.columns
    assert "topic_label" in result.columns
    assert "topic_score" in result.columns


def test_topics_for_returns_empty_when_topics_not_run():
    results = _make_results()
    results["topics"] = None
    result = UserAnalyzer.topics_for("Aminata", results)
    assert result.empty


def test_topics_for_row_count_matches_author_messages():
    # Aminata has 3 messages in the topics df
    result = UserAnalyzer.topics_for("Aminata", _make_results())
    assert len(result) == 3


# --- sentiment_over_time_for ---

def test_sentiment_over_time_for_returns_dataframe():
    result = UserAnalyzer.sentiment_over_time_for("Aminata", _make_results())
    assert isinstance(result, pd.DataFrame)


def test_sentiment_over_time_for_has_expected_columns():
    result = UserAnalyzer.sentiment_over_time_for("Aminata", _make_results())
    assert "timestamp" in result.columns
    assert "sentiment_score" in result.columns
    assert "sentiment_label" in result.columns


def test_sentiment_over_time_for_returns_empty_when_not_run():
    results = _make_results(with_sentiment=False)
    result = UserAnalyzer.sentiment_over_time_for("Aminata", results)
    assert result.empty


def test_sentiment_over_time_for_row_count_matches_author():
    result = UserAnalyzer.sentiment_over_time_for("Aminata", _make_results())
    assert len(result) == 3


# --- activity_heatmap_for ---

def test_activity_heatmap_for_returns_dataframe():
    result = UserAnalyzer.activity_heatmap_for("Aminata", _make_results())
    assert isinstance(result, pd.DataFrame)


def test_activity_heatmap_has_seven_rows():
    result = UserAnalyzer.activity_heatmap_for("Aminata", _make_results())
    assert len(result) == 7


def test_activity_heatmap_has_24_columns():
    result = UserAnalyzer.activity_heatmap_for("Aminata", _make_results())
    assert len(result.columns) == 24


def test_activity_heatmap_for_unknown_author_returns_zeros():
    result = UserAnalyzer.activity_heatmap_for("Nobody", _make_results())
    assert (result == 0).all().all()


def test_activity_heatmap_friday_hour8_is_nonzero_for_aminata():
    # Aminata sent a message on Friday at 08:00
    result = UserAnalyzer.activity_heatmap_for("Aminata", _make_results())
    assert result.loc["Friday", 8] > 0
