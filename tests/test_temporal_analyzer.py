"""
Tests for temporal_analyzer.py.

No mocking needed — pure pandas, no external dependencies.
All input DataFrames are built locally.
"""

import pandas as pd
import pytest

from whatsapp_analyzer.temporal_analyzer import TemporalAnalyzer

# 2024-01-12 = Friday, 2024-01-13 = Saturday, 2024-01-15 = Monday
_TIMESTAMPS_BASE = [
    "2024-01-12 08:00",  # Friday   08h
    "2024-01-12 08:30",  # Friday   08h
    "2024-01-12 08:45",  # Friday   08h  ← peak hour (3 msgs at 08h)
    "2024-01-12 10:00",  # Friday   10h
    "2024-01-13 14:00",  # Saturday 14h
    "2024-01-13 15:00",  # Saturday 15h
]


def _make_df(timestamps: list[str]) -> pd.DataFrame:
    n = len(timestamps)
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps),
            "author": ["Aminata"] * n,
            "message": ["test message"] * n,
            "msg_type": ["text"] * n,
            "group_name": ["TestGroup"] * n,
            "cleaned_message": ["test"] * n,
        }
    )


@pytest.fixture
def base_df():
    return _make_df(_TIMESTAMPS_BASE)


@pytest.fixture
def result(base_df):
    return TemporalAnalyzer().analyze(base_df)


# --- return type and keys ---

def test_analyze_returns_dict(result):
    assert isinstance(result, dict)


def test_analyze_has_all_required_keys(result):
    expected = {
        "timeline", "hourly_heatmap", "weekly_activity",
        "monthly_activity", "peak_hour", "peak_day",
    }
    assert set(result.keys()) == expected


# --- timeline ---

def test_timeline_is_dataframe(result):
    assert isinstance(result["timeline"], pd.DataFrame)


def test_timeline_has_count_column(result):
    assert "count" in result["timeline"].columns


def test_timeline_one_row_per_unique_day(result):
    # base_df spans 2 days (Friday + Saturday)
    assert len(result["timeline"]) == 2


def test_timeline_index_is_datetime(result):
    assert pd.api.types.is_datetime64_any_dtype(result["timeline"].index)


def test_timeline_counts_are_positive(result):
    assert (result["timeline"]["count"] > 0).all()


def test_timeline_total_equals_message_count(base_df, result):
    assert result["timeline"]["count"].sum() == len(base_df)


# --- hourly_heatmap ---

def test_hourly_heatmap_is_dataframe(result):
    assert isinstance(result["hourly_heatmap"], pd.DataFrame)


def test_hourly_heatmap_has_seven_rows(result):
    assert len(result["hourly_heatmap"]) == 7


def test_hourly_heatmap_has_24_columns(result):
    assert len(result["hourly_heatmap"].columns) == 24


def test_hourly_heatmap_columns_are_0_to_23(result):
    assert list(result["hourly_heatmap"].columns) == list(range(24))


def test_hourly_heatmap_index_is_weekday_names(result):
    expected = ["Monday", "Tuesday", "Wednesday", "Thursday",
                "Friday", "Saturday", "Sunday"]
    assert list(result["hourly_heatmap"].index) == expected


def test_hourly_heatmap_values_are_nonnegative(result):
    assert (result["hourly_heatmap"] >= 0).all().all()


def test_hourly_heatmap_friday_08h_is_three(result):
    # 3 messages on Friday at 08h
    assert result["hourly_heatmap"].loc["Friday", 8] == 3


def test_hourly_heatmap_total_equals_message_count(base_df, result):
    assert result["hourly_heatmap"].values.sum() == len(base_df)


# --- weekly_activity ---

def test_weekly_activity_is_series(result):
    assert isinstance(result["weekly_activity"], pd.Series)


def test_weekly_activity_index_is_monday_to_sunday(result):
    expected = ["Monday", "Tuesday", "Wednesday", "Thursday",
                "Friday", "Saturday", "Sunday"]
    assert list(result["weekly_activity"].index) == expected


def test_weekly_activity_friday_count_is_four(result):
    assert result["weekly_activity"]["Friday"] == 4


def test_weekly_activity_saturday_count_is_two(result):
    assert result["weekly_activity"]["Saturday"] == 2


def test_weekly_activity_unused_days_are_zero(result):
    for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Sunday"]:
        assert result["weekly_activity"][day] == 0


# --- monthly_activity ---

def test_monthly_activity_is_series(result):
    assert isinstance(result["monthly_activity"], pd.Series)


def test_monthly_activity_one_month_when_all_same_month(result):
    # all messages in January 2024
    assert len(result["monthly_activity"]) == 1


def test_monthly_activity_two_months_when_spanning(base_df):
    extra = _make_df(["2024-02-05 09:00"])
    df = pd.concat([base_df, extra], ignore_index=True)
    result = TemporalAnalyzer().analyze(df)
    assert len(result["monthly_activity"]) == 2


def test_monthly_activity_total_equals_message_count(base_df, result):
    assert result["monthly_activity"].sum() == len(base_df)


# --- peak_hour ---

def test_peak_hour_is_integer(result):
    assert isinstance(result["peak_hour"], int)


def test_peak_hour_is_in_valid_range(result):
    assert 0 <= result["peak_hour"] <= 23


def test_peak_hour_is_correct(result):
    # 3 messages at 08h, 1 at 10h, 1 at 14h, 1 at 15h
    assert result["peak_hour"] == 8


# --- peak_day ---

def test_peak_day_is_string(result):
    assert isinstance(result["peak_day"], str)


def test_peak_day_is_correct(result):
    # Friday has 4 messages, Saturday has 2
    assert result["peak_day"] == "Friday"


# --- error cases ---

def test_empty_dataframe_raises_value_error():
    df = pd.DataFrame(columns=["timestamp", "author", "message"])
    with pytest.raises(ValueError, match="empty"):
        TemporalAnalyzer().analyze(df)


def test_missing_timestamp_column_raises_value_error():
    df = pd.DataFrame({"author": ["Aminata"], "message": ["hello"]})
    with pytest.raises(ValueError, match="timestamp"):
        TemporalAnalyzer().analyze(df)
