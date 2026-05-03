"""
Tests for sentiment_analyzer.py.

vaderSentiment and transformers are mocked throughout so this file
runs in complete isolation — no model downloads or heavy imports needed.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from whatsapp_analyzer.sentiment_analyzer import SentimentAnalyzer


def _make_df(messages: list[str], authors: list[str] | None = None) -> pd.DataFrame:
    """Return a minimal cleaned DataFrame that mimics Cleaner output."""
    n = len(messages)
    if authors is None:
        authors = ["Aminata"] * n
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-12 08:00"] * n),
            "author": authors,
            "message": messages,
            "msg_type": ["text"] * n,
            "group_name": ["TestGroup"] * n,
            "cleaned_message": messages,
            "language": ["fr"] * n,
            "tokens": [m.split() for m in messages],
        }
    )


_TEXTS = [
    "super excellent fantastique",   
    "terrible horrible mauvais",     
    "bonjour au revoir",            
    "génial bravo merveilleux",      
    "nul catastrophe désastre",      
    "cours demain annulé",          
]
_AUTHORS = ["Aminata", "Moussa", "Aminata", "Moussa", "Aminata", "Moussa"]


"Patch target for VADER inside the module"
_VADER_CLS = "vaderSentiment.vaderSentiment.SentimentIntensityAnalyzer"


def _make_vader_mock(scores: list[float] | None = None):
    """Return a mock VADER analyser that yields preset compound scores."""
    if scores is None:
        scores = [0.8, -0.8, 0.0, 0.6, -0.7, 0.02]
    mock_vader = MagicMock()
    mock_vader.polarity_scores.side_effect = [
        {"compound": s} for s in scores
    ]
    return mock_vader



def test_analyze_returns_dict():
    df = _make_df(_TEXTS, _AUTHORS)
    with patch(_VADER_CLS, return_value=_make_vader_mock()):
        result = SentimentAnalyzer(lang="en").analyze(df)
    assert isinstance(result, dict)


def test_analyze_has_df_key():
    df = _make_df(_TEXTS, _AUTHORS)
    with patch(_VADER_CLS, return_value=_make_vader_mock()):
        result = SentimentAnalyzer(lang="en").analyze(df)
    assert "df" in result


def test_analyze_has_by_user_key():
    df = _make_df(_TEXTS, _AUTHORS)
    with patch(_VADER_CLS, return_value=_make_vader_mock()):
        result = SentimentAnalyzer(lang="en").analyze(df)
    assert "by_user" in result


def test_analyze_has_global_key():
    df = _make_df(_TEXTS, _AUTHORS)
    with patch(_VADER_CLS, return_value=_make_vader_mock()):
        result = SentimentAnalyzer(lang="en").analyze(df)
    assert "global" in result



def test_output_df_has_sentiment_score_column():
    df = _make_df(_TEXTS, _AUTHORS)
    with patch(_VADER_CLS, return_value=_make_vader_mock()):
        result = SentimentAnalyzer(lang="en").analyze(df)
    assert "sentiment_score" in result["df"].columns


def test_output_df_has_sentiment_label_column():
    df = _make_df(_TEXTS, _AUTHORS)
    with patch(_VADER_CLS, return_value=_make_vader_mock()):
        result = SentimentAnalyzer(lang="en").analyze(df)
    assert "sentiment_label" in result["df"].columns


def test_sentiment_score_is_float():
    df = _make_df(_TEXTS, _AUTHORS)
    with patch(_VADER_CLS, return_value=_make_vader_mock()):
        result = SentimentAnalyzer(lang="en").analyze(df)
    assert result["df"]["sentiment_score"].dtype == float


def test_sentiment_score_in_valid_range():
    df = _make_df(_TEXTS, _AUTHORS)
    scores = [-1.0, 0.0, 1.0, -0.5, 0.5, 0.0]
    with patch(_VADER_CLS, return_value=_make_vader_mock(scores)):
        result = SentimentAnalyzer(lang="en").analyze(df)
    col = result["df"]["sentiment_score"]
    assert col.between(-1.0, 1.0).all()


def test_sentiment_label_values_are_valid():
    df = _make_df(_TEXTS, _AUTHORS)
    with patch(_VADER_CLS, return_value=_make_vader_mock()):
        result = SentimentAnalyzer(lang="en").analyze(df)
    assert set(result["df"]["sentiment_label"]).issubset(
        {"positive", "neutral", "negative"}
    )


def test_output_df_preserves_row_count():
    df = _make_df(_TEXTS, _AUTHORS)
    with patch(_VADER_CLS, return_value=_make_vader_mock()):
        result = SentimentAnalyzer(lang="en").analyze(df)
    assert len(result["df"]) == len(df)


def test_input_dataframe_not_mutated():
    df = _make_df(_TEXTS, _AUTHORS)
    original_cols = set(df.columns)
    with patch(_VADER_CLS, return_value=_make_vader_mock()):
        SentimentAnalyzer(lang="en").analyze(df)
    assert set(df.columns) == original_cols



def test_positive_label_above_threshold():
    df = _make_df(["good"], ["Aminata"])
    scores = [0.06]
    with patch(_VADER_CLS, return_value=_make_vader_mock(scores)):
        result = SentimentAnalyzer(lang="en").analyze(df)
    assert result["df"]["sentiment_label"].iloc[0] == "positive"


def test_negative_label_below_threshold():
    df = _make_df(["bad"], ["Aminata"])
    scores = [-0.06]
    with patch(_VADER_CLS, return_value=_make_vader_mock(scores)):
        result = SentimentAnalyzer(lang="en").analyze(df)
    assert result["df"]["sentiment_label"].iloc[0] == "negative"


def test_neutral_label_within_thresholds():
    df = _make_df(["ok"], ["Aminata"])
    scores = [0.02]
    with patch(_VADER_CLS, return_value=_make_vader_mock(scores)):
        result = SentimentAnalyzer(lang="en").analyze(df)
    assert result["df"]["sentiment_label"].iloc[0] == "neutral"


def test_boundary_positive_threshold_is_inclusive():
    df = _make_df(["ok"], ["Aminata"])
    scores = [0.06]
    with patch(_VADER_CLS, return_value=_make_vader_mock(scores)):
        result = SentimentAnalyzer(lang="en").analyze(df)
    assert result["df"]["sentiment_label"].iloc[0] == "positive"


def test_boundary_negative_threshold_is_inclusive():
    df = _make_df(["bad"], ["Aminata"])
    scores = [-0.06]
    with patch(_VADER_CLS, return_value=_make_vader_mock(scores)):
        result = SentimentAnalyzer(lang="en").analyze(df)
    assert result["df"]["sentiment_label"].iloc[0] == "negative"


def test_by_user_is_dataframe():
    df = _make_df(_TEXTS, _AUTHORS)
    with patch(_VADER_CLS, return_value=_make_vader_mock()):
        result = SentimentAnalyzer(lang="en").analyze(df)
    assert isinstance(result["by_user"], pd.DataFrame)


def test_by_user_has_author_column():
    df = _make_df(_TEXTS, _AUTHORS)
    with patch(_VADER_CLS, return_value=_make_vader_mock()):
        result = SentimentAnalyzer(lang="en").analyze(df)
    assert "author" in result["by_user"].columns


def test_by_user_has_mean_score_column():
    df = _make_df(_TEXTS, _AUTHORS)
    with patch(_VADER_CLS, return_value=_make_vader_mock()):
        result = SentimentAnalyzer(lang="en").analyze(df)
    assert "mean_score" in result["by_user"].columns


def test_by_user_row_count_matches_unique_authors():
    df = _make_df(_TEXTS, _AUTHORS)
    with patch(_VADER_CLS, return_value=_make_vader_mock()):
        result = SentimentAnalyzer(lang="en").analyze(df)
    n_authors = df["author"].nunique()
    assert len(result["by_user"]) == n_authors


def test_global_has_mean_key():
    df = _make_df(_TEXTS, _AUTHORS)
    with patch(_VADER_CLS, return_value=_make_vader_mock()):
        result = SentimentAnalyzer(lang="en").analyze(df)
    assert "mean" in result["global"]


def test_global_has_pos_pct_key():
    df = _make_df(_TEXTS, _AUTHORS)
    with patch(_VADER_CLS, return_value=_make_vader_mock()):
        result = SentimentAnalyzer(lang="en").analyze(df)
    assert "pos_pct" in result["global"]


def test_global_has_neg_pct_key():
    df = _make_df(_TEXTS, _AUTHORS)
    with patch(_VADER_CLS, return_value=_make_vader_mock()):
        result = SentimentAnalyzer(lang="en").analyze(df)
    assert "neg_pct" in result["global"]


def test_global_mean_is_float():
    df = _make_df(_TEXTS, _AUTHORS)
    with patch(_VADER_CLS, return_value=_make_vader_mock()):
        result = SentimentAnalyzer(lang="en").analyze(df)
    assert isinstance(result["global"]["mean"], float)


def test_global_pos_pct_is_float():
    df = _make_df(_TEXTS, _AUTHORS)
    with patch(_VADER_CLS, return_value=_make_vader_mock()):
        result = SentimentAnalyzer(lang="en").analyze(df)
    assert isinstance(result["global"]["pos_pct"], float)


def test_global_neg_pct_is_float():
    df = _make_df(_TEXTS, _AUTHORS)
    with patch(_VADER_CLS, return_value=_make_vader_mock()):
        result = SentimentAnalyzer(lang="en").analyze(df)
    assert isinstance(result["global"]["neg_pct"], float)


def test_global_pct_values_are_between_0_and_1():
    df = _make_df(_TEXTS, _AUTHORS)
    with patch(_VADER_CLS, return_value=_make_vader_mock()):
        result = SentimentAnalyzer(lang="en").analyze(df)
    g = result["global"]
    assert 0.0 <= g["pos_pct"] <= 1.0
    assert 0.0 <= g["neg_pct"] <= 1.0


def test_global_mean_correctness():
    df = _make_df(_TEXTS, _AUTHORS)
    scores = [0.8, -0.8, 0.0, 0.6, -0.7, 0.02]
    expected_mean = sum(scores) / len(scores)
    with patch(_VADER_CLS, return_value=_make_vader_mock(scores)):
        result = SentimentAnalyzer(lang="en").analyze(df)
    assert abs(result["global"]["mean"] - expected_mean) < 1e-6


def test_global_pos_pct_correctness():
    df = _make_df(_TEXTS, _AUTHORS)
    # scores > 0.05: 0.8, 0.6 => 2 out of 6
    scores = [0.8, -0.8, 0.0, 0.6, -0.7, 0.02]
    with patch(_VADER_CLS, return_value=_make_vader_mock(scores)):
        result = SentimentAnalyzer(lang="en").analyze(df)
    assert abs(result["global"]["pos_pct"] - 2 / 6) < 1e-6


def test_global_neg_pct_correctness():
    df = _make_df(_TEXTS, _AUTHORS)
    # scores < -0.05: -0.8, -0.7 => 2 out of 6
    scores = [0.8, -0.8, 0.0, 0.6, -0.7, 0.02]
    with patch(_VADER_CLS, return_value=_make_vader_mock(scores)):
        result = SentimentAnalyzer(lang="en").analyze(df)
    assert abs(result["global"]["neg_pct"] - 2 / 6) < 1e-6


def test_vader_polarity_scores_called_for_each_message():
    df = _make_df(_TEXTS, _AUTHORS)
    mock_vader = _make_vader_mock()
    with patch(_VADER_CLS, return_value=mock_vader):
        SentimentAnalyzer(lang="en").analyze(df)
    assert mock_vader.polarity_scores.call_count == len(_TEXTS)


def test_vader_compound_key_is_used():
    df = _make_df(["test message"], ["Aminata"])
    mock_vader = MagicMock()
    mock_vader.polarity_scores.return_value = {
        "compound": 0.5, "pos": 0.3, "neg": 0.0, "neu": 0.7
    }
    with patch(_VADER_CLS, return_value=mock_vader):
        result = SentimentAnalyzer(lang="en").analyze(df)
    assert result["df"]["sentiment_score"].iloc[0] == pytest.approx(0.5)


def test_empty_dataframe_raises_value_error():
    df = pd.DataFrame(columns=["cleaned_message", "author"])
    with pytest.raises(ValueError, match="empty"):
        SentimentAnalyzer().analyze(df)


def test_missing_cleaned_message_column_raises():
    df = pd.DataFrame({"author": ["Aminata"], "message": ["test"]})
    with pytest.raises(ValueError, match="cleaned_message"):
        SentimentAnalyzer().analyze(df)


def test_missing_author_column_raises():
    df = pd.DataFrame({"cleaned_message": ["test message"], "message": ["test"]})
    with pytest.raises(ValueError, match="author"):
        SentimentAnalyzer().analyze(df)


"""
CamemBERT path (issue #06)
transformers is never imported for real — the whole module is mocked.
"""

def _make_camembert_pipeline_mock(
    labels_scores: list[tuple[str, float]] | None = None
):
    """Return a mock HuggingFace pipeline that returns preset results."""
    if labels_scores is None:
        labels_scores = [
            ("LABEL_1", 0.9),
            ("LABEL_0", 0.85),
            ("LABEL_1", 0.55),
            ("LABEL_1", 0.75),
            ("LABEL_0", 0.7),
            ("LABEL_1", 0.52),
        ]
    pipeline_mock = MagicMock()
    pipeline_mock.side_effect = [
        [{"label": label, "score": score}]
        for label, score in labels_scores
    ]
    return pipeline_mock


_TRANSFORMERS_MOCK_PATH = "whatsapp_analyzer.sentiment_analyzer.hf_pipeline"


def _make_transformers_module_mock(pipeline_instance):
    """Build a mock transformers module with a `pipeline` attribute."""
    mod = MagicMock()
    mod.pipeline = MagicMock(return_value=pipeline_instance)
    return mod


""" --- return type and structure --- """

def test_camembert_analyze_returns_dict():
    df = _make_df(_TEXTS, _AUTHORS)
    pipe = _make_camembert_pipeline_mock()
    with patch.dict("sys.modules", {"transformers": _make_transformers_module_mock(pipe)}):
        result = SentimentAnalyzer(lang="fr").analyze(df)
    assert isinstance(result, dict)


def test_camembert_output_has_df_key():
    df = _make_df(_TEXTS, _AUTHORS)
    pipe = _make_camembert_pipeline_mock()
    with patch.dict("sys.modules", {"transformers": _make_transformers_module_mock(pipe)}):
        result = SentimentAnalyzer(lang="fr").analyze(df)
    assert "df" in result


def test_camembert_df_has_sentiment_score_column():
    df = _make_df(_TEXTS, _AUTHORS)
    pipe = _make_camembert_pipeline_mock()
    with patch.dict("sys.modules", {"transformers": _make_transformers_module_mock(pipe)}):
        result = SentimentAnalyzer(lang="fr").analyze(df)
    assert "sentiment_score" in result["df"].columns


def test_camembert_df_has_sentiment_label_column():
    df = _make_df(_TEXTS, _AUTHORS)
    pipe = _make_camembert_pipeline_mock()
    with patch.dict("sys.modules", {"transformers": _make_transformers_module_mock(pipe)}):
        result = SentimentAnalyzer(lang="fr").analyze(df)
    assert "sentiment_label" in result["df"].columns


def test_camembert_preserves_row_count():
    df = _make_df(_TEXTS, _AUTHORS)
    pipe = _make_camembert_pipeline_mock()
    with patch.dict("sys.modules", {"transformers": _make_transformers_module_mock(pipe)}):
        result = SentimentAnalyzer(lang="fr").analyze(df)
    assert len(result["df"]) == len(df)


""" --- positive label maps to +score --- """

def test_camembert_label1_gives_positive_score():
    df = _make_df(["super"], ["Aminata"])
    pipe = _make_camembert_pipeline_mock([("LABEL_1", 0.9)])
    with patch.dict("sys.modules", {"transformers": _make_transformers_module_mock(pipe)}):
        result = SentimentAnalyzer(lang="fr").analyze(df)
    assert result["df"]["sentiment_score"].iloc[0] == pytest.approx(0.9)


def test_camembert_label0_gives_negative_score():
    df = _make_df(["terrible"], ["Aminata"])
    pipe = _make_camembert_pipeline_mock([("LABEL_0", 0.85)])
    with patch.dict("sys.modules", {"transformers": _make_transformers_module_mock(pipe)}):
        result = SentimentAnalyzer(lang="fr").analyze(df)
    assert result["df"]["sentiment_score"].iloc[0] == pytest.approx(-0.85)


def test_camembert_label_values_are_valid():
    df = _make_df(_TEXTS, _AUTHORS)
    pipe = _make_camembert_pipeline_mock()
    with patch.dict("sys.modules", {"transformers": _make_transformers_module_mock(pipe)}):
        result = SentimentAnalyzer(lang="fr").analyze(df)
    assert set(result["df"]["sentiment_label"]).issubset(
        {"positive", "neutral", "negative"}
    )


""" --- fallback to VADER when transformers is absent --- """

def test_camembert_falls_back_to_vader_when_not_installed():
    df = _make_df(_TEXTS, _AUTHORS)
    mock_vader = _make_vader_mock()
    # Remove transformers from sys.modules entirely
    with patch.dict("sys.modules", {"transformers": None}):
        with patch(_VADER_CLS, return_value=mock_vader):
            result = SentimentAnalyzer(lang="fr").analyze(df)
    assert "df" in result
    assert mock_vader.polarity_scores.call_count == len(_TEXTS)


""" --- non-French language routes to VADER even when transformers is present --- """

def test_english_lang_uses_vader_not_camembert():
    df = _make_df(_TEXTS, _AUTHORS)
    mock_vader = _make_vader_mock()
    pipe = _make_camembert_pipeline_mock()
    with patch.dict("sys.modules", {"transformers": _make_transformers_module_mock(pipe)}):
        with patch(_VADER_CLS, return_value=mock_vader):
            SentimentAnalyzer(lang="en").analyze(df)
    assert mock_vader.polarity_scores.call_count == len(_TEXTS)
    assert pipe.call_count == 0


 """--- VADER path unaffected after CamemBERT addition ---"""

def test_vader_path_unaffected():
    df = _make_df(_TEXTS, _AUTHORS)
    mock_vader = _make_vader_mock()
    """ Ensure transformers is absent so VADER is always chosen """
    with patch.dict("sys.modules", {"transformers": None}):
        with patch(_VADER_CLS, return_value=mock_vader):
            result = SentimentAnalyzer(lang="en").analyze(df)
    assert "sentiment_score" in result["df"].columns
    assert "sentiment_label" in result["df"].columns
    assert "by_user" in result
    assert "global" in result
