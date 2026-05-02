"""
Tests for topic_classifier.py.

sklearn (TfidfVectorizer, LatentDirichletAllocation) is mocked throughout so
this file runs in complete isolation — no model fitting or downloads needed.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from whatsapp_analyzer.topic_classifier import TopicClassifier


# Minimal DataFrame that mimics Cleaner output
def _make_df(messages: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-12 08:00"] * len(messages)),
            "author": ["Aminata"] * len(messages),
            "message": messages,
            "msg_type": ["text"] * len(messages),
            "group_name": ["TestGroup"] * len(messages),
            "cleaned_message": messages,
            "language": ["fr"] * len(messages),
            "tokens": [m.split() for m in messages],
        }
    )


# Patch targets for lazy sklearn imports
_TFIDF_CLS = "sklearn.feature_extraction.text.TfidfVectorizer"
_LDA_CLS = "sklearn.decomposition.LatentDirichletAllocation"


def _make_sklearn_mocks(n_topics: int = 3, n_messages: int = 6, n_words: int = 10):
    """Return (mock_tfidf_cls, mock_lda_cls) with realistic numpy return values."""
    # Each message is assigned to one topic with high probability
    topic_matrix = np.zeros((n_messages, n_topics))
    remainder = 0.2 / max(n_topics - 1, 1)
    for i in range(n_messages):
        dominant = i % n_topics
        topic_matrix[i, :] = remainder
        topic_matrix[i, dominant] = 0.8

    components = np.random.default_rng(42).random((n_topics, n_words))
    feature_names = np.array([f"word{i}" for i in range(n_words)])

    mock_tfidf_cls = MagicMock()
    mock_tfidf_cls.return_value.fit_transform.return_value = MagicMock()
    mock_tfidf_cls.return_value.get_feature_names_out.return_value = feature_names

    mock_lda_cls = MagicMock()
    mock_lda_cls.return_value.fit_transform.return_value = topic_matrix
    mock_lda_cls.return_value.components_ = components

    return mock_tfidf_cls, mock_lda_cls


# --- return type and structure ---

def test_fit_transform_returns_dict():
    df = _make_df(["word1 word2 word3"] * 6)
    mock_tfidf, mock_lda = _make_sklearn_mocks()
    with patch(_TFIDF_CLS, mock_tfidf), patch(_LDA_CLS, mock_lda):
        result = TopicClassifier(n_topics=3).fit_transform(df)
    assert isinstance(result, dict)


def test_fit_transform_has_df_key():
    df = _make_df(["word1 word2 word3"] * 6)
    mock_tfidf, mock_lda = _make_sklearn_mocks()
    with patch(_TFIDF_CLS, mock_tfidf), patch(_LDA_CLS, mock_lda):
        result = TopicClassifier(n_topics=3).fit_transform(df)
    assert "df" in result


def test_fit_transform_has_group_topics_key():
    df = _make_df(["word1 word2 word3"] * 6)
    mock_tfidf, mock_lda = _make_sklearn_mocks()
    with patch(_TFIDF_CLS, mock_tfidf), patch(_LDA_CLS, mock_lda):
        result = TopicClassifier(n_topics=3).fit_transform(df)
    assert "group_topics" in result


# --- output DataFrame columns ---

def test_output_df_has_topic_id_column():
    df = _make_df(["word1 word2 word3"] * 6)
    mock_tfidf, mock_lda = _make_sklearn_mocks()
    with patch(_TFIDF_CLS, mock_tfidf), patch(_LDA_CLS, mock_lda):
        result = TopicClassifier(n_topics=3).fit_transform(df)
    assert "topic_id" in result["df"].columns


def test_output_df_has_topic_score_column():
    df = _make_df(["word1 word2 word3"] * 6)
    mock_tfidf, mock_lda = _make_sklearn_mocks()
    with patch(_TFIDF_CLS, mock_tfidf), patch(_LDA_CLS, mock_lda):
        result = TopicClassifier(n_topics=3).fit_transform(df)
    assert "topic_score" in result["df"].columns


def test_topic_id_column_is_integer():
    df = _make_df(["word1 word2 word3"] * 6)
    mock_tfidf, mock_lda = _make_sklearn_mocks()
    with patch(_TFIDF_CLS, mock_tfidf), patch(_LDA_CLS, mock_lda):
        result = TopicClassifier(n_topics=3).fit_transform(df)
    assert result["df"]["topic_id"].dtype in (np.int32, np.int64, int)


def test_topic_score_column_is_float():
    df = _make_df(["word1 word2 word3"] * 6)
    mock_tfidf, mock_lda = _make_sklearn_mocks()
    with patch(_TFIDF_CLS, mock_tfidf), patch(_LDA_CLS, mock_lda):
        result = TopicClassifier(n_topics=3).fit_transform(df)
    assert result["df"]["topic_score"].dtype in (np.float32, np.float64, float)


def test_output_df_preserves_input_rows():
    df = _make_df(["word1 word2 word3"] * 6)
    mock_tfidf, mock_lda = _make_sklearn_mocks(n_messages=6)
    with patch(_TFIDF_CLS, mock_tfidf), patch(_LDA_CLS, mock_lda):
        result = TopicClassifier(n_topics=3).fit_transform(df)
    assert len(result["df"]) == len(df)


# --- group_topics DataFrame ---

def test_group_topics_has_expected_columns():
    df = _make_df(["word1 word2 word3"] * 6)
    mock_tfidf, mock_lda = _make_sklearn_mocks()
    with patch(_TFIDF_CLS, mock_tfidf), patch(_LDA_CLS, mock_lda):
        result = TopicClassifier(n_topics=3).fit_transform(df)
    gt = result["group_topics"]
    assert "topic_id" in gt.columns
    assert "topic_label" in gt.columns
    assert "weight" in gt.columns


def test_group_topics_row_count_matches_n_topics():
    df = _make_df(["word1 word2 word3"] * 6)
    mock_tfidf, mock_lda = _make_sklearn_mocks(n_topics=3)
    with patch(_TFIDF_CLS, mock_tfidf), patch(_LDA_CLS, mock_lda):
        result = TopicClassifier(n_topics=3).fit_transform(df)
    assert len(result["group_topics"]) == 3


def test_topic_label_uses_slash_separator():
    df = _make_df(["word1 word2 word3"] * 6)
    mock_tfidf, mock_lda = _make_sklearn_mocks()
    with patch(_TFIDF_CLS, mock_tfidf), patch(_LDA_CLS, mock_lda):
        result = TopicClassifier(n_topics=3).fit_transform(df)
    for label in result["group_topics"]["topic_label"]:
        assert " / " in label


def test_topic_label_contains_five_words():
    df = _make_df(["word1 word2 word3"] * 6)
    mock_tfidf, mock_lda = _make_sklearn_mocks()
    with patch(_TFIDF_CLS, mock_tfidf), patch(_LDA_CLS, mock_lda):
        result = TopicClassifier(n_topics=3).fit_transform(df)
    for label in result["group_topics"]["topic_label"]:
        assert len(label.split(" / ")) == 5


# --- sklearn wiring ---

def test_lda_fitted_with_correct_n_topics():
    df = _make_df(["word1 word2 word3"] * 6)
    mock_tfidf, mock_lda = _make_sklearn_mocks(n_topics=4, n_messages=6)
    with patch(_TFIDF_CLS, mock_tfidf), patch(_LDA_CLS, mock_lda):
        TopicClassifier(n_topics=4).fit_transform(df)
    mock_lda.assert_called_once_with(n_components=4, random_state=42)


def test_tfidf_vectorizer_is_called():
    df = _make_df(["word1 word2 word3"] * 6)
    mock_tfidf, mock_lda = _make_sklearn_mocks()
    with patch(_TFIDF_CLS, mock_tfidf), patch(_LDA_CLS, mock_lda):
        TopicClassifier(n_topics=3).fit_transform(df)
    mock_tfidf.assert_called_once()


# --- error cases ---

def test_empty_dataframe_raises_value_error():
    df = pd.DataFrame(columns=["cleaned_message"])
    with pytest.raises(ValueError, match="empty"):
        TopicClassifier().fit_transform(df)


def test_missing_cleaned_message_column_raises_value_error():
    df = pd.DataFrame({"message": ["hello world test"]})
    with pytest.raises(ValueError, match="cleaned_message"):
        TopicClassifier().fit_transform(df)


def test_unknown_method_raises_value_error():
    df = _make_df(["word1 word2 word3"] * 6)
    with pytest.raises(ValueError, match="Unknown method"):
        TopicClassifier(method="unknown").fit_transform(df)


def test_bertopic_raises_when_not_installed():
    df = _make_df(["word1 word2 word3"] * 6)
    # Simulate BERTopic not being installed by blocking the import
    with patch.dict("sys.modules", {"bertopic": None}):
        with pytest.raises(RuntimeError, match="BERTopic is not installed"):
            TopicClassifier(method="bertopic").fit_transform(df)


# --- input DataFrame is not mutated ---

def test_input_dataframe_is_not_mutated():
    df = _make_df(["word1 word2 word3"] * 6)
    original_columns = set(df.columns)
    mock_tfidf, mock_lda = _make_sklearn_mocks()
    with patch(_TFIDF_CLS, mock_tfidf), patch(_LDA_CLS, mock_lda):
        TopicClassifier(n_topics=3).fit_transform(df)
    assert set(df.columns) == original_columns


# ---------------------------------------------------------------------------
# BERTopic path (issue #09)
# bertopic is never imported for real — the whole module is mocked via
# patch.dict("sys.modules", {"bertopic": mock_module}).
# ---------------------------------------------------------------------------

def _make_bertopic_mock(n_topics: int = 3, n_messages: int = 6):
    """Return (mock_bertopic_module, mock_model_instance) for BERTopic tests."""
    topics_list = [i % n_topics for i in range(n_messages)]

    probs = np.zeros((n_messages, n_topics))
    remainder = 0.2 / max(n_topics - 1, 1)
    for i, t in enumerate(topics_list):
        probs[i, :] = remainder
        probs[i, t] = 0.8

    mock_model = MagicMock()
    mock_model.fit_transform.return_value = (topics_list, probs)
    mock_model.get_topic.return_value = [
        (f"word{j}", round(0.5 - j * 0.05, 2)) for j in range(5)
    ]

    mock_cls = MagicMock(return_value=mock_model)

    mock_module = MagicMock()
    mock_module.BERTopic = mock_cls

    return mock_module, mock_model


# --- return type and structure ---

def test_bertopic_fit_transform_returns_dict():
    df = _make_df(["word1 word2 word3"] * 6)
    mock_module, _ = _make_bertopic_mock()
    with patch.dict("sys.modules", {"bertopic": mock_module}):
        result = TopicClassifier(n_topics=3, method="bertopic").fit_transform(df)
    assert isinstance(result, dict)


def test_bertopic_output_has_df_key():
    df = _make_df(["word1 word2 word3"] * 6)
    mock_module, _ = _make_bertopic_mock()
    with patch.dict("sys.modules", {"bertopic": mock_module}):
        result = TopicClassifier(n_topics=3, method="bertopic").fit_transform(df)
    assert "df" in result


def test_bertopic_output_has_group_topics_key():
    df = _make_df(["word1 word2 word3"] * 6)
    mock_module, _ = _make_bertopic_mock()
    with patch.dict("sys.modules", {"bertopic": mock_module}):
        result = TopicClassifier(n_topics=3, method="bertopic").fit_transform(df)
    assert "group_topics" in result


# --- output DataFrame columns ---

def test_bertopic_df_has_topic_id_column():
    df = _make_df(["word1 word2 word3"] * 6)
    mock_module, _ = _make_bertopic_mock()
    with patch.dict("sys.modules", {"bertopic": mock_module}):
        result = TopicClassifier(n_topics=3, method="bertopic").fit_transform(df)
    assert "topic_id" in result["df"].columns


def test_bertopic_df_has_topic_score_column():
    df = _make_df(["word1 word2 word3"] * 6)
    mock_module, _ = _make_bertopic_mock()
    with patch.dict("sys.modules", {"bertopic": mock_module}):
        result = TopicClassifier(n_topics=3, method="bertopic").fit_transform(df)
    assert "topic_score" in result["df"].columns


def test_bertopic_topic_id_is_integer():
    df = _make_df(["word1 word2 word3"] * 6)
    mock_module, _ = _make_bertopic_mock()
    with patch.dict("sys.modules", {"bertopic": mock_module}):
        result = TopicClassifier(n_topics=3, method="bertopic").fit_transform(df)
    assert result["df"]["topic_id"].dtype in (np.int32, np.int64, int)


def test_bertopic_topic_score_is_float():
    df = _make_df(["word1 word2 word3"] * 6)
    mock_module, _ = _make_bertopic_mock()
    with patch.dict("sys.modules", {"bertopic": mock_module}):
        result = TopicClassifier(n_topics=3, method="bertopic").fit_transform(df)
    assert result["df"]["topic_score"].dtype in (np.float32, np.float64, float)


def test_bertopic_preserves_input_row_count():
    df = _make_df(["word1 word2 word3"] * 6)
    mock_module, _ = _make_bertopic_mock(n_messages=6)
    with patch.dict("sys.modules", {"bertopic": mock_module}):
        result = TopicClassifier(n_topics=3, method="bertopic").fit_transform(df)
    assert len(result["df"]) == len(df)


# --- group_topics ---

def test_bertopic_group_topics_has_expected_columns():
    df = _make_df(["word1 word2 word3"] * 6)
    mock_module, _ = _make_bertopic_mock()
    with patch.dict("sys.modules", {"bertopic": mock_module}):
        result = TopicClassifier(n_topics=3, method="bertopic").fit_transform(df)
    gt = result["group_topics"]
    assert "topic_id" in gt.columns
    assert "topic_label" in gt.columns
    assert "weight" in gt.columns


def test_bertopic_group_topics_label_uses_slash_separator():
    df = _make_df(["word1 word2 word3"] * 6)
    mock_module, _ = _make_bertopic_mock()
    with patch.dict("sys.modules", {"bertopic": mock_module}):
        result = TopicClassifier(n_topics=3, method="bertopic").fit_transform(df)
    for label in result["group_topics"]["topic_label"]:
        assert " / " in label


# --- BERTopic wiring ---

def test_bertopic_model_instantiated_with_n_topics():
    df = _make_df(["word1 word2 word3"] * 6)
    mock_module, _ = _make_bertopic_mock()
    with patch.dict("sys.modules", {"bertopic": mock_module}):
        TopicClassifier(n_topics=4, method="bertopic").fit_transform(df)
    mock_module.BERTopic.assert_called_once_with(nr_topics=4, calculate_probabilities=True)


def test_bertopic_get_topic_called_for_each_unique_topic():
    df = _make_df(["word1 word2 word3"] * 6)
    mock_module, mock_model = _make_bertopic_mock(n_topics=3, n_messages=6)
    with patch.dict("sys.modules", {"bertopic": mock_module}):
        TopicClassifier(n_topics=3, method="bertopic").fit_transform(df)
    # 3 unique topics → get_topic called 3 times
    assert mock_model.get_topic.call_count == 3


# --- outlier handling ---

def test_bertopic_outlier_topics_are_remapped_to_nonnegative():
    df = _make_df(["word1 word2 word3"] * 4)
    mock_module, mock_model = _make_bertopic_mock(n_topics=2, n_messages=4)
    probs_with_outlier = np.array([
        [0.8, 0.2],
        [0.2, 0.8],
        [0.5, 0.5],
        [0.3, 0.7],
    ])
    # First document assigned to outlier topic (-1)
    mock_model.fit_transform.return_value = ([-1, 1, 0, 1], probs_with_outlier)
    with patch.dict("sys.modules", {"bertopic": mock_module}):
        result = TopicClassifier(n_topics=2, method="bertopic").fit_transform(df)
    assert all(t >= 0 for t in result["df"]["topic_id"])


# --- LDA path unaffected ---

def test_lda_path_unaffected_after_bertopic_addition():
    df = _make_df(["word1 word2 word3"] * 6)
    mock_tfidf, mock_lda = _make_sklearn_mocks()
    with patch(_TFIDF_CLS, mock_tfidf), patch(_LDA_CLS, mock_lda):
        result = TopicClassifier(n_topics=3, method="lda").fit_transform(df)
    assert "topic_id" in result["df"].columns
    assert "group_topics" in result
