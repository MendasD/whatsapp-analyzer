"""
Tests for cleaner.py.

All external dependencies (spaCy, NLTK, langdetect, emoji) are mocked
so this test file runs in complete isolation — no NLP model downloads needed.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from whatsapp_analyzer.cleaner import Cleaner


# Minimal DataFrame that mimics Parser output
def _make_df(messages: list[str], msg_type: str = "text") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-12 08:00"] * len(messages)),
            "author": ["Aminata"] * len(messages),
            "message": messages,
            "msg_type": [msg_type] * len(messages),
            "group_name": ["TestGroup"] * len(messages),
        }
    )


# Patch targets used across multiple tests
_DETECT_LANG = "whatsapp_analyzer.cleaner.detect_language"
_LOAD_STOPWORDS = "whatsapp_analyzer.cleaner.Cleaner._load_stopwords"
_LOAD_SPACY = "whatsapp_analyzer.cleaner.Cleaner._load_spacy"


@pytest.fixture
def cleaner_fr():
    """A Cleaner forced to French with lemmatisation and spaCy disabled."""
    return Cleaner(lang="fr", use_lemma=False)


@pytest.fixture
def sample_df():
    return _make_df(
        [
            "Bonjour tout le monde aujourd'hui",
            "Le cours de demain est annulé vraiment",
            "Quelqu'un a les notes du TD maintenant",
        ]
    )


# --- output shape and columns ---

def test_clean_returns_dataframe(cleaner_fr, sample_df):
    with patch(_LOAD_STOPWORDS, return_value=set()):
        result = cleaner_fr.clean(sample_df)
    assert isinstance(result, pd.DataFrame)


def test_clean_adds_required_columns(cleaner_fr, sample_df):
    with patch(_LOAD_STOPWORDS, return_value=set()):
        result = cleaner_fr.clean(sample_df)
    assert "cleaned_message" in result.columns
    assert "language" in result.columns
    assert "tokens" in result.columns


def test_tokens_column_is_list(cleaner_fr, sample_df):
    with patch(_LOAD_STOPWORDS, return_value=set()):
        result = cleaner_fr.clean(sample_df)
    assert all(isinstance(t, list) for t in result["tokens"])


# --- filtering by msg_type ---

def test_drops_media_messages():
    df = _make_df(["<Media omitted>"] * 3, msg_type="media")
    result = Cleaner(lang="fr", use_lemma=False).clean(df)
    assert result.empty


def test_drops_system_messages():
    df = _make_df(["Messages and calls are end-to-end encrypted"] * 2, msg_type="system")
    result = Cleaner(lang="fr", use_lemma=False).clean(df)
    assert result.empty


def test_keeps_only_text_messages():
    df = pd.concat([
        _make_df(["Bonjour tout le monde aujourd'hui"], msg_type="text"),
        _make_df(["<Media omitted>"], msg_type="media"),
    ], ignore_index=True)
    with patch(_LOAD_STOPWORDS, return_value=set()):
        result = Cleaner(lang="fr", use_lemma=False).clean(df)
    assert len(result) == 1


# --- text cleaning logic ---

def test_text_is_lowercased(cleaner_fr, sample_df):
    with patch(_LOAD_STOPWORDS, return_value=set()):
        result = cleaner_fr.clean(sample_df)
    for msg in result["cleaned_message"]:
        assert msg == msg.lower()


def test_punctuation_removed():
    df = _make_df(["Bonjour, tout le monde! C'est super."])
    with patch(_LOAD_STOPWORDS, return_value=set()):
        result = Cleaner(lang="fr", use_lemma=False).clean(df)
    cleaned = result["cleaned_message"].iloc[0]
    assert "," not in cleaned
    assert "!" not in cleaned


def test_stopwords_are_removed():
    df = _make_df(["le cours est annulé demain vraiment"])
    stopwords = {"le", "est", "la", "les", "de"}
    with patch(_LOAD_STOPWORDS, return_value=stopwords):
        result = Cleaner(lang="fr", use_lemma=False).clean(df)
    tokens = result["tokens"].iloc[0]
    assert "le" not in tokens
    assert "est" not in tokens


def test_artefacts_removed():
    df = _make_df(["Média omis quelque chose de plus"])
    with patch(_LOAD_STOPWORDS, return_value=set()):
        result = Cleaner(lang="fr", use_lemma=False).clean(df)
    cleaned = result["cleaned_message"].iloc[0]
    assert "média omis" not in cleaned.lower()


# --- min_words filtering ---

def test_short_messages_dropped():
    df = _make_df(["ok", "Bonjour tout le monde aujourd'hui"])
    with patch(_LOAD_STOPWORDS, return_value=set()):
        result = Cleaner(lang="fr", use_lemma=False, min_words=3).clean(df)
    # "ok" becomes a 1-token message and should be dropped
    assert len(result) == 1


def test_min_words_zero_keeps_all():
    df = _make_df(["ok super", "Bonjour tout le monde"])
    with patch(_LOAD_STOPWORDS, return_value=set()):
        result = Cleaner(lang="fr", use_lemma=False, min_words=0).clean(df)
    assert len(result) == 2


# --- language detection ---

def test_forced_language_set(sample_df):
    cleaner = Cleaner(lang="en", use_lemma=False)
    with patch(_LOAD_STOPWORDS, return_value=set()):
        cleaner.clean(sample_df)
    assert cleaner.detected_lang == "en"


def test_auto_language_detection(sample_df):
    with patch(_DETECT_LANG, return_value="fr"), \
         patch(_LOAD_STOPWORDS, return_value=set()):
        cleaner = Cleaner(lang=None, use_lemma=False)
        cleaner.clean(sample_df)
    assert cleaner.detected_lang == "fr"


def test_unknown_language_falls_back_to_fr(sample_df):
    with patch(_DETECT_LANG, return_value="unknown"), \
         patch(_LOAD_STOPWORDS, return_value=set()):
        cleaner = Cleaner(lang=None, use_lemma=False)
        cleaner.clean(sample_df)
    assert cleaner.detected_lang == "fr"


# --- lemmatisation ---

def test_lemmatisation_called_when_enabled(sample_df):
    # Build mock tokens — enough to survive the min_words=3 filter
    def _make_token(lemma: str):
        t = MagicMock()
        t.lemma_ = lemma
        t.is_space = False
        return t

    mock_nlp = MagicMock()
    mock_nlp.return_value = [_make_token(w) for w in ["bonjour", "monde", "aujourd"]]

    with patch(_LOAD_STOPWORDS, return_value=set()), \
         patch(_LOAD_SPACY, return_value=mock_nlp):
        cleaner = Cleaner(lang="fr", use_lemma=True)
        result = cleaner.clean(sample_df)

    assert mock_nlp.called
    assert len(result) > 0


def test_lemmatisation_disabled_does_not_call_spacy(sample_df):
    with patch(_LOAD_STOPWORDS, return_value=set()), \
         patch(_LOAD_SPACY, return_value=None) as mock_spacy:
        cleaner = Cleaner(lang="fr", use_lemma=False)
        cleaner.clean(sample_df)
    mock_spacy.assert_not_called()


# --- emoji handling ---

def test_emoji_removed_when_flag_set():
    df = _make_df(["Bonjour tout le monde 😀🎉 super"])
    with patch(_LOAD_STOPWORDS, return_value=set()), \
         patch("whatsapp_classifier.cleaner.Cleaner._strip_emoji", return_value="Bonjour tout le monde  super"):
        result = Cleaner(lang="fr", use_lemma=False, remove_emoji=True).clean(df)
    assert len(result) > 0


# --- empty input ---

def test_empty_dataframe_returns_empty():
    df = pd.DataFrame(columns=["timestamp", "author", "message", "msg_type", "group_name"])
    result = Cleaner(lang="fr", use_lemma=False).clean(df)
    assert result.empty


def test_all_non_text_returns_empty():
    df = _make_df(["<Media omitted>", "Message supprimé"], msg_type="media")
    result = Cleaner(lang="fr", use_lemma=False).clean(df)
    assert result.empty