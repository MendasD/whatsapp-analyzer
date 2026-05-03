"""
Tests for core.py.

All sub-modules are mocked — no real files, no NLP, no rendering.
State is injected directly into _results where needed to test steps
in isolation without relying on the preceding step.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pandas as pd
import pytest

from whatsapp_analyzer.core import UserView, WhatsAppAnalyzer

# patch targets
_LOADER = "whatsapp_analyzer.loader.Loader"
_PARSER = "whatsapp_analyzer.parser.Parser"
_CLEANER = "whatsapp_analyzer.cleaner.Cleaner"
_TOPIC_CLS = "whatsapp_analyzer.topic_classifier.TopicClassifier"
_SENT_CLS = "whatsapp_analyzer.sentiment_analyzer.SentimentAnalyzer"
_TEMP_CLS = "whatsapp_analyzer.temporal_analyzer.TemporalAnalyzer"
_MEDIA_CLS = "whatsapp_analyzer.media_analyzer.MediaAnalyzer"
_USER_CLS = "whatsapp_analyzer.user_analyzer.UserAnalyzer"
_VISUALIZER = "whatsapp_analyzer.visualizer.Visualizer"


def _make_df():
    return pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-10 09:00", "2024-01-11 10:00"]),
        "author": ["Alice", "Bob"],
        "message": ["hello", "world"],
        "msg_type": ["text", "text"],
        "group_name": ["TestGroup", "TestGroup"],
        "cleaned_message": ["hello", "world"],
        "language": ["fr", "fr"],
        "tokens": [["hello"], ["world"]],
    })


def _make_loaded(tmp_path, group_name="TestGroup"):
    """Fake LoadedGroup-like object."""
    loaded = MagicMock()
    loaded.group_name = group_name
    loaded.chat_path = tmp_path / "_chat.txt"
    loaded.media_dir = tmp_path / "media"
    return loaded


# init

class TestInit:

    def test_input_path_stored_as_path(self, tmp_path):
        az = WhatsAppAnalyzer(tmp_path / "chat.zip")
        assert isinstance(az.input_path, Path)

    def test_default_n_topics(self, tmp_path):
        az = WhatsAppAnalyzer(tmp_path / "chat.zip")
        assert az.n_topics == 5

    def test_custom_n_topics(self, tmp_path):
        az = WhatsAppAnalyzer(tmp_path / "chat.zip", n_topics=8)
        assert az.n_topics == 8

    def test_default_lang_is_none(self, tmp_path):
        az = WhatsAppAnalyzer(tmp_path / "chat.zip")
        assert az.lang is None

    def test_results_initially_empty(self, tmp_path):
        az = WhatsAppAnalyzer(tmp_path / "chat.zip")
        assert az._results == {}


# parse()

class TestParse:

    def test_returns_self(self, tmp_path):
        az = WhatsAppAnalyzer(tmp_path / "chat.zip")
        loaded = _make_loaded(tmp_path)
        with patch(_LOADER) as mlc, patch(_PARSER) as mpc:
            mlc.return_value.load.return_value = loaded
            mpc.return_value.parse.return_value = _make_df()
            result = az.parse()
        assert result is az

    def test_stores_group_name(self, tmp_path):
        az = WhatsAppAnalyzer(tmp_path / "chat.zip")
        loaded = _make_loaded(tmp_path, "MaGroupe")
        with patch(_LOADER) as mlc, patch(_PARSER) as mpc:
            mlc.return_value.load.return_value = loaded
            mpc.return_value.parse.return_value = _make_df()
            az.parse()
        assert az._results["group_name"] == "MaGroupe"

    def test_stores_df_raw(self, tmp_path):
        az = WhatsAppAnalyzer(tmp_path / "chat.zip")
        df = _make_df()
        loaded = _make_loaded(tmp_path)
        with patch(_LOADER) as mlc, patch(_PARSER) as mpc:
            mlc.return_value.load.return_value = loaded
            mpc.return_value.parse.return_value = df
            az.parse()
        assert "df_raw" in az._results

    def test_stores_media_dir(self, tmp_path):
        az = WhatsAppAnalyzer(tmp_path / "chat.zip")
        loaded = _make_loaded(tmp_path)
        with patch(_LOADER) as mlc, patch(_PARSER) as mpc:
            mlc.return_value.load.return_value = loaded
            mpc.return_value.parse.return_value = _make_df()
            az.parse()
        assert az._media_dir == loaded.media_dir

    def test_loader_called_with_input_path(self, tmp_path):
        input_path = tmp_path / "chat.zip"
        az = WhatsAppAnalyzer(input_path)
        loaded = _make_loaded(tmp_path)
        with patch(_LOADER) as mlc, patch(_PARSER) as mpc:
            mlc.return_value.load.return_value = loaded
            mpc.return_value.parse.return_value = _make_df()
            az.parse()
        mlc.return_value.load.assert_called_once_with(input_path)


# clean()

class TestClean:

    def test_returns_self(self, tmp_path):
        az = WhatsAppAnalyzer(tmp_path / "chat.zip")
        az._results["df_raw"] = _make_df()
        with patch(_CLEANER) as mcc:
            mcc.return_value.clean.return_value = _make_df()
            result = az.clean()
        assert result is az

    def test_stores_df_clean(self, tmp_path):
        az = WhatsAppAnalyzer(tmp_path / "chat.zip")
        az._results["df_raw"] = _make_df()
        cleaned = _make_df()
        with patch(_CLEANER) as mcc:
            mcc.return_value.clean.return_value = cleaned
            az.clean()
        assert az._results["df_clean"] is cleaned

    def test_raises_if_parse_not_called(self, tmp_path):
        az = WhatsAppAnalyzer(tmp_path / "chat.zip")
        with pytest.raises(RuntimeError, match="parse"):
            az.clean()

    def test_passes_lang_to_cleaner(self, tmp_path):
        az = WhatsAppAnalyzer(tmp_path / "chat.zip", lang="fr")
        az._results["df_raw"] = _make_df()
        with patch(_CLEANER) as mcc:
            mcc.return_value.clean.return_value = _make_df()
            az.clean()
        _, kwargs = mcc.call_args
        assert kwargs.get("lang") == "fr" or mcc.call_args[0][0] == "fr"

    def test_clean_lang_override_takes_precedence(self, tmp_path):
        az = WhatsAppAnalyzer(tmp_path / "chat.zip", lang="fr")
        az._results["df_raw"] = _make_df()
        with patch(_CLEANER) as mcc:
            mcc.return_value.clean.return_value = _make_df()
            az.clean(lang="en")
        args, kwargs = mcc.call_args
        lang_value = kwargs.get("lang") or (args[0] if args else None)
        assert lang_value == "en"

    def test_passes_min_words_to_cleaner(self, tmp_path):
        az = WhatsAppAnalyzer(tmp_path / "chat.zip", min_words=7)
        az._results["df_raw"] = _make_df()
        with patch(_CLEANER) as mcc:
            mcc.return_value.clean.return_value = _make_df()
            az.clean()
        args, kwargs = mcc.call_args
        min_w = kwargs.get("min_words") or (args[1] if len(args) > 1 else None)
        assert min_w == 7


# analyze()

class TestAnalyze:

    def _az_with_clean(self, tmp_path):
        az = WhatsAppAnalyzer(tmp_path / "chat.zip")
        az._results["df_clean"] = _make_df()
        return az

    def test_returns_self(self, tmp_path):
        az = self._az_with_clean(tmp_path)
        with patch(_TOPIC_CLS), patch(_SENT_CLS, create=True), \
             patch(_TEMP_CLS), patch(_USER_CLS):
            result = az.analyze()
        assert result is az

    def test_raises_if_clean_not_called(self, tmp_path):
        az = WhatsAppAnalyzer(tmp_path / "chat.zip")
        with pytest.raises(RuntimeError, match="clean"):
            az.analyze()

    def test_calls_topic_classifier(self, tmp_path):
        az = self._az_with_clean(tmp_path)
        mock_topics = MagicMock()
        with patch(_TOPIC_CLS) as mtc, patch(_SENT_CLS, create=True), \
             patch(_TEMP_CLS), patch(_USER_CLS):
            mtc.return_value.fit_transform.return_value = mock_topics
            az.analyze(topics=True, sentiment=False, temporal=False)
        mtc.return_value.fit_transform.assert_called_once()

    def test_topics_disabled_skips_classifier(self, tmp_path):
        az = self._az_with_clean(tmp_path)
        with patch(_TOPIC_CLS) as mtc, patch(_SENT_CLS, create=True), \
             patch(_TEMP_CLS), patch(_USER_CLS):
            az.analyze(topics=False, sentiment=False, temporal=False)
        mtc.return_value.fit_transform.assert_not_called()

    def test_calls_temporal_analyzer(self, tmp_path):
        az = self._az_with_clean(tmp_path)
        with patch(_TOPIC_CLS), patch(_SENT_CLS, create=True), \
             patch(_TEMP_CLS) as mtm, patch(_USER_CLS):
            az.analyze(topics=False, sentiment=False, temporal=True)
        mtm.return_value.analyze.assert_called_once()

    def test_temporal_disabled_skips_analyzer(self, tmp_path):
        az = self._az_with_clean(tmp_path)
        with patch(_TOPIC_CLS), patch(_SENT_CLS, create=True), \
             patch(_TEMP_CLS) as mtm, patch(_USER_CLS):
            az.analyze(topics=False, sentiment=False, temporal=False)
        mtm.return_value.analyze.assert_not_called()

    def test_media_disabled_by_default(self, tmp_path):
        az = self._az_with_clean(tmp_path)
        az._media_dir = tmp_path / "media"
        with patch(_TOPIC_CLS), patch(_SENT_CLS, create=True), \
             patch(_TEMP_CLS), patch(_USER_CLS), patch(_MEDIA_CLS) as mmc:
            az.analyze(media=False)
        mmc.return_value.analyze.assert_not_called()

    def test_media_enabled_calls_media_analyzer(self, tmp_path):
        az = self._az_with_clean(tmp_path)
        az._media_dir = tmp_path / "media"
        with patch(_TOPIC_CLS), patch(_SENT_CLS, create=True), \
             patch(_TEMP_CLS), patch(_USER_CLS), patch(_MEDIA_CLS) as mmc:
            az.analyze(media=True)
        mmc.return_value.analyze.assert_called_once_with(az._media_dir)

    def test_media_enabled_but_no_media_dir_skips(self, tmp_path):
        az = self._az_with_clean(tmp_path)
        az._media_dir = None
        with patch(_TOPIC_CLS), patch(_SENT_CLS, create=True), \
             patch(_TEMP_CLS), patch(_USER_CLS), patch(_MEDIA_CLS) as mmc:
            az.analyze(media=True)
        mmc.return_value.analyze.assert_not_called()

    def test_always_calls_user_analyzer(self, tmp_path):
        az = self._az_with_clean(tmp_path)
        with patch(_TOPIC_CLS), patch(_SENT_CLS, create=True), \
             patch(_TEMP_CLS), patch(_USER_CLS) as muc:
            az.analyze(topics=False, sentiment=False, temporal=False)
        muc.return_value.build_profiles.assert_called_once()

    def test_step_failure_sets_key_to_none(self, tmp_path):
        az = self._az_with_clean(tmp_path)
        with patch(_TOPIC_CLS) as mtc, patch(_SENT_CLS, create=True), \
             patch(_TEMP_CLS), patch(_USER_CLS):
            mtc.return_value.fit_transform.side_effect = RuntimeError("boom")
            az.analyze(topics=True, sentiment=False, temporal=False)
        assert az._results.get("topics") is None

    def test_step_failure_does_not_crash_pipeline(self, tmp_path):
        az = self._az_with_clean(tmp_path)
        with patch(_TOPIC_CLS) as mtc, patch(_SENT_CLS, create=True), \
             patch(_TEMP_CLS) as mtm, patch(_USER_CLS):
            mtc.return_value.fit_transform.side_effect = RuntimeError("topics fail")
            az.analyze(topics=True, sentiment=False, temporal=True)
        mtm.return_value.analyze.assert_called_once()

    def test_topic_classifier_receives_n_topics(self, tmp_path):
        az = WhatsAppAnalyzer(tmp_path / "chat.zip", n_topics=7)
        az._results["df_clean"] = _make_df()
        with patch(_TOPIC_CLS) as mtc, patch(_SENT_CLS, create=True), \
             patch(_TEMP_CLS), patch(_USER_CLS):
            az.analyze(topics=True, sentiment=False, temporal=False)
        args, kwargs = mtc.call_args
        n = kwargs.get("n_topics") or (args[0] if args else None)
        assert n == 7


# report()

class TestReport:

    def test_returns_path(self, tmp_path):
        az = WhatsAppAnalyzer(tmp_path / "chat.zip")
        az._results = {"group_name": "G", "df_clean": _make_df()}
        with patch(_VISUALIZER) as mvc:
            mvc.return_value.generate_report.return_value = tmp_path / "report.html"
            result = az.report()
        assert isinstance(result, Path)

    def test_delegates_to_visualizer(self, tmp_path):
        az = WhatsAppAnalyzer(tmp_path / "chat.zip")
        az._results = {"group_name": "G", "df_clean": _make_df()}
        with patch(_VISUALIZER) as mvc:
            mvc.return_value.generate_report.return_value = tmp_path / "report.html"
            az.report()
        mvc.return_value.generate_report.assert_called_once()

    def test_custom_output_dir_passed(self, tmp_path):
        az = WhatsAppAnalyzer(tmp_path / "chat.zip")
        az._results = {"group_name": "G", "df_clean": _make_df()}
        custom = tmp_path / "custom_out"
        with patch(_VISUALIZER) as mvc:
            mvc.return_value.generate_report.return_value = custom / "report.html"
            az.report(output=custom)
        _, kwargs = mvc.return_value.generate_report.call_args
        passed_dir = mvc.return_value.generate_report.call_args[0][1]
        assert str(custom) in str(passed_dir)


# to_csv() 

class TestToCsv:

    def test_raises_if_no_df_clean(self, tmp_path):
        az = WhatsAppAnalyzer(tmp_path / "chat.zip")
        with pytest.raises(RuntimeError, match="clean"):
            az.to_csv(tmp_path)

    def test_returns_path(self, tmp_path):
        az = WhatsAppAnalyzer(tmp_path / "chat.zip")
        az._results = {"group_name": "TestGroup", "df_clean": _make_df()}
        result = az.to_csv(tmp_path)
        assert isinstance(result, Path)

    def test_creates_csv_file(self, tmp_path):
        az = WhatsAppAnalyzer(tmp_path / "chat.zip")
        az._results = {"group_name": "TestGroup", "df_clean": _make_df()}
        path = az.to_csv(tmp_path)
        assert path.exists()

    def test_filename_contains_group_name(self, tmp_path):
        az = WhatsAppAnalyzer(tmp_path / "chat.zip")
        az._results = {"group_name": "Famille", "df_clean": _make_df()}
        path = az.to_csv(tmp_path)
        assert "Famille" in path.name

    def test_csv_is_readable_dataframe(self, tmp_path):
        az = WhatsAppAnalyzer(tmp_path / "chat.zip")
        df = _make_df()
        az._results = {"group_name": "G", "df_clean": df}
        path = az.to_csv(tmp_path)
        loaded = pd.read_csv(path)
        assert len(loaded) == len(df)

    def test_creates_output_dir_if_missing(self, tmp_path):
        az = WhatsAppAnalyzer(tmp_path / "chat.zip")
        az._results = {"group_name": "G", "df_clean": _make_df()}
        new_dir = tmp_path / "sub" / "dir"
        az.to_csv(new_dir)
        assert new_dir.exists()


# user() and UserView 

class TestUserView:

    def _az_with_results(self, tmp_path):
        az = WhatsAppAnalyzer(tmp_path / "chat.zip")
        az._results = {"group_name": "G", "df_clean": _make_df()}
        return az

    def test_user_returns_user_view(self, tmp_path):
        az = self._az_with_results(tmp_path)
        assert isinstance(az.user("Alice"), UserView)

    def test_user_view_holds_author(self, tmp_path):
        az = self._az_with_results(tmp_path)
        view = az.user("Alice")
        assert view._author == "Alice"

    def test_summary_delegates_to_user_analyzer(self, tmp_path):
        az = self._az_with_results(tmp_path)
        view = az.user("Alice")
        mock_profile = {"message_count": 2, "most_active_day": "Friday"}
        with patch(_USER_CLS) as muc:
            muc.summary_for.return_value = mock_profile
            result = view.summary()
        muc.summary_for.assert_called_once_with("Alice", az._results)
        assert result == mock_profile

    def test_topics_delegates_to_user_analyzer(self, tmp_path):
        az = self._az_with_results(tmp_path)
        view = az.user("Alice")
        mock_df = pd.DataFrame(columns=["topic_id", "topic_label", "topic_score"])
        with patch(_USER_CLS) as muc:
            muc.topics_for.return_value = mock_df
            result = view.topics()
        muc.topics_for.assert_called_once_with("Alice", az._results)

    def test_sentiment_over_time_delegates(self, tmp_path):
        az = self._az_with_results(tmp_path)
        view = az.user("Alice")
        mock_df = pd.DataFrame(columns=["timestamp", "sentiment_score", "sentiment_label"])
        with patch(_USER_CLS) as muc:
            muc.sentiment_over_time_for.return_value = mock_df
            view.sentiment_over_time()
        muc.sentiment_over_time_for.assert_called_once_with("Alice", az._results)

    def test_activity_heatmap_delegates(self, tmp_path):
        az = self._az_with_results(tmp_path)
        view = az.user("Alice")
        mock_df = MagicMock()
        with patch(_USER_CLS) as muc:
            muc.activity_heatmap_for.return_value = mock_df
            view.activity_heatmap()
        muc.activity_heatmap_for.assert_called_once_with("Alice", az._results)


# chaining 

class TestChaining:

    def test_parse_clean_analyze_chaining_returns_self(self, tmp_path):
        az = WhatsAppAnalyzer(tmp_path / "chat.zip")
        loaded = _make_loaded(tmp_path)
        df = _make_df()
        with patch(_LOADER) as mlc, patch(_PARSER) as mpc, \
             patch(_CLEANER) as mcc, patch(_TOPIC_CLS), \
             patch(_SENT_CLS, create=True), patch(_TEMP_CLS), patch(_USER_CLS):
            mlc.return_value.load.return_value = loaded
            mpc.return_value.parse.return_value = df
            mcc.return_value.clean.return_value = df
            result = az.parse().clean().analyze()
        assert result is az

    def test_parse_clean_analyze_report_returns_path(self, tmp_path):
        az = WhatsAppAnalyzer(tmp_path / "chat.zip")
        loaded = _make_loaded(tmp_path)
        df = _make_df()
        with patch(_LOADER) as mlc, patch(_PARSER) as mpc, \
             patch(_CLEANER) as mcc, patch(_TOPIC_CLS), \
             patch(_SENT_CLS, create=True), patch(_TEMP_CLS), \
             patch(_USER_CLS), patch(_VISUALIZER) as mvc:
            mlc.return_value.load.return_value = loaded
            mpc.return_value.parse.return_value = df
            mcc.return_value.clean.return_value = df
            mvc.return_value.generate_report.return_value = tmp_path / "report.html"
            path = az.parse().clean().analyze().report()
        assert isinstance(path, Path)

    def test_run_convenience_calls_all_steps(self, tmp_path):
        az = WhatsAppAnalyzer(tmp_path / "chat.zip")
        loaded = _make_loaded(tmp_path)
        df = _make_df()
        with patch(_LOADER) as mlc, patch(_PARSER) as mpc, \
             patch(_CLEANER) as mcc, patch(_TOPIC_CLS), \
             patch(_SENT_CLS, create=True), patch(_TEMP_CLS), \
             patch(_USER_CLS), patch(_VISUALIZER) as mvc:
            mlc.return_value.load.return_value = loaded
            mpc.return_value.parse.return_value = df
            mcc.return_value.clean.return_value = df
            mvc.return_value.generate_report.return_value = tmp_path / "report.html"
            results = az.run()
        assert isinstance(results, dict)
        assert "report_path" in results


# __init__.py public API 

class TestPublicApi:

    def test_whatsapp_analyzer_importable_from_package(self):
        from whatsapp_analyzer import WhatsAppAnalyzer as WA
        assert WA is WhatsAppAnalyzer

    def test_group_comparator_importable_from_package(self):
        from whatsapp_analyzer import GroupComparator
        from whatsapp_analyzer.comparator import GroupComparator as GC
        assert GroupComparator is GC
