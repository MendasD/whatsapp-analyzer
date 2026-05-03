"""
Tests for cli.py.

Uses click.testing.CliRunner — no real files, no real pipeline.
WhatsAppAnalyzer, GroupComparator, and subprocess.run are all mocked.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pandas as pd
import pytest
from click.testing import CliRunner

from whatsapp_analyzer.cli import cli

# patch targets
_CORE_CLS = "whatsapp_analyzer.core.WhatsAppAnalyzer"
_COMPARATOR_CLS = "whatsapp_analyzer.comparator.GroupComparator"
_SUBPROCESS_RUN = "subprocess.run"


def _make_results(report_path: Path | None = None) -> dict:
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-10 09:00", "2024-01-11 10:00"]),
        "author": ["Aminata", "Moussa"],
        "message": ["hello", "world"],
    })
    return {
        "group_name": "TestGroup",
        "df_clean": df,
        "topics": {
            "group_topics": pd.DataFrame({
                "topic_id": [0, 1],
                "topic_label": ["sport / match / jouer", "cours / td / examen"],
                "weight": [0.6, 0.4],
            })
        },
        "sentiment": None,
        "temporal": None,
        "users": None,
        "report_path": report_path or Path("/tmp/reports/report.html"),
    }


def _mock_az(results: dict | None = None) -> MagicMock:
    az = MagicMock()
    az.run.return_value = results or _make_results()
    az._results = results or _make_results()
    return az


# analyze — success

def test_analyze_exits_zero_on_success(tmp_path):
    fake = tmp_path / "chat.zip"
    fake.write_bytes(b"")
    mock_az = _mock_az()
    with patch(_CORE_CLS, return_value=mock_az):
        result = CliRunner().invoke(cli, ["analyze", "--input", str(fake)])
    assert result.exit_code == 0


def test_analyze_output_contains_group_name(tmp_path):
    fake = tmp_path / "chat.zip"
    fake.write_bytes(b"")
    mock_az = _mock_az()
    with patch(_CORE_CLS, return_value=mock_az):
        result = CliRunner().invoke(cli, ["analyze", "--input", str(fake)])
    assert "TestGroup" in result.output


def test_analyze_output_contains_message_count(tmp_path):
    fake = tmp_path / "chat.zip"
    fake.write_bytes(b"")
    mock_az = _mock_az()
    with patch(_CORE_CLS, return_value=mock_az):
        result = CliRunner().invoke(cli, ["analyze", "--input", str(fake)])
    assert "2" in result.output


def test_analyze_output_contains_topic_label(tmp_path):
    fake = tmp_path / "chat.zip"
    fake.write_bytes(b"")
    mock_az = _mock_az()
    with patch(_CORE_CLS, return_value=mock_az):
        result = CliRunner().invoke(cli, ["analyze", "--input", str(fake)])
    assert "sport / match / jouer" in result.output


def test_analyze_output_contains_report_path(tmp_path):
    fake = tmp_path / "chat.zip"
    fake.write_bytes(b"")
    mock_az = _mock_az(_make_results(report_path=tmp_path / "report.html"))
    with patch(_CORE_CLS, return_value=mock_az):
        result = CliRunner().invoke(cli, ["analyze", "--input", str(fake)])
    assert "report.html" in result.output


def test_analyze_passes_topics_option(tmp_path):
    fake = tmp_path / "chat.zip"
    fake.write_bytes(b"")
    mock_cls = MagicMock(return_value=_mock_az())
    with patch(_CORE_CLS, mock_cls):
        CliRunner().invoke(cli, ["analyze", "--input", str(fake), "--topics", "8"])
    _, kwargs = mock_cls.call_args
    assert kwargs.get("n_topics") == 8 or mock_cls.call_args[0][1] == 8


def test_analyze_passes_output_option(tmp_path):
    fake = tmp_path / "chat.zip"
    fake.write_bytes(b"")
    out_dir = str(tmp_path / "out")
    mock_cls = MagicMock(return_value=_mock_az())
    with patch(_CORE_CLS, mock_cls):
        CliRunner().invoke(cli, ["analyze", "--input", str(fake), "--output", out_dir])
    args, kwargs = mock_cls.call_args
    assert out_dir in (str(a) for a in args) or str(kwargs.get("output_dir", "")) == out_dir


def test_analyze_exits_one_on_exception(tmp_path):
    fake = tmp_path / "chat.zip"
    fake.write_bytes(b"")
    mock_cls = MagicMock(side_effect=RuntimeError("pipeline failed"))
    with patch(_CORE_CLS, mock_cls):
        result = CliRunner().invoke(cli, ["analyze", "--input", str(fake)])
    assert result.exit_code == 1


def test_analyze_prints_error_message_on_exception(tmp_path):
    fake = tmp_path / "chat.zip"
    fake.write_bytes(b"")
    mock_cls = MagicMock(side_effect=RuntimeError("pipeline failed"))
    with patch(_CORE_CLS, mock_cls):
        result = CliRunner().invoke(cli, ["analyze", "--input", str(fake)])
    assert "Error" in result.output or "pipeline failed" in result.output


def test_analyze_no_traceback_on_exception(tmp_path):
    fake = tmp_path / "chat.zip"
    fake.write_bytes(b"")
    mock_cls = MagicMock(side_effect=RuntimeError("oops"))
    with patch(_CORE_CLS, mock_cls):
        result = CliRunner().invoke(cli, ["analyze", "--input", str(fake)])
    assert "Traceback" not in result.output


def test_analyze_missing_input_flag_exits_nonzero():
    result = CliRunner().invoke(cli, ["analyze"])
    assert result.exit_code != 0


def test_analyze_works_without_topics_data(tmp_path):
    fake = tmp_path / "chat.zip"
    fake.write_bytes(b"")
    results = _make_results()
    results["topics"] = None
    mock_az = _mock_az(results)
    with patch(_CORE_CLS, return_value=mock_az):
        result = CliRunner().invoke(cli, ["analyze", "--input", str(fake)])
    assert result.exit_code == 0


# compare — success

def test_compare_exits_zero_on_success(tmp_path):
    f1 = tmp_path / "g1.zip"
    f2 = tmp_path / "g2.zip"
    f1.write_bytes(b"")
    f2.write_bytes(b"")

    mock_az1 = _mock_az(_make_results())
    mock_az2 = _mock_az(_make_results())
    mock_az1._results["group_name"] = "G1"
    mock_az2._results["group_name"] = "G2"

    mock_cmp = MagicMock()
    mock_cmp.compare_activity.return_value = pd.DataFrame(
        {"nb_messages": [2, 3]}, index=["G1", "G2"]
    )
    mock_cmp.report.return_value = tmp_path / "comparison_report.html"

    with patch(_CORE_CLS, side_effect=[mock_az1, mock_az2]), \
         patch(_COMPARATOR_CLS, return_value=mock_cmp):
        result = CliRunner().invoke(
            cli, ["compare", "--input", str(f1), "--input", str(f2)]
        )
    assert result.exit_code == 0


def test_compare_output_contains_comparison_report(tmp_path):
    f1 = tmp_path / "g1.zip"
    f1.write_bytes(b"")
    mock_az = _mock_az()
    mock_cmp = MagicMock()
    mock_cmp.compare_activity.return_value = pd.DataFrame(
        {"nb_messages": [2]}, index=["TestGroup"]
    )
    mock_cmp.report.return_value = tmp_path / "comparison_report.html"
    with patch(_CORE_CLS, return_value=mock_az), \
         patch(_COMPARATOR_CLS, return_value=mock_cmp):
        result = CliRunner().invoke(cli, ["compare", "--input", str(f1)])
    assert "comparison_report.html" in result.output


def test_compare_exits_one_on_exception(tmp_path):
    f1 = tmp_path / "g1.zip"
    f1.write_bytes(b"")
    with patch(_CORE_CLS, side_effect=RuntimeError("load failed")):
        result = CliRunner().invoke(cli, ["compare", "--input", str(f1)])
    assert result.exit_code == 1


def test_compare_no_traceback_on_exception(tmp_path):
    f1 = tmp_path / "g1.zip"
    f1.write_bytes(b"")
    with patch(_CORE_CLS, side_effect=RuntimeError("load failed")):
        result = CliRunner().invoke(cli, ["compare", "--input", str(f1)])
    assert "Traceback" not in result.output


def test_compare_missing_input_flag_exits_nonzero():
    result = CliRunner().invoke(cli, ["compare"])
    assert result.exit_code != 0


# serve

def test_serve_calls_streamlit_run():
    with patch(_SUBPROCESS_RUN) as mock_sub:
        CliRunner().invoke(cli, ["serve"])
    mock_sub.assert_called_once()
    cmd = mock_sub.call_args[0][0]
    assert cmd[0] == "streamlit"
    assert cmd[1] == "run"


def test_serve_passes_app_py_path():
    with patch(_SUBPROCESS_RUN) as mock_sub:
        CliRunner().invoke(cli, ["serve"])
    cmd = mock_sub.call_args[0][0]
    assert cmd[2].endswith("app.py")


def test_serve_exits_zero_on_success():
    with patch(_SUBPROCESS_RUN):
        result = CliRunner().invoke(cli, ["serve"])
    assert result.exit_code == 0


def test_serve_exits_one_on_exception():
    with patch(_SUBPROCESS_RUN, side_effect=FileNotFoundError("streamlit not found")):
        result = CliRunner().invoke(cli, ["serve"])
    assert result.exit_code == 1


def test_serve_prints_error_on_exception():
    with patch(_SUBPROCESS_RUN, side_effect=FileNotFoundError("streamlit not found")):
        result = CliRunner().invoke(cli, ["serve"])
    assert "Error" in result.output or "streamlit not found" in result.output


def test_serve_no_traceback_on_exception():
    with patch(_SUBPROCESS_RUN, side_effect=FileNotFoundError("streamlit not found")):
        result = CliRunner().invoke(cli, ["serve"])
    assert "Traceback" not in result.output
