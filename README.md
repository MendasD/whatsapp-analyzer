# 💬 whatsapp-analyzer

![Python](https://img.shields.io/badge/python-%3E%3D3.12-blue)
![Tests](https://img.shields.io/badge/tests-383%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

A local-only Python pipeline for analysing WhatsApp group chat exports. Parse, clean, classify topics, score sentiment, profile users, and generate self-contained HTML reports — entirely on your device. No message content ever leaves your machine.

> 📖 **Full documentation** — [English](docs/documentation_en.md) · [Français](docs/documentation_fr.md)

## Table of contents

- [About](#about)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Data examples](#data-examples)
- [Web UI](#web-ui)
- [Documentation](#documentation)
- [Module overview](#module-overview)
- [Privacy and legal](#privacy-and-legal)
- [Contributing](#contributing)

## About

`whatsapp-analyzer` processes native WhatsApp exports (`.zip`, `_chat.txt`, or a decompressed folder) through a multi-step NLP pipeline:

1. **Load** — detect format, decompress ZIP if needed
2. **Parse** — extract messages into a structured DataFrame (supports Android and iOS export formats)
3. **Clean** — remove emoji, detect language, strip stopwords, lemmatise with spaCy
4. **Analyse** — topics (LDA / BERTopic), sentiment (VADER / CamemBERT), user profiles, temporal patterns, media stats
5. **Report** — self-contained HTML with charts, word clouds, and per-user breakdowns

## Installation

```bash
git clone https://github.com/MendasD/whatsapp-analyzer.git
cd whatsapp-analyzer
pip install -e .
```

spaCy language models are auto-downloaded on first run. To pre-install them:

```bash
python -m spacy download fr_core_news_sm
python -m spacy download en_core_web_sm
```

Optional extras:

```bash
pip install -e ".[bertopic]"   # BERTopic topic modelling
pip install -e ".[camembert]"  # CamemBERT French sentiment analysis
pip install -e ".[media]"      # Whisper audio/video transcription
```

## Quick start

### CLI

```bash
# Analyse a single group
whatsapp-analyzer analyze --input data-example/_chat.txt --topics 5 --output reports/

# Compare multiple groups
whatsapp-analyzer compare \
  --input data-example/_chat.txt \
  --input data-example/_chat0.txt \
  --output reports/

# Launch the web interface
whatsapp-analyzer serve
```

The `--user` and `--media` options for the `analyze` command are planned for a future release.

### Python API

```python
from whatsapp_analyzer import WhatsAppAnalyzer, GroupComparator

# Fluent pipeline — parse, clean, analyse, then export
az = WhatsAppAnalyzer("data-example/_chat.txt", n_topics=5)
az.parse().clean().analyze()
az.report(output="reports/")   # → reports/report.html
az.to_csv(output="reports/")   # → reports/_chat.csv

# One-call shorthand
results = az.run()             # parse → clean → analyze → report

# Per-user deep-dive
view = az.user("Alice")
view.summary()
view.topics()
view.sentiment_over_time()
view.activity_heatmap()
```

### Comparing multiple groups

```python
from pathlib import Path
from whatsapp_analyzer import WhatsAppAnalyzer
from whatsapp_analyzer.comparator import GroupComparator

az1 = WhatsAppAnalyzer("data-example/_chat.txt",  n_topics=5)
az2 = WhatsAppAnalyzer("data-example/_chat0.txt", n_topics=5)
az1.parse().clean().analyze()
az2.parse().clean().analyze()

comp = GroupComparator([az1, az2])
print(comp.compare_activity())
print(comp.compare_topics())
print(comp.compare_sentiment())
print(comp.common_users())
comp.report(Path("reports/"))  # → reports/comparison_report.html
```

## Data examples

The `data-example/` directory contains two anonymised export files — `_chat.txt` and `_chat0.txt` — that can be used immediately to explore the pipeline without any real data.

## Web UI

```bash
whatsapp-analyzer serve
# Opens at http://localhost:8501
```

The Streamlit interface lets you upload a `.zip` or `.txt` export, configure analysis parameters, and download the generated HTML report without writing any code. The UI is currently in progress and not all features are exposed yet.

## Documentation

Full documentation is available in two languages and covers installation, all CLI options, the complete Python API, module-by-module reference, output schemas, optional extras, and troubleshooting:

| Language | Link |
|----------|------|
| 🇬🇧 English | [docs/documentation_en.md](docs/documentation_en.md) |
| 🇫🇷 Français | [docs/documentation_fr.md](docs/documentation_fr.md) |

## Module overview

| Module | Role | Status |
|---|---|---|
| `utils.py` | Shared helpers: path resolution, anonymisation, language detection, logging | ✅ Done |
| `loader.py` | Format detection, ZIP decompression → `LoadedGroup` | ✅ Done |
| `parser.py` | Regex parsing of `_chat.txt` → DataFrame; Android and iOS formats | ✅ Done |
| `cleaner.py` | Emoji removal, language detection, stopwords (NLTK), spaCy lemmatisation | ✅ Done |
| `topic_classifier.py` | LDA topic modelling (sklearn), optional BERTopic | ✅ Done |
| `sentiment_analyzer.py` | VADER (default) or CamemBERT for French | ✅ Done |
| `temporal_analyzer.py` | Activity timelines, weekday×hour heatmaps, monthly stats | ✅ Done |
| `user_analyzer.py` | Per-user profiles: message count, top topics, mean sentiment, activity patterns | ✅ Done |
| `media_analyzer.py` | File stats by extension, optional Whisper transcription | ✅ Done |
| `comparator.py` | `GroupComparator`: compare activity, topics, sentiment, common users | ✅ Done |
| `visualizer.py` | matplotlib/seaborn/wordcloud charts, self-contained HTML reports | ✅ Done |
| `core.py` | `WhatsAppAnalyzer` orchestrator: fluent API, `run()` shorthand, `UserView` | ✅ Done |
| `cli.py` | Click CLI: `analyze`, `compare`, `serve` | ✅ Done |
| `app.py` | Streamlit web UI | 🔄 In progress |

## Privacy and legal

- All processing is 100% local. No message content is sent to any external server.
- Exports are generated by the user from their own groups — no automation of the WhatsApp interface is involved.
- Anonymisation (names and phone numbers) is available via the `anonymize=True` parameter on `WhatsAppAnalyzer`.
- Intended for personal or academic use. Raw chat data must not be redistributed.
- Compliant with WhatsApp (Meta) Terms of Service.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the project structure, coding conventions, branch naming, commit format, and test isolation rules.

```bash
# Run the full test suite
python -m pytest tests/

# Run a single module in isolation
python -m pytest tests/test_parser.py -v
```
