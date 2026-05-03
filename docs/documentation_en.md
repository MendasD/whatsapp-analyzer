# whatsapp-analyzer — Complete Documentation

> **Language / Langue :** English · [Français](documentation_fr.md)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Installation](#2-installation)
3. [Quick Start](#3-quick-start)
   - 3.1 [CLI](#31-cli)
   - 3.2 [Python API](#32-python-api)
   - 3.3 [Web interface](#33-web-interface)
4. [Input formats](#4-input-formats)
5. [Pipeline architecture](#5-pipeline-architecture)
6. [Module reference](#6-module-reference)
   - 6.1 [Loader](#61-loader)
   - 6.2 [Parser](#62-parser)
   - 6.3 [Cleaner](#63-cleaner)
   - 6.4 [TopicClassifier](#64-topicclassifier)
   - 6.5 [SentimentAnalyzer](#65-sentimentanalyzer)
   - 6.6 [TemporalAnalyzer](#66-temporalanalyzer)
   - 6.7 [UserAnalyzer](#67-useranalyzer)
   - 6.8 [MediaAnalyzer](#68-mediaanalyzer)
   - 6.9 [GroupComparator](#69-groupcomparator)
   - 6.10 [Visualizer](#610-visualizer)
   - 6.11 [WhatsAppAnalyzer (core)](#611-whatsappanalyzer-core)
   - 6.12 [UserView](#612-userview)
   - 6.13 [CLI commands](#613-cli-commands)
7. [Output reference](#7-output-reference)
8. [Optional extras](#8-optional-extras)
9. [Privacy and legal](#9-privacy-and-legal)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Overview

`whatsapp-analyzer` is a **local-only** Python package for analysing WhatsApp group chat exports. It processes a `.zip` archive, a bare `_chat.txt` file, or a decompressed folder through a multi-step NLP pipeline:

| Step | What happens |
|------|-------------|
| **Load** | Detect format, decompress if needed |
| **Parse** | Extract messages into a structured DataFrame |
| **Clean** | Remove emoji, detect language, strip stopwords, lemmatise |
| **Analyse** | Topics (LDA / BERTopic), sentiment (VADER / CamemBERT), user profiles, temporal patterns, media stats |
| **Report** | Self-contained HTML with charts, word clouds, per-user breakdowns |

**Nothing ever leaves your machine.** All computation is local.

---

## 2. Installation

### Requirements

- Python ≥ 3.12
- `pip` or `uv`

### Base install

```bash
git clone https://github.com/MendasD/whatsapp-analyzer.git
cd whatsapp-analyzer

# with uv (recommended)
uv sync

# or with pip
pip install -e .
```

### spaCy language models

spaCy models are downloaded automatically on first use. To pre-install them manually:

```bash
python -m spacy download fr_core_news_sm   # French
python -m spacy download en_core_web_sm   # English
```

### Optional extras

Install only what you need:

```bash
pip install -e ".[bertopic]"   # BERTopic topic modelling (GPU-friendly)
pip install -e ".[camembert]"  # CamemBERT French sentiment (requires PyTorch)
pip install -e ".[media]"      # Whisper audio/video transcription
```

---

## 3. Quick Start

### 3.1 CLI

All commands are available through the `whatsapp-analyzer` entry point.

#### Analyse a single group

```bash
whatsapp-analyzer analyze \
  --input data-example/_chat.txt \
  --topics 5 \
  --output reports/
```

Expected output:

```
Analysing data-example/_chat.txt …
┌──────────────┬──────────────────────────────────────────┐
│ Metric       │ Value                                    │
├──────────────┼──────────────────────────────────────────┤
│ Group        │ _chat                                    │
│ Messages     │ 1 243                                    │
│ Participants │ 12                                       │
│ Period       │ 2024-01-01 → 2024-06-30                  │
│ Top topic    │ cours / td / examen / prof / notes       │
│ Top topic    │ sport / match / jouer / gagner / equipe  │
│ Top topic    │ sortie / soirée / vendredi / venir / ok  │
└──────────────┴──────────────────────────────────────────┘
Report written to reports/report.html
```

#### Compare multiple groups

```bash
whatsapp-analyzer compare \
  --input data-example/_chat.txt \
  --input data-example/_chat0.txt \
  --output reports/
```

Each `--input` flag adds one group. You can pass as many as needed.

#### Launch the web interface

```bash
whatsapp-analyzer serve
# Opens at http://localhost:8501
```

---

### 3.2 Python API

#### Single-group analysis — fluent style

```python
from whatsapp_analyzer import WhatsAppAnalyzer

az = WhatsAppAnalyzer("data-example/_chat.txt", n_topics=5)
az.parse().clean().analyze()

report_path = az.report(output="reports/")   # → reports/report.html
csv_path    = az.to_csv(output="reports/")   # → reports/_chat.csv

print(f"Report: {report_path}")
```

#### Single-group analysis — one-call shorthand

```python
from whatsapp_analyzer import WhatsAppAnalyzer

results = WhatsAppAnalyzer("data-example/_chat.txt").run()
# results["report_path"] contains the path to the HTML report
```

#### Per-user deep-dive

```python
from whatsapp_analyzer import WhatsAppAnalyzer

az = WhatsAppAnalyzer("data-example/_chat.txt")
az.parse().clean().analyze()

view = az.user("Alice")
print(view.summary())              # dict: message_count, sentiment_mean, …
print(view.topics())               # DataFrame of topic assignments
print(view.sentiment_over_time())  # DataFrame with timestamps and scores
print(view.activity_heatmap())     # 7×24 DataFrame (weekday × hour)
```

#### Comparing multiple groups

```python
from pathlib import Path
from whatsapp_analyzer import WhatsAppAnalyzer
from whatsapp_analyzer.comparator import GroupComparator

az1 = WhatsAppAnalyzer("data-example/_chat.txt",  n_topics=5)
az2 = WhatsAppAnalyzer("data-example/_chat0.txt", n_topics=5)
az1.parse().clean().analyze()
az2.parse().clean().analyze()

comp = GroupComparator([az1, az2])
print(comp.compare_activity())   # DataFrame: one row per group
print(comp.compare_topics())     # DataFrame: topic weights per group
print(comp.compare_sentiment())  # DataFrame: sentiment stats per group
print(comp.common_users())       # DataFrame: authors in multiple groups

report_path = comp.report(Path("reports/"))
# → reports/comparison_report.html
```

#### Selecting topics and analysis steps

```python
az = WhatsAppAnalyzer("data-example/_chat.txt")
az.parse().clean()

# Disable sentiment, enable media transcription
az.analyze(topics=True, sentiment=False, temporal=True, media=True)
```

#### Forcing a language

```python
az = WhatsAppAnalyzer("data-example/_chat.txt", lang="fr")
az.parse().clean().analyze()
```

#### Anonymising author names

```python
from whatsapp_analyzer.parser import Parser

df = Parser(anonymize=True).parse("data-example/_chat.txt")
# Author names are hashed — no plain-text names in the output
```

---

### 3.3 Web interface

```bash
whatsapp-analyzer serve
```

The Streamlit interface lets you:

- Upload a `.zip` or `.txt` export
- Configure the number of topics (2–15), minimum words per message, language
- Optionally anonymise author names
- Browse results in five tabs: Overview, Topics, Sentiment, Activity, Report
- Download the self-contained HTML report

---

## 4. Input formats

The package accepts three input types, detected automatically:

| Format | Example | Notes |
|--------|---------|-------|
| `.zip` | `WhatsApp Chat - Family.zip` | Native WhatsApp export, contains `_chat.txt` and media |
| `.txt` | `_chat.txt` | Chat text file alone, no media |
| Directory | `WhatsApp Chat - Family/` | Decompressed export folder |

### Platform formats

Two timestamp layouts are supported:

**Android:**
```
12/01/2024, 08:15 - Alice: Hello everyone!
12/01/2024, 08:15 - Alice: This is a
multiline message.
```

**iOS:**
```
[12/01/2024 à 08:15:00] Alice : Hello everyone!
[12/01/2024, 08:15:00] Alice : Another variant
```

Multiline messages are automatically merged.

### Message types

Each parsed message is classified as one of:

| Type | Description |
|------|-------------|
| `text` | Normal user message |
| `media` | Message containing an omitted media placeholder |
| `system` | WhatsApp system event (member added, group name changed, etc.) |

---

## 5. Pipeline architecture

```
Input (ZIP / TXT / folder)
         │
         ▼
    ┌─────────┐
    │ Loader  │  detect format, decompress ZIP → LoadedGroup
    └────┬────┘
         │ chat_path, media_dir, group_name
         ▼
    ┌─────────┐
    │ Parser  │  regex → DataFrame [timestamp, author, message, msg_type, group_name]
    └────┬────┘
         │
         ▼
    ┌─────────┐
    │ Cleaner │  emoji removal, language detection, stopwords, lemmatisation
    └────┬────┘  adds [cleaned_message, language, tokens]
         │
         ├──────────────────────────────────────────────────────┐
         ▼                                                      ▼
  ┌──────────────────┐   ┌───────────────────┐   ┌──────────────────────┐
  │ TopicClassifier  │   │ SentimentAnalyzer │   │  TemporalAnalyzer    │
  │ LDA / BERTopic   │   │ VADER / CamemBERT │   │  heatmaps, timelines │
  └────────┬─────────┘   └────────┬──────────┘   └──────────┬───────────┘
           │                      │                          │
           └──────────────────────┼──────────────────────────┘
                                  │
                                  ▼
                          ┌──────────────┐
                          │ UserAnalyzer │  per-user profiles
                          └──────┬───────┘
                                  │
                   ┌──────────────┼──────────────┐
                   ▼              ▼               ▼
             ┌──────────┐  ┌───────────┐  ┌────────────┐
             │Visualizer│  │  CLI      │  │  Web UI    │
             │HTML report│  │(click)   │  │(Streamlit) │
             └──────────┘  └───────────┘  └────────────┘
```

### Dependency rules

Modules only import from modules listed to their left in the pipeline. Heavy libraries (spaCy, transformers, sklearn, Whisper) are imported **lazily** inside method bodies, so startup is instant and optional dependencies raise `ImportError` only when the feature is actually called.

---

## 6. Module reference

### 6.1 Loader

```python
from whatsapp_analyzer.loader import Loader, LoadedGroup
```

#### `Loader`

**`Loader().load(path)`**

Load a single WhatsApp group from any supported input format.

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `str \| Path` | Path to a `.zip`, `_chat.txt`, or directory |

Returns: `LoadedGroup`

Raises: `FileNotFoundError` if no `_chat.txt` is found inside the path.  
Raises: `ValueError` if a `.zip` file is corrupted.

**`Loader().load_many(paths)`**

Load multiple groups, skipping any that fail with a warning.

| Parameter | Type | Description |
|-----------|------|-------------|
| `paths` | `list[str \| Path]` | One path per group |

Returns: `list[LoadedGroup]`

Raises: `RuntimeError` if **no** group could be loaded.

#### `LoadedGroup`

Attributes:

| Attribute | Type | Description |
|-----------|------|-------------|
| `chat_path` | `Path` | Path to the `_chat.txt` file |
| `media_dir` | `Path \| None` | Media folder, or `None` if absent |
| `group_name` | `str` | Human-readable group name |

**`loaded.cleanup()`** — Remove the temporary decompression directory if one was created (called automatically when using `WhatsAppAnalyzer`).

---

### 6.2 Parser

```python
from whatsapp_analyzer.parser import Parser
```

#### `Parser`

**Constructor parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `anonymize` | `bool` | `False` | Hash author names with SHA-256 |
| `group_name` | `str \| None` | `None` | Override the group name stored in the output |

**`Parser().parse(chat_path)`**

Parse a `_chat.txt` file into a DataFrame.

| Parameter | Type | Description |
|-----------|------|-------------|
| `chat_path` | `Path` | Path to the WhatsApp export text file |

Returns: `pd.DataFrame` with columns:

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | `datetime64[ns]` | Message timestamp |
| `author` | `str` | Author display name |
| `message` | `str` | Raw message text |
| `msg_type` | `str` | `'text'`, `'media'`, or `'system'` |
| `group_name` | `str` | Group name |

Raises: `ValueError` if no messages could be parsed.

---

### 6.3 Cleaner

```python
from whatsapp_analyzer.cleaner import Cleaner
```

#### `Cleaner`

**Constructor parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lang` | `str \| None` | `None` | Force language code (`'fr'`, `'en'`). `None` = auto-detect |
| `remove_emoji` | `bool` | `True` | Strip emoji characters |
| `min_words` | `int` | `3` | Drop cleaned messages shorter than this word count |
| `use_lemma` | `bool` | `True` | Apply spaCy lemmatisation when a model is available |

**`Cleaner().clean(df)`**

Apply the full NLP preprocessing pipeline to a parsed DataFrame.

| Parameter | Type | Description |
|-----------|------|-------------|
| `df` | `pd.DataFrame` | Output of `Parser.parse()` |

Returns: filtered `pd.DataFrame` with three additional columns:

| Column | Type | Description |
|--------|------|-------------|
| `cleaned_message` | `str` | Cleaned, lemmatised token string |
| `language` | `str` | Detected language code (e.g. `'fr'`) |
| `tokens` | `list[str]` | Tokenised list of cleaned words |

**Processing steps:**
1. Drop system and media messages (keep `msg_type == 'text'` only)
2. Normalise Unicode and strip control characters
3. Remove emoji (using the `emoji` library if installed, ASCII fallback otherwise)
4. Detect dominant language from the first 10 messages
5. Lowercase and strip punctuation
6. Remove stopwords (spaCy → NLTK → empty set, in priority order)
7. Lemmatise with spaCy when a matching model is available
8. Drop messages shorter than `min_words`

---

### 6.4 TopicClassifier

```python
from whatsapp_analyzer.topic_classifier import TopicClassifier
```

#### `TopicClassifier`

**Constructor parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_topics` | `int` | `5` | Number of topics to extract |
| `method` | `str` | `'lda'` | `'lda'` (always available) or `'bertopic'` (requires `[bertopic]` extra) |

**`TopicClassifier().fit_transform(df)`**

Fit a topic model and annotate each message with a topic.

| Parameter | Type | Description |
|-----------|------|-------------|
| `df` | `pd.DataFrame` | Output of `Cleaner.clean()`, must contain `cleaned_message` |

Returns: `dict` with keys:

| Key | Type | Description |
|-----|------|-------------|
| `df` | `pd.DataFrame` | Input DataFrame plus `topic_id` (int) and `topic_score` (float) |
| `group_topics` | `pd.DataFrame` | Columns: `topic_id`, `topic_label`, `weight` |

`topic_label` is a slash-separated string of the five most representative words, e.g. `"cours / td / examen / prof / notes"`.

Raises: `ValueError` if `df` is empty or missing `cleaned_message`.  
Raises: `RuntimeError` if `method='bertopic'` and BERTopic is not installed.

---

### 6.5 SentimentAnalyzer

```python
from whatsapp_analyzer.sentiment_analyzer import SentimentAnalyzer
```

#### `SentimentAnalyzer`

**Constructor parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | `str` | `'vader'` | `'vader'` (always available) or `'camembert'` (requires `[camembert]` extra) |

**`SentimentAnalyzer().analyze(df)`**

Score each message and aggregate statistics.

| Parameter | Type | Description |
|-----------|------|-------------|
| `df` | `pd.DataFrame` | Output of `Cleaner.clean()`, must contain `cleaned_message` |

Returns: `dict` with keys:

| Key | Type | Description |
|-----|------|-------------|
| `df` | `pd.DataFrame` | Input DataFrame plus `sentiment_score` (float, −1 to 1) and `sentiment_label` (str) |
| `by_user` | `pd.DataFrame` | Columns `author` and `sentiment_score` (mean per user) |
| `global` | `dict` | Keys: `mean` (float), `pos_pct` (float 0–1), `neg_pct` (float 0–1) |

Sentiment labels: `'positive'` (score ≥ 0.05), `'negative'` (score ≤ −0.05), `'neutral'` otherwise.

Raises: `ValueError` if `df` is empty or missing `cleaned_message`.  
Raises: `RuntimeError` if `method='camembert'` and the library is not installed.

**Backends:**

- **VADER** — rule-based, language-agnostic, fast, no GPU required.
- **CamemBERT** (`cmarkea/distilcamembert-base-sentiment`) — transformer model fine-tuned for French sentiment, returns 1–5 star ratings mapped to [−1, 1].

---

### 6.6 TemporalAnalyzer

```python
from whatsapp_analyzer.temporal_analyzer import TemporalAnalyzer
```

#### `TemporalAnalyzer`

No constructor parameters.

**`TemporalAnalyzer().analyze(df)`**

Compute temporal activity metrics from the cleaned DataFrame.

| Parameter | Type | Description |
|-----------|------|-------------|
| `df` | `pd.DataFrame` | Must contain a `timestamp` column (datetime64[ns]) |

Returns: `dict` with keys:

| Key | Type | Description |
|-----|------|-------------|
| `timeline` | `pd.DataFrame` | Message count per day; datetime index, column `count` |
| `hourly_heatmap` | `pd.DataFrame` | 7 rows (Mon–Sun) × 24 columns (0–23), values are message counts |
| `weekly_activity` | `pd.Series` | Message count per day of week (Monday → Sunday) |
| `monthly_activity` | `pd.Series` | Message count per calendar month (Period index) |
| `peak_hour` | `int` | Overall busiest hour (0–23) |
| `peak_day` | `str` | Overall busiest day of week (e.g. `"Friday"`) |

Raises: `ValueError` if `df` is empty or missing `timestamp`.

---

### 6.7 UserAnalyzer

```python
from whatsapp_analyzer.user_analyzer import UserAnalyzer
```

#### `UserAnalyzer`

No constructor parameters.

**`UserAnalyzer().build_profiles(results)`**

Build a profile dict for every author in the pipeline results.

| Parameter | Type | Description |
|-----------|------|-------------|
| `results` | `dict` | Pipeline results with keys `df_clean`, `topics`, `sentiment` |

Returns: `dict[str, dict]` — mapping of author name → profile dict.

Each profile contains:

| Key | Type | Description |
|-----|------|-------------|
| `message_count` | `int` | Total messages sent |
| `avg_message_length` | `float` | Average token count per message |
| `activity_hours` | `list[int]` | Top 3 most active hours |
| `most_active_day` | `str` | Most active day of week |
| `top_topics` | `list[str]` | Up to 3 most frequent topic labels |
| `sentiment_mean` | `float \| None` | Mean sentiment score, or `None` if not run |

**Static methods:**

**`UserAnalyzer.summary_for(author, results)`** — Return the profile dict for a single author (empty dict if not found).

**`UserAnalyzer.topics_for(author, results)`** — Return a DataFrame of topic assignments for a single author.

**`UserAnalyzer.sentiment_over_time_for(author, results)`** — Return a DataFrame with `timestamp` and `sentiment_score` columns for a single author.

**`UserAnalyzer.activity_heatmap_for(author, results)`** — Return a 7×24 activity heatmap DataFrame for a single author.

---

### 6.8 MediaAnalyzer

```python
from whatsapp_analyzer.media_analyzer import MediaAnalyzer
```

#### `MediaAnalyzer`

No constructor parameters.

**`MediaAnalyzer().analyze(media_dir)`**

Scan a media directory, compute file statistics, and optionally transcribe audio/video.

| Parameter | Type | Description |
|-----------|------|-------------|
| `media_dir` | `Path` | Directory containing WhatsApp media files |

Returns: `dict` with keys:

| Key | Type | Description |
|-----|------|-------------|
| `stats` | `pd.DataFrame` | Columns: `file_type`, `count`, `total_size_mb` |
| `transcriptions` | `pd.DataFrame` | Columns: `file_path`, `text` (empty if Whisper is not installed) |

**Supported extensions:** `.jpg`, `.jpeg`, `.png`, `.webp`, `.mp4`, `.opus`, `.ogg`, `.mp3`

Transcription requires the `[media]` extra (`pip install -e ".[media]"`).

---

### 6.9 GroupComparator

```python
from whatsapp_analyzer.comparator import GroupComparator
```

#### `GroupComparator`

**Constructor parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `analyzers` | `list[WhatsAppAnalyzer]` | Analyzed groups to compare |

**`GroupComparator().compare_activity()`**

Return one activity-summary row per group.

Returns: `pd.DataFrame` with group names as index and columns:
`nb_messages`, `nb_participants`, `msgs_per_day`, `period_start`, `period_end`.

**`GroupComparator().compare_topics()`**

Return a pivot table of topic weights across groups.

Returns: `pd.DataFrame` with group names as index, topic labels as columns, values are weights.

**`GroupComparator().compare_sentiment()`**

Return one sentiment-summary row per group.

Returns: `pd.DataFrame` with group names as index and columns:
`sentiment_mean`, `pos_pct`, `neg_pct`.

**`GroupComparator().common_users()`**

Return authors present in more than one group.

Returns: `pd.DataFrame` with columns `author` and `groups` (list of group names).

**`GroupComparator().report(output)`**

Generate a multi-group comparison HTML report.

| Parameter | Type | Description |
|-----------|------|-------------|
| `output` | `Path` | Output directory |

Returns: `Path` to `comparison_report.html`.

---

### 6.10 Visualizer

```python
from whatsapp_analyzer.visualizer import Visualizer
```

#### `Visualizer`

No constructor parameters.

All `plot_*` methods return a `matplotlib.figure.Figure` and do **not** write files. Pass the figure to `st.pyplot()` (Streamlit) or save it with `fig.savefig()`.

**`Visualizer().plot_topic_distribution(results)`** — Bar chart of topic weights.

**`Visualizer().plot_wordcloud(results, topic_id)`** — Word cloud for a specific topic.

| Parameter | Type | Description |
|-----------|------|-------------|
| `results` | `dict` | Pipeline results |
| `topic_id` | `int` | Topic index to visualise |

**`Visualizer().plot_sentiment_timeline(results)`** — Rolling mean sentiment over time.

**`Visualizer().plot_user_activity(results)`** — Horizontal bar chart of messages per user.

**`Visualizer().plot_hourly_heatmap(results)`** — Seaborn heatmap (weekday × hour).

**`Visualizer().generate_report(results, output_dir)`**

Write a self-contained HTML report.

| Parameter | Type | Description |
|-----------|------|-------------|
| `results` | `dict` | Pipeline results dict |
| `output_dir` | `Path` | Directory where `report.html` will be written |

Returns: `Path` to `report.html`. Creates the directory if it does not exist.

All images are embedded as base64 PNG — the file opens offline with no CDN.

**`Visualizer().generate_comparison_report(comparison_data, output_dir)`**

Write a multi-group comparison HTML report.

Returns: `Path` to `comparison_report.html`.

---

### 6.11 WhatsAppAnalyzer (core)

```python
from whatsapp_analyzer import WhatsAppAnalyzer
```

The main orchestrator. All sub-module imports are lazy — no NLP code runs at construction time.

**Constructor parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_path` | `str \| Path` | — | Path to `.zip`, `_chat.txt`, or folder |
| `n_topics` | `int` | `5` | Number of LDA topics |
| `lang` | `str \| None` | `None` | Force language (`'fr'`, `'en'`). `None` = auto-detect |
| `min_words` | `int` | `3` | Minimum token count after cleaning |
| `output_dir` | `str \| Path` | `'reports'` | Default output directory |

**Fluent methods — must be called in order:**

| Method | Description | Returns |
|--------|-------------|---------|
| `.parse()` | Load + parse the export | `self` |
| `.clean(lang=None, min_words=None)` | Apply NLP preprocessing | `self` |
| `.analyze(topics=True, sentiment=True, temporal=True, media=False)` | Run all analysis modules | `self` |
| `.report(output=None)` | Generate HTML report | `Path` |
| `.to_csv(output=None)` | Export enriched DataFrame as CSV | `Path` |

**Other methods:**

| Method | Description | Returns |
|--------|-------------|---------|
| `.run()` | Convenience: parse → clean → analyze → report | `dict` (results) |
| `.user(author)` | Get a `UserView` for a specific author | `UserView` |

**Fault tolerance:** each analysis step (topics, sentiment, temporal, media, users) is wrapped in a try/except. A failing step logs a warning and sets its result key to `None` — the rest of the pipeline continues.

---

### 6.12 UserView

Returned by `WhatsAppAnalyzer.user(author)`. Scoped view for a single author.

| Method | Returns | Description |
|--------|---------|-------------|
| `.summary()` | `dict` | Full profile dict for this author |
| `.topics()` | `pd.DataFrame` | Topic assignment DataFrame |
| `.sentiment_over_time()` | `pd.DataFrame` | Sentiment scores with timestamps |
| `.activity_heatmap()` | `pd.DataFrame` | 7×24 activity heatmap |

---

### 6.13 CLI commands

The entry point is `whatsapp-analyzer` (installed with the package).

#### `analyze`

```
whatsapp-analyzer analyze [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--input PATH` | required | — | Path to `.zip`, `_chat.txt`, or folder |
| `--topics INT` | optional | `5` | Number of LDA topics |
| `--output DIR` | optional | `reports` | Output directory for `report.html` |

#### `compare`

```
whatsapp-analyzer compare [OPTIONS]
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--input PATH` | repeatable | — | One path per group (pass flag multiple times) |
| `--output DIR` | optional | `reports` | Output directory for `comparison_report.html` |

#### `serve`

```
whatsapp-analyzer serve
```

Launches the Streamlit web interface at `http://localhost:8501`. No options.

---

## 7. Output reference

### `results` dict keys

The `results` dict returned by `WhatsAppAnalyzer.run()` or populated via the fluent API:

| Key | Type | Populated by |
|-----|------|-------------|
| `group_name` | `str` | `parse()` |
| `df_raw` | `pd.DataFrame` | `parse()` |
| `df_clean` | `pd.DataFrame` | `clean()` |
| `topics` | `dict \| None` | `analyze(topics=True)` |
| `sentiment` | `dict \| None` | `analyze(sentiment=True)` |
| `temporal` | `dict \| None` | `analyze(temporal=True)` |
| `media` | `dict \| None` | `analyze(media=True)` |
| `users` | `dict[str, dict] \| None` | `analyze()` (always) |
| `report_path` | `Path` | `run()` only |

### DataFrame columns summary

**After `parse()`:**
`timestamp`, `author`, `message`, `msg_type`, `group_name`

**After `clean()`:**
all of the above + `cleaned_message`, `language`, `tokens`

**After `analyze(topics=True)`:**
all of the above + `topic_id`, `topic_score`

**After `analyze(sentiment=True)`:**
all of the above + `sentiment_score`, `sentiment_label`

---

## 8. Optional extras

| Extra | Package(s) installed | Feature unlocked |
|-------|---------------------|-----------------|
| `[bertopic]` | `bertopic`, `sentence-transformers` | BERTopic topic modelling |
| `[camembert]` | `transformers`, `torch` | CamemBERT French sentiment |
| `[media]` | `openai-whisper` | Audio/video transcription |

Install multiple extras at once:

```bash
pip install -e ".[bertopic,camembert,media]"
```

---

## 9. Privacy and legal

- **100% local.** No message content is sent to any external server.
- **No WhatsApp automation.** Exports are generated manually by the user from within the WhatsApp app. This package does not automate the WhatsApp interface.
- **Anonymisation** is available via `Parser(anonymize=True)` or the `anonymise` checkbox in the web UI. Author names are replaced with SHA-256 hashes.
- **Data storage.** Real chat exports should be placed in `data/raw/` (gitignored). Never commit real WhatsApp data.
- **Intended use.** Personal or academic analysis of conversations you participate in. Raw chat data must not be redistributed.
- **Terms of Service.** Compliant with WhatsApp (Meta) Terms of Service.

---

## 10. Troubleshooting

### spaCy model not found

```
OSError: [E050] Can't find model 'fr_core_news_sm'.
```

**Fix:** The model is downloaded automatically on first run. If that fails:
```bash
python -m spacy download fr_core_news_sm
python -m spacy download en_core_web_sm
```

### NLTK `vader_lexicon` not found

The lexicon is downloaded automatically the first time `SentimentAnalyzer` is called with `method='vader'`. If you are offline:

```python
import nltk
nltk.download("vader_lexicon")
nltk.download("stopwords")
```

### BERTopic or CamemBERT not installed

```
RuntimeError: CamemBERT requires the [camembert] extra.
```

**Fix:**
```bash
pip install -e ".[camembert]"
```

### `No messages could be parsed`

The file is not a valid WhatsApp export, or it uses an unsupported date format. Verify that the file was exported directly from WhatsApp and that it contains lines matching the Android or iOS format described in [section 4](#4-input-formats).

### Report is empty / missing charts

One or more analysis steps may have failed silently. Check the log output (set `logging.basicConfig(level=logging.WARNING)`) to see which steps were skipped and why.
