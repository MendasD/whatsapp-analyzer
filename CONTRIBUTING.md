# Contributing — whatsapp-analyzer

This document explains the project structure, the role of every file,
and the workflow all contributors must follow.
Read it entirely before opening your first PR.

---

## Project structure

```
whatsapp-analyzer/
│
├── whatsapp_analyzer/        # Main package (pip-installable)
│   ├── __init__.py             # Public API: exposes WhatsAppAnalyzer, GroupComparator
│   ├── core.py                 # WhatsAppAnalyzer class — orchestrates the full pipeline
│   ├── loader.py               # ✅ Detects input format (.zip/.txt/dir), decompresses
│   ├── parser.py               # ✅ Regex parsing of _chat.txt → DataFrame
│   ├── cleaner.py              # ✅ NLP preprocessing: stopwords, lemmatisation
│   ├── topic_classifier.py     # Topic modelling: LDA (default) or BERTopic
│   ├── sentiment_analyzer.py   # Sentiment scoring: VADER (EN/fallback) / CamemBERT (FR)
│   ├── user_analyzer.py        # Per-user profiles aggregated from analysis results
│   ├── temporal_analyzer.py    # Activity timelines and heatmaps
│   ├── media_analyzer.py       # Optional media analysis (Whisper audio, image stats)
│   ├── comparator.py           # Multi-group comparison
│   ├── visualizer.py           # Charts, wordclouds, self-contained HTML report
│   ├── cli.py                  # Click CLI: commands analyze / compare / serve
│   ├── app.py                  # Streamlit web interface
│   └── utils.py                # ✅ Shared helpers (logging, anonymisation, encoding)
│
├── tests/                      # One test file per module — must run in isolation
│   ├── conftest.py             # Shared fixtures (fake WhatsApp exports as strings)
│   ├── test_utils.py           # ✅ 21 tests
│   ├── test_loader.py          # ✅ 9 tests
│   ├── test_parser.py          # ✅ 14 tests
│   ├── test_cleaner.py         # ✅ 20 tests
│   └── test_*.py               # One file per remaining module (to be created)
│
├── data/
│   ├── raw/                    # Real WhatsApp exports — NEVER committed to git
│   └── processed/              # Generated CSVs / SQLite — NEVER committed to git
│
├── notebooks/
│   ├── 01_exploration.ipynb              # Full pipeline walkthrough
│   ├── 02_analyse_utilisateurs.ipynb     # Per-user analysis
│   └── 03_comparaison_groupes.ipynb      # Multi-group comparison
│
├── docs/
│   └── data_exploration.md     # Corpus statistics from real export validation
│
├── .github/
│   └── PULL_REQUEST_TEMPLATE.md
│
├── pyproject.toml              # Package config, dependencies, linting settings
├── CONTRIBUTING.md             # This file
├── ISSUES.md                   # GitHub issues ready to copy-paste
└── README.md                   # Quick-start guide for users
```

✅ = already implemented and tested.

---

## How the pipeline works

Every module in the pipeline receives the output of the previous one.
No module skips steps or talks directly to a module it does not depend on.

```
Input (.zip / .txt / dir / list of paths)
        │
        ▼
    loader.py
        Detects format, decompresses ZIP, returns LoadedGroup
        (chat_path, media_dir, group_name)
        │
        ▼
    parser.py
        Reads _chat.txt with regex, returns DataFrame
        columns: timestamp | author | message | msg_type | group_name
        │
        ▼
    cleaner.py
        Filters to text messages, normalises, removes stopwords, lemmatises
        adds columns: cleaned_message | language | tokens
        │
        ├──▶ topic_classifier.py
        │       adds: topic_id | topic_label | topic_score
        │
        ├──▶ sentiment_analyzer.py
        │       adds: sentiment_score | sentiment_label
        │
        ├──▶ user_analyzer.py
        │       builds: dict[author → profile]
        │
        ├──▶ temporal_analyzer.py
        │       builds: timeline, heatmap, peak_hour, peak_day
        │
        └──▶ media_analyzer.py  (optional, --media flag)
                builds: stats DataFrame, transcriptions DataFrame
                        │
                        ▼
                  comparator.py
                        Merges results from N WhatsAppAnalyzer instances
                        │
                        ▼
                  visualizer.py
                        Generates charts and self-contained HTML report
                        │
                  ┌─────┴─────┐
                cli.py      app.py
                Click CLI   Streamlit web UI
```

`core.py` owns the `WhatsAppAnalyzer` class that chains these steps.
`utils.py` provides helpers used across all modules — it must never
import from any other module in the package to avoid circular dependencies.

---

## Dependency rules between modules

| Module | May import from |
|---|---|
| `utils.py` | stdlib + third-party only |
| `loader.py` | `utils` |
| `parser.py` | `utils` |
| `cleaner.py` | `utils` |
| `topic_classifier.py` | `utils` |
| `sentiment_analyzer.py` | `utils` |
| `user_analyzer.py` | `utils` |
| `temporal_analyzer.py` | `utils` |
| `media_analyzer.py` | `utils` |
| `comparator.py` | `utils` · `core` (type hint only) |
| `visualizer.py` | `utils` |
| `core.py` | all modules (lazy imports inside methods) |
| `cli.py` | `core` · `comparator` · `utils` |
| `app.py` | `core` · `comparator` · `utils` |

Circular imports are a common mistake — always follow this table.

---

## Coding conventions

These rules are enforced by `ruff` and `black`.
A PR that violates them will not be merged.

**Language** — all code, comments, docstrings, and variable names are in **English**.

**Comments** — use a single `#`. Never use decorative separators like `#---` or `#===`.

**Docstrings** — add them to every public class and every non-trivial function.
Use Google style:

```python
def analyze(self, df: pd.DataFrame) -> dict:
    """
    Analyse the cleaned DataFrame.

    Args:
        df: Output of Cleaner.clean().

    Returns:
        Dict with keys 'df', 'by_user', and 'global'.
    """
```

**Type hints** — required on every public function signature.

**Logging** — use `logging.getLogger(__name__)`, never `print()`.

**Imports** — three groups separated by a blank line: stdlib → third-party → local.

**Naming** — `snake_case` for functions and variables, `PascalCase` for classes.

**No dead code** — no commented-out blocks, no leftover `TODO` in production code.

---

## GitHub workflow (GitHub Flow)

We use one branch per issue, all merged into `main` via Pull Request.

### Step 1 — Always start from a fresh main

```bash
git checkout main
git pull origin main
```

### Step 2 — Create your branch

Name your branch after your issue number and a short description:

```
feature/<issue-number>-short-description
```

Examples:
```bash
git checkout -b feature/04-temporal-analyzer
git checkout -b feature/08-cli
git checkout -b feature/11-readme-setup
```

### Step 3 — Commit regularly with clear messages

Format: `<type>(<scope>): <what you did>`

```bash
git commit -m "feat(temporal_analyzer): add hourly heatmap computation"
git commit -m "test(temporal_analyzer): add isolation tests for peak_hour"
git commit -m "fix(parser): handle iOS timestamp with comma separator"
git commit -m "docs(contributing): update module status table"
```

Allowed types: `feat` · `fix` · `test` · `docs` · `refactor` · `chore`

### Step 4 — Push and open a Pull Request

```bash
git push origin feature/04-temporal-analyzer
```

Then open a PR on GitHub against `main`.
The PR template will appear automatically — fill every checkbox.

### Step 5 — Request a review

Assign at least **one other contributor** as reviewer.
Do not merge your own PR without a review.

### Step 6 — Clean up after merge

```bash
git checkout main
git pull origin main
git branch -d feature/04-temporal-analyzer
```

---

## Writing tests — isolation rules

Every test file must pass **completely alone**, without executing any other
module from the package (beyond `utils.py`).

```bash
# This must work independently — no other test files needed
python -m pytest tests/test_temporal_analyzer.py -v
```

### Rules

1. **Mock all heavy external dependencies** — spaCy, NLTK, langdetect, Whisper,
   transformers, sklearn, matplotlib. Use `unittest.mock.patch`.

2. **Never call real NLP models or load real files** in tests.
   Tests must run in under 2 seconds and require no downloads.

3. **Build test input data locally** using helper functions defined at the top
   of each test file. Never import `Parser` or `Cleaner` to produce test data
   for another module's tests.

4. **One clear assertion per test** — keep tests focused and readable.

5. **Name tests descriptively**:
   ```python
   def test_peak_hour_returns_integer(): ...
   def test_empty_dataframe_returns_empty_timeline(): ...
   def test_sentiment_falls_back_to_vader_when_camembert_missing(): ...
   ```

### Example — building isolated test data

```python
# Inside tests/test_temporal_analyzer.py

import pandas as pd

def _make_df(timestamps: list[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": pd.to_datetime(timestamps),
        "author": ["Aminata"] * len(timestamps),
        "message": ["test message"] * len(timestamps),
        "msg_type": ["text"] * len(timestamps),
        "group_name": ["TestGroup"] * len(timestamps),
        "cleaned_message": ["test message"] * len(timestamps),
    })

def test_peak_hour_returns_integer():
    df = _make_df(["2024-01-12 08:00", "2024-01-12 08:30", "2024-01-12 10:00"])
    result = TemporalAnalyzer().analyze(df)
    assert isinstance(result["peak_hour"], int)
```

---

## Setting up locally

```bash
# Clone and install in editable mode with dev dependencies
git clone https://github.com/<your-org>/whatsapp-classifier.git
cd whatsapp-classifier
pip install -e ".[dev]"

# Optional: NLP models (needed for cleaner.py with use_lemma=True)
python -m spacy download fr_core_news_sm
python -m spacy download en_core_web_sm
python -m nltk.downloader stopwords

# Optional: media analysis extras
pip install ".[media]"

# Run the full test suite
python -m pytest

# Run a single module's tests
python -m pytest tests/test_parser.py -v
```

---

## Current implementation status

| Module | Status | Tests |
|---|---|---|
| `utils.py` | ✅ Done | 21 passing |
| `loader.py` | ✅ Done | 9 passing |
| `parser.py` | ✅ Done | 14 passing |
| `cleaner.py` | ✅ Done | 20 passing |
| `topic_classifier.py` | 🔲 Issue #02a · #02b | — |
| `sentiment_analyzer.py` | 🔲 Issue #03a · #03b | — |
| `user_analyzer.py` | 🔲 Issue #05 | — |
| `temporal_analyzer.py` | 🔲 Issue #04 | — |
| `media_analyzer.py` | 🔲 Issue #06 | — |
| `comparator.py` | 🔲 Issue #12 | — |
| `visualizer.py` | 🔲 Issue #07 | — |
| `core.py` | 🔲 Issue #01 | — |
| `cli.py` | 🔲 Issue #08 | — |
| `app.py` | 🔲 Issue #09 | — |
| Notebooks | 🔲 Issue #10 | — |
| README + repo setup | 🔲 Issue #11 | — |
| Data exploration | 🔲 Issue #13 | — |

---

## Questions and communication

- Open a **GitHub Discussion** for design questions.
- Open a **GitHub Issue** with the `bug` label for any bug found during development.
- For urgent questions, use the project group chat.