# GitHub Issues — whatsapp-analyzer
# Sprint v0.1.0 — 3 days

## Before creating issues

Create these labels on GitHub:

| Label | Description |
|---|---|
| `module` | Core module implementation |
| `tests` | Test coverage |
| `docs` | Documentation |
| `exploration` | Data exploration |
| `cli` | Command-line interface |
| `ui` | Streamlit web interface |
| `nlp` | NLP / machine learning |
| `integration` | Cross-module wiring |
| `good first issue` | Entry point for newcomers |
| `bug` | Something is broken |

Create milestone: **v0.1.0** — deadline: 3 days from today.

---

## Sprint overview

```
Day 1  →  All independent modules (no inter-issue dependencies)
Day 2  →  Assembly modules (depend on Day 1 merges)
Day 3  →  core.py wiring + integration + release
```

Assign issues to contributors on GitHub after creation.
Suggested distribution:

| | Day 1 | Day 2 | Day 3 |
|---|---|---|---|
| Dev A (advanced NLP) | #01 · #05 | #09 · #13 | #16 (pair) |
| Dev B (intermediate) | #02 · #06 | #10 · #14 | #16 (pair) |
| Dev C (intermediate) | #03 · #07 | #11 · #15 | #17 (pair) |
| Dev D (any level) | #04 · #08 | #12 | #17 (pair) |

---
---

## DAY 1 — Independent modules

---

## ISSUE #01 — Implement `topic_classifier.py` (LDA)

**Labels**: `module` · `nlp`
**Day**: 1
**Branch**: `feature/01-topic-classifier-lda`
**Difficulty**: Advanced

### Context

Receives the cleaned DataFrame from `cleaner.py` and assigns a topic
to each message using Latent Dirichlet Allocation (LDA) from scikit-learn.
This is one of the two core NLP modules of the package.

### What to implement

File: `whatsapp_classifier/topic_classifier.py`

Class `TopicClassifier(n_topics: int = 5, method: str = "lda")`:

- `fit_transform(df: pd.DataFrame) -> dict`
  - Vectorise `cleaned_message` with `TfidfVectorizer`.
  - Fit `LatentDirichletAllocation` on the resulting matrix.
  - Add `topic_id` (int) and `topic_score` (float) columns to the DataFrame.
  - Build `group_topics` summary: columns `topic_id`, `topic_label`
    (top-5 words joined by ` / `), `weight`.
  - Return `{"df": enriched_df, "group_topics": summary_df}`.

### Rules

- LDA only in this issue — BERTopic is covered by #09.
- Only `scikit-learn` is required (always installed).
- All external deps mocked in tests — no model fitting in tests.
- Follow all conventions in `CONTRIBUTING.md`.

### Definition of done

- [ ] `topic_classifier.py` implemented with LDA path.
- [ ] `tests/test_topic_classifier.py` written — sklearn mocked, runs alone.
- [ ] `python -m pytest tests/test_topic_classifier.py` passes.

---

## ISSUE #02 — Implement `sentiment_analyzer.py` (VADER)

**Labels**: `module` · `nlp`
**Day**: 1
**Branch**: `feature/02-sentiment-vader`
**Difficulty**: Advanced

### Context

Scores the sentiment of each cleaned message. This issue implements
the VADER engine only (always installed). CamemBERT is covered by #06.

### What to implement

File: `whatsapp_classifier/sentiment_analyzer.py`

Class `SentimentAnalyzer(lang: str = "fr")`:

- `analyze(df: pd.DataFrame) -> dict`
  - Score each row with `vaderSentiment.SentimentIntensityAnalyzer`.
  - Add `sentiment_score` (float, -1 to 1) and `sentiment_label`
    (`"positive"` / `"neutral"` / `"negative"`).
  - Thresholds: score > 0.05 → positive, < -0.05 → negative, else neutral.
  - Return:
    - `"df"` — enriched DataFrame.
    - `"by_user"` — mean sentiment per author (DataFrame).
    - `"global"` — dict: `mean`, `pos_pct`, `neg_pct`.

### Rules

- VADER only — no `transformers` dependency in this issue.
- All external deps mocked in tests.
- Follow all conventions in `CONTRIBUTING.md`.

### Definition of done

- [ ] `sentiment_analyzer.py` implemented with VADER path.
- [ ] `tests/test_sentiment_analyzer.py` written — VADER mocked, runs alone.
- [ ] `python -m pytest tests/test_sentiment_analyzer.py` passes.

---

## ISSUE #03 — Implement `temporal_analyzer.py`

**Labels**: `module`
**Day**: 1
**Branch**: `feature/03-temporal-analyzer`
**Difficulty**: Intermediate

### Context

Analyses when participants are active — by hour, day, week, month.
Pure pandas, no NLP dependency. Straightforward and fully independent.

### What to implement

File: `whatsapp_classifier/temporal_analyzer.py`

Class `TemporalAnalyzer`:

- `analyze(df: pd.DataFrame) -> dict`
  - `"timeline"` — DataFrame: message count indexed by date.
  - `"hourly_heatmap"` — DataFrame 7×24: weekday × hour, values = message count.
  - `"weekly_activity"` — Series: count per day of week (Monday … Sunday).
  - `"monthly_activity"` — Series: count per calendar month.
  - `"peak_hour"` — int (0–23), overall busiest hour.
  - `"peak_day"` — str, overall busiest day of week.

### Rules

- Input DataFrame has `timestamp` typed as `datetime64[ns]`.
- `pandas` only — zero NLP or external libraries.
- Build test DataFrames locally in the test file — do not import other modules.
- Follow all conventions in `CONTRIBUTING.md`.

### Definition of done

- [ ] `temporal_analyzer.py` implemented.
- [ ] `tests/test_temporal_analyzer.py` builds its own DataFrame — runs alone.
- [ ] `python -m pytest tests/test_temporal_analyzer.py` passes.

---

## ISSUE #04 — Set up the GitHub repository and write `README.md`

**Labels**: `docs` · `good first issue`
**Day**: 1
**Branch**: `feature/04-repo-setup`
**Difficulty**: Beginner-friendly

### Context

First step that unblocks the whole team: a properly configured repository
means everyone can follow the same workflow from day one.

### What to do

**GitHub repository setup:**
- Create all labels listed at the top of this file.
- Create the milestone **v0.1.0** with the correct deadline.
- Enable branch protection on `main`: require at least 1 review before merge.
- Confirm all four contributors have Write access.
- Push `.github/PULL_REQUEST_TEMPLATE.md` (already generated in the repo).

**`README.md`** — write or finalise:
- 2–3 sentence project description (mention the course context).
- Installation instructions: `pip install -e ".[dev]"`, spaCy models, extras.
- Quick-start examples: API Python, CLI, Streamlit.
- Module status table (mark Done / In progress / Planned).
- Legal / RGPD note (local processing, no data sent to servers).
- Link to `CONTRIBUTING.md`.

### Rules

- Do not commit any real chat data.
- Follow all conventions in `CONTRIBUTING.md`.

### Definition of done

- [ ] All labels and milestone created on GitHub.
- [ ] Branch protection active on `main`.
- [ ] `README.md` complete and pushed.
- [ ] PR template visible when opening a new PR.

---

## ISSUE #05 — Implement `user_analyzer.py`

**Labels**: `module`
**Day**: 1
**Branch**: `feature/05-user-analyzer`
**Difficulty**: Intermediate

### Context

Builds a per-user profile from the results dict produced by `core.py`.
Powers `az.user("Name").summary()` in the public API.
No NLP logic — pure aggregation over the results dict.

### What to implement

File: `whatsapp_classifier/user_analyzer.py`

Class `UserAnalyzer`:

- `build_profiles(results: dict) -> dict[str, dict]`
  Reads `results["df_clean"]`, `results["topics"]`, `results["sentiment"]`.
  Returns one profile dict per author:
  - `message_count` (int)
  - `avg_message_length` (float, average word count)
  - `activity_hours` (dict: hour → count)
  - `most_active_day` (str, day of week)
  - `top_topics` (list of `(label, score)` tuples)
  - `sentiment_mean` (float or None if sentiment not run)

Static methods called by `UserView` in `core.py`:
  - `summary_for(author: str, results: dict) -> dict`
  - `topics_for(author: str, results: dict) -> pd.DataFrame`
  - `sentiment_over_time_for(author: str, results: dict) -> pd.DataFrame`
  - `activity_heatmap_for(author: str, results: dict) -> pd.DataFrame`

### Rules

- Must not import `topic_classifier`, `sentiment_analyzer`, or `cleaner`.
- Reads exclusively from the `results` dict.
- Build a fake `results` dict in the test file — do not run the pipeline.
- Follow all conventions in `CONTRIBUTING.md`.

### Definition of done

- [ ] `user_analyzer.py` implemented.
- [ ] `tests/test_user_analyzer.py` builds a fake `results` dict locally — runs alone.
- [ ] `python -m pytest tests/test_user_analyzer.py` passes.

---

## ISSUE #06 — Extend `sentiment_analyzer.py` with CamemBERT

**Labels**: `module` · `nlp`
**Day**: 1 (start after #02 is merged, or mock the base class)
**Branch**: `feature/06-sentiment-camembert`
**Difficulty**: Advanced
**Note**: Can be developed in parallel with #02 — add the method to a local copy,
then merge cleanly once #02 is on `main`.

### Context

CamemBERT is a French-language model from HuggingFace that significantly
improves sentiment quality over VADER on French text.
It is optional — falls back to VADER when not installed.

### What to implement

Add to `whatsapp_classifier/sentiment_analyzer.py`:

- Private method `_score_camembert(texts: list[str]) -> list[float]`.
- Route to CamemBERT when `lang == "fr"` and `transformers` is available.
- Guard with `try/except ImportError` — fall back to VADER with a log warning.
- Load the model lazily (inside the method, not at import time).
- Output format identical to the VADER path.

### Rules

- The VADER path must remain unchanged after this change.
- CamemBERT never loaded in tests — mock `transformers` entirely.
- Follow all conventions in `CONTRIBUTING.md`.

### Definition of done

- [ ] CamemBERT path implemented and guarded.
- [ ] Existing VADER tests (#02) still pass.
- [ ] New tests for the CamemBERT path — transformers mocked.

---

## ISSUE #07 — Data exploration and parser validation

**Labels**: `exploration`
**Day**: 1
**Branch**: `feature/07-data-exploration`
**Difficulty**: Intermediate

### Context

Validates the parser on real export files before the rest of the team
builds on top of it. Any bugs found here feed directly back into `parser.py`.

### What to do

1. Export 2–3 WhatsApp group chats from real devices (Android + iOS).
2. Anonymise immediately: replace phone numbers and names with pseudonyms.
   Do not commit any raw export files.
3. Run `Parser().parse(chat_path)` on each file.
4. Document any lines not parsed correctly (timestamp variants, encoding issues,
   Wolof / Arabic characters, platform-specific system messages).
5. If parser bugs are found, open a new issue with the `bug` label and fix
   `parser.py` in that branch.
6. Compute and record basic corpus statistics:
   - message count, author count, date range per group.
   - Distribution of `msg_type` (text / media / system).
   - Estimated language distribution.
7. Write findings in `docs/data_exploration.md`.

### Rules

- **No real personal data committed** — no phone numbers, no real names.
- Follow all conventions in `CONTRIBUTING.md`.

### Definition of done

- [ ] `docs/data_exploration.md` written with statistics for ≥ 2 groups.
- [ ] Parser validated on both Android and iOS formats.
- [ ] Any bugs found are reported as separate issues.

---

## ISSUE #08 — Extend `conftest.py` with shared fixtures for Day 2 modules

**Labels**: `tests` · `good first issue`
**Day**: 1
**Branch**: `feature/08-conftest-fixtures`
**Difficulty**: Beginner-friendly

### Context

Day 2 modules (visualizer, comparator, CLI) need realistic fake data to test
against. Centralising these fixtures in `conftest.py` prevents duplication
across test files.

### What to add to `tests/conftest.py`

- `sample_results` fixture — a fake `results` dict mimicking what `core.py`
  produces, with keys: `df_raw`, `df_clean`, `topics`, `sentiment`, `temporal`,
  `users`, `group_name`.
- `sample_topics` fixture — a `group_topics` DataFrame with 3 topics.
- `sample_sentiment` fixture — a sentiment dict with `df`, `by_user`, `global`.
- `sample_temporal` fixture — a temporal dict with `timeline`, `hourly_heatmap`,
  `peak_hour`, `peak_day`.
- `mock_analyzer` fixture — a `MagicMock` that mimics `WhatsAppAnalyzer`
  with `_results`, `_group_name`, `raw_data()`, `users()`, `topics()` methods.

### Rules

- All fixtures must be buildable with pandas and standard library only.
- No real chat data — use the fake author names already in `conftest.py`.
- Follow all conventions in `CONTRIBUTING.md`.

### Definition of done

- [ ] All fixtures added to `conftest.py`.
- [ ] Running `python -m pytest tests/` still passes after the change.

---
---

## DAY 2 — Assembly modules

> Day 2 issues depend on Day 1 modules being merged.
> Start as soon as the relevant branches land on `main`.

---

## ISSUE #09 — Extend `topic_classifier.py` with BERTopic

**Labels**: `module` · `nlp`
**Day**: 2
**Branch**: `feature/09-topic-bertopic`
**Depends on**: #01 merged

### What to implement

Add to `whatsapp_classifier/topic_classifier.py`:

- Private method `_fit_bertopic(df: pd.DataFrame) -> dict`.
- Activated when `method="bertopic"`.
- Guard with `try/except ImportError` — raise `RuntimeError` with install
  instructions if BERTopic is missing.
- Output format must be identical to the LDA path (same dict structure).

### Rules

- LDA path must remain unaffected.
- BERTopic never loaded in tests — mock entirely.
- Follow all conventions in `CONTRIBUTING.md`.

### Definition of done

- [ ] BERTopic path implemented and guarded.
- [ ] Existing LDA tests (#01) still pass.
- [ ] New tests for the BERTopic path with mocks.

---

## ISSUE #10 — Implement `visualizer.py`

**Labels**: `module`
**Day**: 2
**Branch**: `feature/10-visualizer`
**Depends on**: #08 merged (fixtures)

### What to implement

File: `whatsapp_classifier/visualizer.py`

Class `Visualizer`:

Individual plot methods (all return `matplotlib.figure.Figure`):
- `plot_topic_distribution(results: dict) -> Figure`
- `plot_wordcloud(results: dict, topic_id: int) -> Figure`
- `plot_sentiment_timeline(results: dict) -> Figure`
- `plot_user_activity(results: dict) -> Figure`
- `plot_hourly_heatmap(results: dict) -> Figure`

Report methods:
- `generate_report(results: dict, output_dir: Path) -> Path`
  Embeds all charts as base64 PNG in a self-contained HTML file.
  Uses a Jinja2 template string defined as a constant in the file.
  Returns the path to the written file.
- `generate_comparison_report(analyzers: list, output_dir: Path) -> Path`

### Rules

- HTML must open in a browser without internet access (no external CDN).
- Use `matplotlib`, `seaborn`, `wordcloud`. Plotly optional and guarded.
- Mock `matplotlib` entirely in tests — no rendering.
- Follow all conventions in `CONTRIBUTING.md`.

### Definition of done

- [ ] All plot methods implemented.
- [ ] `report.html` generated and opens correctly in a browser.
- [ ] `tests/test_visualizer.py` — matplotlib patched, runs alone.

---

## ISSUE #11 — Complete `comparator.py`

**Labels**: `module`
**Day**: 2
**Branch**: `feature/11-comparator`
**Depends on**: #08 merged (fixtures)

### What to implement

Complete the method stubs in `whatsapp_classifier/comparator.py`:

- `compare_topics() -> pd.DataFrame`
  Pivot table: groups as rows, topic labels as columns, weights as values.
- `compare_activity() -> pd.DataFrame`
  One row per group: `nb_messages`, `nb_participants`, `msgs_per_day`,
  `period_start`, `period_end`.
- `compare_sentiment() -> pd.DataFrame`
  One row per group: `sentiment_mean`, `pos_pct`, `neg_pct`.
- `common_users() -> pd.DataFrame`
  Authors present in more than one group, with their group list.
- `report(output: Path) -> Path`
  Delegates to `Visualizer.generate_comparison_report()`.

### Rules

- Do not re-implement analysis logic — read from `az._results`.
- Mock `WhatsAppAnalyzer` with `MagicMock` in tests.
- Follow all conventions in `CONTRIBUTING.md`.

### Definition of done

- [ ] All methods implemented.
- [ ] `tests/test_comparator.py` uses `mock_analyzer` fixture — runs alone.
- [ ] `python -m pytest tests/test_comparator.py` passes.

---

## ISSUE #12 — Complete `cli.py`

**Labels**: `cli` · `good first issue`
**Day**: 2
**Branch**: `feature/12-cli`
**Depends on**: #04 merged

### What to implement

Complete the three Click commands in `whatsapp_classifier/cli.py`:

`analyze` command:
- Run the full pipeline.
- Print a summary to the console with `rich`:
  group name, message count, author count, date range, top topics.
- Handle errors gracefully: catch exceptions, print a readable message, exit 1.

`compare` command:
- Run `GroupComparator`, print a side-by-side comparison table with `rich`.

`serve` command:
- Launch `app.py` via `subprocess` + `streamlit run`.

### Rules

- No raw Python traceback must ever reach the user — catch and format all errors.
- Use `click.testing.CliRunner` in tests — no real files needed.
- Follow all conventions in `CONTRIBUTING.md`.

### Definition of done

- [ ] All three commands implemented.
- [ ] `whatsapp-classifier analyze --input chat.zip --topics 5` runs end-to-end.
- [ ] `tests/test_cli.py` uses `CliRunner` — runs alone.
- [ ] `python -m pytest tests/test_cli.py` passes.

---

## ISSUE #13 — Implement `media_analyzer.py`

**Labels**: `module` · `nlp`
**Day**: 2
**Branch**: `feature/13-media-analyzer`
**Difficulty**: Advanced

### What to implement

File: `whatsapp_classifier/media_analyzer.py`

Class `MediaAnalyzer`:

- `analyze(media_dir: Path) -> dict`
  - `"stats"` — DataFrame: `file_type`, `count`, `total_size_mb`.
  - `"transcriptions"` — DataFrame: `file_path`, `text`
    (empty DataFrame when Whisper not available).

Routing:
- `.opus` / `.ogg` / `.mp3` → Whisper transcription if available.
- `.mp4` → extract audio with `ffmpeg` (via `subprocess`), then Whisper.
- `.jpg` / `.png` / `.webp` → stats only (count + size).

When Whisper is not installed:
- Log a warning.
- Return empty `"transcriptions"` DataFrame.
- `"stats"` must always be computed regardless.

### Rules

- Guard `import whisper` with `try/except ImportError`.
- Check `shutil.which("ffmpeg")` before using ffmpeg — log warning if absent.
- Mock Whisper and ffmpeg entirely in tests.
- Follow all conventions in `CONTRIBUTING.md`.

### Definition of done

- [ ] `media_analyzer.py` implemented.
- [ ] Stats always computed, transcriptions conditional on Whisper.
- [ ] `tests/test_media_analyzer.py` — Whisper and ffmpeg mocked, runs alone.

---

## ISSUE #14 — Implement `app.py` (Streamlit interface)

**Labels**: `ui` · `good first issue`
**Day**: 2
**Branch**: `feature/14-streamlit-app`
**Difficulty**: Beginner-friendly

### What to implement

Replace the current placeholder in `whatsapp_classifier/app.py`:

Sidebar controls:
- File uploader (`.zip` or `.txt`).
- `n_topics` slider (2–15, default 5).
- `min_words` slider (1–10, default 3).
- Language selector: Auto / French / English.
- Toggles: Anonymise, Enable media (hidden if Whisper not installed).

Main panel (shown after "Run analysis" button):
- Progress bar while pipeline runs.
- Topic distribution chart (`st.pyplot`).
- Wordcloud per topic (`st.image`).
- Per-user activity bar chart.
- Download button for the HTML report.

Error handling:
- Display `st.error()` for invalid files — no raw traceback.

### Rules

- App must work without `[media]` extras.
- Follow all conventions in `CONTRIBUTING.md`.

### Definition of done

- [ ] `whatsapp-classifier serve` opens the app in the browser.
- [ ] Uploading a `.zip` and clicking Run produces visible results.
- [ ] Download button produces a valid HTML report.

---

## ISSUE #15 — Write the three Jupyter notebooks

**Labels**: `exploration` · `good first issue`
**Day**: 2
**Branch**: `feature/15-notebooks`
**Difficulty**: Beginner-friendly

### What to implement

**`notebooks/01_exploration.ipynb`**
- Load an anonymised export.
- Run step by step: `parse()` → `clean()` → `analyze()`.
- Display raw and cleaned DataFrames.
- Show topic distribution chart and a wordcloud.
- Add markdown cells explaining each step.

**`notebooks/02_analyse_utilisateurs.ipynb`**
- Show `az.user("Name").summary()` for each participant.
- Plot activity heatmap and sentiment timeline per user.

**`notebooks/03_comparaison_groupes.ipynb`**
- Load two group exports.
- Run `GroupComparator` and display comparison tables.
- Side-by-side topic distribution chart.

### Rules

- Anonymised or synthetic data only — no real names or phone numbers.
- Each notebook must run top-to-bottom without errors.
- Do not commit raw chat files.

### Definition of done

- [ ] All three notebooks run without errors.
- [ ] Charts render inline.
- [ ] No personal data committed.

---
---

## DAY 3 — Integration and release

---

## ISSUE #16 — Complete `core.py` and wire the full pipeline

**Labels**: `module` · `integration`
**Day**: 3 — pair programming Dev A + Dev B
**Branch**: `feature/16-core-wiring`
**Depends on**: all Day 1 and Day 2 modules merged on `main`

### What to implement

Complete `WhatsAppAnalyzer` in `whatsapp_classifier/core.py`:

- `parse()` — call `Loader` then `Parser`, store result in `self._results`.
- `clean(lang)` — call `Cleaner`, store cleaned DataFrame.
- `analyze(topics, sentiment, temporal, media)` — call each analysis module
  in order, collect results. All module imports must be **lazy** (inside
  the method body) to prevent circular imports.
- `report(output)` — delegate to `Visualizer.generate_report()`.
- `to_csv(output)` — export enriched DataFrame.
- Verify `UserView` methods delegate correctly to `UserAnalyzer`.
- Update `__init__.py` to expose `WhatsAppAnalyzer` and `GroupComparator`.

### Rules

- `core.py` must contain **no NLP logic** — it only orchestrates.
- All sub-module imports inside method bodies (lazy imports).
- `tests/test_core.py` — mock all sub-modules with `MagicMock`.
- Follow all conventions in `CONTRIBUTING.md`.

### Definition of done

- [ ] Full pipeline runs: `az.parse().clean().analyze().report()`.
- [ ] `az.user("Name").summary()` returns a populated dict.
- [ ] `tests/test_core.py` passes in isolation.

---

## ISSUE #17 — Integration tests and final merge

**Labels**: `tests` · `integration`
**Day**: 3 — pair programming Dev C + Dev D
**Branch**: `feature/17-integration`
**Depends on**: #16 merged

### What to do

1. Run the full test suite — fix any failing tests:
   ```bash
   python -m pytest -v
   ```

2. Run an end-to-end integration test on a real (anonymised) export:
   ```python
   from whatsapp_classifier import WhatsAppAnalyzer
   az = WhatsAppAnalyzer("data/raw/test_group.zip")
   az.parse().clean().analyze(topics=5, sentiment=True)
   az.report(output="data/processed/rapport/")
   ```

3. Fix any integration bugs found (open `bug` issues for non-trivial fixes).

4. Resolve any merge conflicts between Day 2 branches.

5. Verify CLI works end-to-end:
   ```bash
   whatsapp-classifier analyze --input data/raw/test_group.zip --topics 5
   ```

### Definition of done

- [ ] `python -m pytest` — all tests passing.
- [ ] End-to-end pipeline produces a valid HTML report.
- [ ] CLI `analyze` command works on a real export.
- [ ] No open merge conflicts on `main`.

---

## ISSUE #18 — Live demo on a real WhatsApp export

**Labels**: `exploration`
**Day**: 3 — whole team
**Branch**: none (demo run locally, no commit needed)

### What to do

Run the full pipeline on a real (anonymised) WhatsApp export as a team:

1. Each contributor runs on their own machine:
   ```bash
   whatsapp-classifier analyze --input data/raw/my_group.zip --topics 5 --sentiment
   ```
2. Open the generated `report.html` in a browser.
3. Check that all sections render correctly:
   - Topic distribution chart.
   - Wordcloud per topic.
   - Per-user activity.
   - Sentiment timeline.
4. Run the Streamlit app:
   ```bash
   whatsapp-classifier serve
   ```
5. Note any visual or functional bugs → open `bug` issues if found.

### Definition of done

- [ ] Pipeline runs without errors on at least 2 real exports.
- [ ] HTML report opens correctly in a browser.
- [ ] Streamlit app launches without errors.

---

## ISSUE #19 — Release checklist v0.1.0

**Labels**: `docs`
**Day**: 3 — whole team
**Branch**: `feature/19-release`

### What to do

- [ ] All issues #01–#17 closed.
- [ ] `python -m pytest` — 100% passing.
- [ ] `README.md` module status table updated (all Done).
- [ ] `CONTRIBUTING.md` status table updated.
- [ ] `pyproject.toml` version confirmed as `0.1.0`.
- [ ] Git tag created:
  ```bash
  git tag v0.1.0
  git push origin v0.1.0
  ```
- [ ] Milestone **v0.1.0** closed on GitHub.