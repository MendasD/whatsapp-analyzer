## Summary

<!-- One sentence describing what this PR does. -->

Closes #<!-- issue number -->

## Changes

<!-- List the files modified or created. -->

-
-

## Checklist

### Code quality
- [ ] All code and comments are in **English**
- [ ] No decorative comment separators (`#---`, `#===`, etc.) — single `#` only
- [ ] Type hints on every public function signature
- [ ] Docstrings on every public class and non-trivial function
- [ ] No `print()` statements — `logging.getLogger(__name__)` used instead
- [ ] No dead code, no commented-out blocks

### Tests
- [ ] Test file written for this module: `tests/test_<module>.py`
- [ ] All external dependencies mocked — no real NLP models called in tests
- [ ] Module runs in **isolation**: `python -m pytest tests/test_<module>.py`
- [ ] Full suite still passes: `python -m pytest`

### Dependencies
- [ ] No circular imports introduced (see dependency table in `CONTRIBUTING.md`)
- [ ] Heavy external imports are **lazy** (inside method bodies, not at module level)

### Data & privacy
- [ ] No real WhatsApp exports committed (`data/` is in `.gitignore`)
- [ ] No real phone numbers or full names in test fixtures or docs

### Branch & commit
- [ ] Branch named `feature/<issue-number>-short-description`
- [ ] Commits follow the format: `feat|fix|test|docs|refactor(<scope>): message`
- [ ] PR targets `main` and is up to date with it
