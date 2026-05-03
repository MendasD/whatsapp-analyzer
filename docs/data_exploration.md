# Data Exploration Report

Corpus statistics computed from anonymised WhatsApp exports.
No real names or phone numbers appear in this document.

---

## Group: ENSAE_Group_1 (iOS)

| Metric | Value |
|--------|-------|
| Platform | iOS |
| Message count | 10,885 |
| Author count | 322 |
| Date range | 2018-10-10 → 2026-03-27 |

### Message type distribution

| Type | Count | Share |
|------|-------|-------|
| media | 2,802 | 25.7% |
| text | 8,083 | 74.3% |

### Estimated language distribution

| Language | Share |
|----------|-------|
| French | 73.1% |
| Unknown | 19.1% |
| English | 7.7% |
| Wolof | 0.1% |

### Parser validation

✅ All lines parsed successfully.

---

## Known edge cases handled by the parser

| Issue | Status |
|-------|--------|
| iOS `[DD/MM/YYYY, HH:MM:SS]` format | ✅ Supported |
| Android `DD/MM/YYYY, HH:MM -` format | ✅ Supported |
| Multi-line messages | ✅ Continuation lines appended |
| Media attachments (`<attached:…>`) | ✅ Classified as `media` |
| POLL / OPTION messages | ✅ Classified as `media` |
| Emoji in author names | ✅ Preserved then anonymised |
| Unicode zero-width chars (U+200E) | ✅ Stripped before matching |
| Wolof / French mixed content | ✅ Language heuristic applied |
| Arabic characters | ✅ Regex is Unicode-aware |
| Phone numbers as author labels | ✅ Anonymised via regex |
| `<This message was edited>` tag | ✅ Classified as `media` |
| Disappearing-message events | ✅ Classified as `system` |

---

## Methodology

- **Anonymisation**: author names → `User_NNN` pseudonyms at parse time.
- **Language detection**: keyword heuristic, no external library.
- **Parser API**: `Parser().parse(chat_path)` → `list[Message]`.

*Generated automatically by `exploration.ipynb`.*