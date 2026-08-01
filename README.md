# WhatsApp Message Notification Router

A hybrid rule-engine + Gemini-powered system that classifies every incoming WhatsApp
message into **`notify`**, **`digest`**, or **`mute`**, using text, image, and voice-note
content plus per-user behavioral history.

Built for the **HackerRank Orchestrate** hackathon. Full task spec, schema, and
grading criteria are in [`problem_statement.md`](./problem_statement.md).

---

## 1. How It Works

```
dataset/messages.csv
        │
        ▼
data_loader.py          → loads messages + users, groups, business_accounts,
                           message_history, message_events, images, voice_notes
        │
        ▼
feature_engineering.py  → computes per-message signals: urgency score, scam/promo
                           score, sender relationship, quiet-hours, past engagement
        │
        ▼
router.py                → NotificationRouter
   ├─ rule engine fires first (fast, deterministic)
   │     confidence ≥ 0.90 → use rule engine result directly
   │     confidence <  0.90 → escalate to Gemini
   ▼
gemini_client.py         → calls Gemini with the message + multimodal context
                            (multimodal.py extracts image/audio content first)
   ├─ Gemini available + key set  → "gemini_fusion" decision
   └─ Gemini unavailable/fails    → "rule_engine_fallback" (rule result reused)
        │
        ▼
main.py                  → validates schema/contract, writes output.csv
```

Every prediction is tagged internally with a `decision_type`
(`rule_engine`, `gemini_fusion`, or `rule_engine_fallback`) and a summary is
printed at the end of the run so you can see how many messages actually used
Gemini vs. the fallback rules.

### Files

| File | Responsibility |
|---|---|
| `code/config.py` | Paths, allowed `action`/`message_type` values, required output columns |
| `code/data_loader.py` | Reads all CSVs in `dataset/` into a single `Dataset` object |
| `code/feature_engineering.py` | Derives urgency/scam/relationship/quiet-hours features per message |
| `code/multimodal.py` | Extracts/describes image and voice-note content for messages with media |
| `code/gemini_client.py` | Wraps the Gemini API call; reads `GEMINI_API_KEY` from env; fails safe |
| `code/router.py` | `NotificationRouter` — decides rule engine vs. Gemini per message |
| `code/main.py` | End-to-end pipeline entry point; validates and writes `output.csv` |
| `code/evaluation/main.py` | Scores predictions against `dataset/sample_messages.csv` (ground truth) |

---

## 2. Setup

### Requirements
- Python 3.10+
- A Gemini API key (optional — the pipeline runs without one, but only uses the rule
  engine). Get one at https://aistudio.google.com/apikey

### Install

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

`requirements.txt` includes `google-genai`, which is required for Gemini calls to
work at all. If it's missing, `gemini_client.py` fails its import silently inside a
try/except and every message quietly falls back to the rule engine — the pipeline
still runs, but Gemini is never actually called. If you're on an older checkout,
confirm it's installed:

```bash
pip install google-genai
```

### Set your Gemini API key

The key is read from the `GEMINI_API_KEY` environment variable — never hardcode it
in `config.py` or `gemini_client.py`.

**macOS/Linux (bash/zsh):**
```bash
export GEMINI_API_KEY="your_actual_key_here"
```

**Windows PowerShell:**
```powershell
$env:GEMINI_API_KEY = "your_actual_key_here"
```

This only lasts for the current terminal session — set it again in any new window,
or use a `.env` file (see below).

**Optional: `.env` file** — `python-dotenv` is already a dependency. If you'd rather
not re-export the key every session, create a `.env` file in the project root:
```
GEMINI_API_KEY=your_actual_key_here
```
(Confirm `main.py`/`config.py` actually calls `load_dotenv()` before relying on this —
add the call yourself if it isn't already wired in.)

---

## 3. Running

```bash
cd code
python main.py
```

On success you'll see a log line like:

```
Decision Fusion Summary: {'rule_engine': 22, 'gemini_fusion': 88, 'rule_engine_fallback': 0}
```

- `rule_engine` — high-confidence (≥0.90) messages the rule engine handled directly
- `gemini_fusion` — lower-confidence messages resolved via a live Gemini call
- `rule_engine_fallback` — Gemini was requested but unavailable/failed, so the rule
  engine's result was reused

If `GEMINI_API_KEY` isn't set (or `google-genai` isn't installed), you'll instead see:
```
gemini_client: GEMINI_API_KEY not set. Gemini API calls will fall back to rule engine.
```
and every low-confidence message lands in `rule_engine_fallback` — the run still
completes and produces a valid `output.csv`, just without any LLM reasoning.

Output is written to both:
- `output.csv` (project root)
- `dataset/output.csv`

### Evaluate against sample ground truth

```bash
cd code
python -m evaluation.main
```

Prints accuracy, macro/micro precision/recall/F1, and confusion matrices for both
`action` and `message_type`, scored against `dataset/sample_messages.csv`.

---

## 4. Output Format

`output.csv` has one row per message in `dataset/messages.csv`:

| Column | Meaning |
|---|---|
| `message_id` | Incoming message ID |
| `action` | `notify`, `digest`, or `mute` |
| `message_type` | Best-fit category (see `config.py` for allowed values) |
| `reason` | Short human-readable explanation |
| `confidence` | Float between `0.0` and `1.0` |
| `evidence_message_ids` | Semicolon-separated historical message IDs used as evidence, or `none` |

`main.py` validates this contract automatically before writing the file (correct
columns, row count matches input, only allowed `action`/`message_type` values,
confidence in range, no nulls in `evidence_message_ids`) and raises immediately if
anything is off.

---

## 5. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Decision Fusion Summary` always shows `'gemini_fusion': 0` | `GEMINI_API_KEY` not set in the current shell | `export`/`$env:` it, then rerun in the *same* terminal |
| Same as above, key is set | `google-genai` not installed | `pip install google-genai` (should already be in `requirements.txt`) |
| `ModuleNotFoundError: No module named 'google.genai'` | Package not installed in the active venv | Confirm you activated `.venv` before `pip install`, and that `pip` points at the same interpreter as `python code/main.py` |
| `FileNotFoundError: Messages file not found or empty` | Running from the wrong working directory | `cd code` before running `main.py`, or check `config.py` paths |
| `ValueError: Columns mismatch!` / `Invalid action values found` | Router is producing values outside the allowed contract | Check `config.REQUIRED_OUTPUT_COLUMNS` / `ALLOWED_ACTIONS` against what `router.py` emits |

---

## 6. Repository Layout

```text
.
├── AGENTS.md                    # Rules for AI coding tools + transcript logging
├── problem_statement.md         # Full challenge statement
├── README.md                    # You are here
├── requirements.txt
├── output.csv                   # Generated predictions
├── code/
│   ├── config.py
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── multimodal.py
│   ├── gemini_client.py
│   ├── router.py
│   ├── main.py
│   └── evaluation/
│       └── main.py
└── dataset/
    ├── messages.csv
    ├── output.csv
    ├── sample_messages.csv
    ├── users.csv
    ├── groups.csv
    ├── group_members.csv
    ├── business_accounts.csv
    ├── user_business_history.csv
    ├── message_history.csv
    ├── message_events.csv
    ├── images.csv
    ├── voice_notes.csv
    └── media/
        ├── images/
        └── audio/
```
