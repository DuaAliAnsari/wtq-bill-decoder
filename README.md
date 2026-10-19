# Utility Bill Decoder (WTQ 2026 Build Track)

Reads an electricity bill image (K-Electric, LESCO or IESCO) and

* **Level 1** - extracts the bill's details into one JSON object per bill, and
* **Level 2** - answers customer questions about the bill in plain English.

**Runtime:** Python 3.12 (3.11 or newer works). **Dependencies:** `requests`, `python-dotenv`, `Pillow` (exact versions in `requirements.txt`).
Model calls are made over plain REST, so no provider SDK is needed.

## Setup

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate        macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # Windows: copy .env.example .env
# open .env and replace API_KEY=your-api-key-here with your real key
```

`.env` holds your key and is never packaged. `.env.example` shows every variable the code reads:

| Variable | Meaning | Default |
|---|---|---|
| `MODEL_PROVIDER` | `google`, `anthropic` or `openai` | `google` |
| `MODEL_NAME` | one of the allowed model IDs (others are refused at start-up) | `gemini-3.5-flash-lite` |
| `API_KEY` | provider API key | - |
| `TEMPERATURE`, `SEED` | determinism settings | `0`, `7` |
| `REQUESTS_PER_MINUTE` | client-side throttle (free Gemini tier allows 15) | `12` |
| `MAX_RETRIES`, `REQUEST_TIMEOUT_SECONDS`, `CACHE_DIR` | robustness settings | `6`, `120`, `.cache` |

## Run on a folder of bills (generates both CSV files)

```bash
python main.py run --bills test/bills --level1 test/csv/level1.csv --level2 test/csv/level2.csv --out output
```

This reads every `bill_id` from the two template CSVs, finds `bills/<bill_id>.png|.jpg`, and writes
`output/level1.csv` (one-line JSON per bill, quotes doubled by the CSV writer) and `output/level2.csv`
(`bill_id, question_id, question, answer`) in the template's row order. It exits with status 1 and lists the
bills if any needed a fallback value.

Other commands:

```bash
# Demo mode: decode one bill and answer questions about it
python main.py bill train/bills/KESC_0008.png --ask "How much of my bill is taxes?" --ask "How many months went above 200 units?"

# Build templates for the training bills (12 sample questions) and run on them
python main.py make-templates --bills train/bills --out train/csv
python main.py run --bills train/bills --level1 train/csv/level1.csv --level2 train/csv/level2.csv --out train/output

# Score Level 1 against gold JSON (tests/gold holds the official KESC_0008 answer)
python main.py evaluate --pred train/output/level1.csv

# Check results and build the submission ZIP (CSV shape, empty cells, leaked keys, 15 MB limit)
python main.py package --out-dir output --zip submission.zip --demo demo.mp4

# Unit tests (offline, use a scripted fake model; no API key needed)
python -m unittest discover -s tests -t .
```

## How it works

For every bill the pipeline makes four focused model calls, and code does everything that must be exact:

1. **Level 1 extraction** (image -> JSON). The prompt carries the competition's rules, charge/tax type definitions
   and layout hints for each provider. `schema.py` then deterministically cleans the reply: numbers, `CR` -> negative,
   date formats, enum mapping, subsidy sign, dropping zero/blank lines, fixed key order. If key fields are unreadable,
   one repair call is made with the specific problems listed.
2. **Context extraction** (image -> JSON). Usage history, billing history, the calculation table, meter registers, the
   printed energy/tax split and message-board text. This is what Level 2 questions such as "how many months above
   200 units" depend on.
3. **Planning** (text). For each question the model states its interpretation and writes small arithmetic expressions
   (e.g. `sum(1 for u in prior_units if u > 200)`). `calc.py` evaluates them in a **whitelisted AST sandbox** (no
   imports, attributes or builtins).
4. **Answering** (text). The model writes the final answers from the bill fields, pre-computed facts (tax share,
   averages, highest/lowest month, late-payment extra, ...) and the verified calculation results, so figures are
   computed by code rather than by the model.

### Reproducibility ("the same answer every time")

* Temperature 0 and a fixed seed (where the provider supports them).
* Every model response is cached on disk by a hash of model + prompts + image, so re-running gives byte-identical
  CSVs and costs no extra API calls (`--no-cache` forces fresh calls).
* All formatting and numeric clean-up is code, not model output.
* Image pre-processing is deterministic.

### Reliability

* Retries with exponential backoff on 429/5xx (honours `Retry-After`); a client-side throttle keeps the free tier
  under its per-minute limit; a hard daily-quota error stops cleanly instead of burning retries.
* Each stage fails independently: a failed answer step never discards the Level 1 result; one unreadable bill
  never stops the others.
* Output files are always complete: a failed bill gets an all-null Level 1 JSON and plain fallback sentences, never
  an empty cell. Re-running resumes quickly because finished calls are cached.
* `logs/run_report.json` lists per-bill warnings (for example "sum of charges differs from total_charges").

### Interpretation choices (where the guide leaves room)

* Several meter registers: the first row gives `previous_reading` / `current_reading`; `units_consumed` is the figure
  the bill labels as the billed units.
* LESCO/IESCO print only a summary box: "Total Electricity Charges" becomes one `energy` entry, a printed Subsidies
  figure becomes a negative `subsidy` entry, `total_charges` is the "Net Electricity Charges", and a combined "Taxes"
  figure goes to `total_taxes` with `taxes: []`.
* Text such as "NOT TO BE PAID" is not a number, so `payable_after_due_date` is `null`.

## AI usage

* **Models called by the solution:** Gemini 3.5 Flash-Lite (`gemini-3.5-flash-lite`) via the Google Gemini API
  (Google AI Studio free tier). The code can also run Claude Haiku 4.5, Gemini 3.1 Flash-Lite, GPT-4.1 mini or
  GPT-5 mini by changing `MODEL_PROVIDER` / `MODEL_NAME`; no other model is accepted.
* **APIs / services:** Google Gemini `generateContent` REST API only.
* **Invoice/receipt extraction services:** none. **OCR libraries:** none (the vision model reads the image directly).
* **AI tools used to write the code:** Claude (Anthropic's chat assistant) helped design and write this project.
  The author reviewed and tested the result.

## Project layout

```
main.py                 command line entry point
billdecoder/config.py   environment config, allowed-model check
billdecoder/llm.py      Google/Anthropic/OpenAI REST client: cache, throttle, retries, JSON parsing
billdecoder/imaging.py  load, orient, resize bill images
billdecoder/prompts.py  every prompt
billdecoder/schema.py   Level 1 normalisation and validation
billdecoder/extract.py  Level 1 + context extraction
billdecoder/context.py  usage/billing history normalisation and derived facts
billdecoder/calc.py     safe expression evaluator
billdecoder/answer.py   Level 2 planning and answering
billdecoder/pipeline.py end-to-end run, fallbacks, run report
billdecoder/csv_io.py   template reading and result writing
billdecoder/evaluate.py scoring against gold JSON
billdecoder/package.py  pre-submission checks and ZIP builder
tests/                  offline unit tests
```

## Troubleshooting

* `HTTP 404` / model not found: check `MODEL_NAME` (the Gemini IDs are `gemini-3.5-flash-lite` and `gemini-3.1-flash-lite`).
* `HTTP 429`: the free tier limit was reached; the client waits and retries. Lower `REQUESTS_PER_MINUTE` if it persists.
  A daily-quota error stops the run; re-run later and cached bills are reused.
* Answers or JSON look odd with temperature 0: try `TEMPERATURE=1` in `.env` (Google recommends the default for some Gemini 3 models), then re-run with `--no-cache`.
